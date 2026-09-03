"""TP=2 (head/expert sharding) tests for glm5_next (GLM-5.3-Flash).

Single-process simulations, same pattern as qwen4_exp/test_tp.py: each rank's
layer/loader is built under a patched ``freetoken.distributed.info._TP_INFO`` and
the all-reduce is a no-op stub, so the two partials are summed by hand -- exactly
what the real all-reduce computes. Covered:

* ``_shard_dense_weight`` row/col splits for every sharded dense tensor: the KDA
  fused ``in_proj`` mixed layout (q|k|v|b head-sharded, the low-rank f_a|g_a
  replicated), the conv1d channel split, per-head gate params (A_log/dt_bias,
  f_b/g_b), MLA q_b/kv_b column shards, o/down row shards, MLP gate/up, the
  div_ceil vocab shards, and the replicated pass-through (latents, indexer, mHC,
  norms, router);
* an end-to-end ``iter_weights`` run over a synthetic checkpoint at tp=2;
* the gated MLP (CPU) plus the KDA and DSA attention ops (CUDA): two TP halves
  summed reproduce the full layer;
* the full-model state-dict shape contract at tp=2 (what load_state_dict enforces
  at boot).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.models.glm5_next.weight import _shard_dense_weight

HIDDEN = 256
KDA_H, KDA_D, CONV_K = 4, 128, 4  # KDA heads x head_dim (D=128: the kernels specialize on it)
P = KDA_H * KDA_D
MLA_H, QK_DIM, V_DIM, Q_LORA, KV_LORA = 4, 32, 32, 64, 32
IDX_H, IDX_D, KPOOL = 4, 8, 4
VOCAB = 1000
INTER, MOE_INTER, EXPERTS, TOPK = 64, 32, 8, 2
N_STREAMS = 4  # mhc hc_mult
_MIX = 2 * N_STREAMS + N_STREAMS * N_STREAMS


def _patch_tp(monkeypatch, rank: int, size: int) -> None:
    import freetoken.distributed.info as dist_info

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank=rank, size=size))


@pytest.fixture
def noop_comm(monkeypatch):
    """all_reduce/all_gather as identity: the test sums the per-rank partials itself."""
    from freetoken.distributed import DistributedCommunicator
    from freetoken.distributed.impl import DistributedImpl

    class _NoOp(DistributedImpl):
        def all_reduce(self, x):
            return x

        def all_gather(self, x):
            return x

    monkeypatch.setattr(DistributedCommunicator, "plugins", [_NoOp()])


def _text_config() -> dict:
    return {
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "num_hidden_layers": 2,
        "num_attention_heads": MLA_H,
        "num_key_value_heads": MLA_H,
        "vocab_size": VOCAB,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": False,
        # MLA (NoPE)
        "q_lora_rank": Q_LORA,
        "kv_lora_rank": KV_LORA,
        "qk_nope_head_dim": QK_DIM,
        "qk_rope_head_dim": 0,
        "v_head_dim": V_DIM,
        "mla_use_nope": True,
        # DSA indexer + kpool
        "index_n_heads": IDX_H,
        "index_head_dim": IDX_D,
        "index_topk": 32,
        "indexer_types": ["full", "full"],
        "indexer_rope_interleave": True,
        "index_kpool": KPOOL,
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
        # KDA
        "linear_attn_config": {
            "num_heads": KDA_H,
            "head_dim": KDA_D,
            "short_conv_kernel_size": CONV_K,
            "gate_lower_bound": -5.0,
        },
        # layout: layer 0 KDA + dense MLP, layer 1 DSA + sparse MoE
        "layer_types": ["linear_attention", "deepseek_sparse_attention"],
        "mlp_layer_types": ["dense", "sparse"],
        "first_k_dense_replace": 1,
        # mHC (checkpoint spellings)
        "mhc": True,
        "hc_mult": N_STREAMS,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        # MoE
        "n_routed_experts": EXPERTS,
        "num_experts_per_tok": TOPK,
        "n_shared_experts": 1,
        "moe_intermediate_size": MOE_INTER,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "n_group": 1,
        "topk_group": 1,
        "swiglu_limit": 10.0,
        "attention_bias": False,
        "model_type": "glm5_next_text",
    }


def _hf_config():
    from freetoken.utils.hf import RawConfigShim

    return RawConfigShim(
        {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "model_type": "glm5_next",
            "text_config": _text_config(),
        }
    )


def _args():
    from freetoken.models.glm5_next.config import parse_config

    return parse_config(_hf_config()).glm5_args


def _kda_op_config():
    """The duck-typed ModelConfig the KDA op reads (no MoE/KV fields needed)."""
    return SimpleNamespace(glm5_args=_args(), attn_quant="none")


# ======================================================================================
# _shard_dense_weight
# ======================================================================================


def test_shard_kda_in_proj_head_shard_plus_replicated_lowrank():
    args = _args()
    h, d, p = KDA_H, KDA_D, P
    rows = 3 * p + h + 2 * d
    full = torch.arange(rows, dtype=torch.float32).unsqueeze(1).expand(-1, 3)
    for rank in (0, 1):
        got = _shard_dense_weight(
            "model.layers.0.self_attn.in_proj.weight", full, rank=rank, size=2, args=args
        )
        hp, hh = p // 2, h // 2
        want = torch.cat(
            [
                full[rank * hp : (rank + 1) * hp],  # q heads
                full[p + rank * hp : p + (rank + 1) * hp],  # k heads
                full[2 * p + rank * hp : 2 * p + (rank + 1) * hp],  # v heads
                full[3 * p + rank * hh : 3 * p + (rank + 1) * hh],  # b (per-head beta)
                full[3 * p + h :],  # f_a | g_a: low-rank bottlenecks, replicated
            ]
        )
        assert torch.equal(got, want)
        # the local layout the op splits by: [conv_in | b | f_a | g_a]
        assert got.shape[0] == 3 * hp + hh + 2 * d


def test_shard_kda_conv1d_matches_in_proj_qkv_rows():
    args = _args()
    conv = torch.arange(3 * P, dtype=torch.float32).view(-1, 1, CONV_K)
    for rank in (0, 1):
        got = _shard_dense_weight(
            "model.layers.0.self_attn.conv1d.weight", conv, rank=rank, size=2, args=args
        )
        hp = P // 2
        want = torch.cat(
            [
                conv[rank * hp : (rank + 1) * hp],
                conv[P + rank * hp : P + (rank + 1) * hp],
                conv[2 * P + rank * hp : 2 * P + (rank + 1) * hp],
            ]
        )
        assert torch.equal(got, want)


def test_shard_kda_gate_params_and_projections():
    args = _args()
    a_log = torch.arange(KDA_H, dtype=torch.float32)
    got = _shard_dense_weight("model.layers.0.self_attn.A_log", a_log, rank=1, size=2, args=args)
    assert torch.equal(got, a_log[KDA_H // 2 :])
    dt = torch.arange(P, dtype=torch.float32)
    got = _shard_dense_weight("model.layers.0.self_attn.dt_bias", dt, rank=0, size=2, args=args)
    assert torch.equal(got, dt[: P // 2])
    for proj in ("f_b_proj", "g_b_proj"):
        w = torch.arange(P * KDA_D, dtype=torch.float32).view(P, KDA_D)
        got = _shard_dense_weight(
            f"model.layers.0.self_attn.{proj}.weight", w, rank=1, size=2, args=args
        )
        assert torch.equal(got, w[P // 2 :])
    o = torch.arange(HIDDEN * P, dtype=torch.float32).view(HIDDEN, P)
    got = _shard_dense_weight(
        "model.layers.0.self_attn.o_proj.weight", o, rank=0, size=2, args=args
    )
    assert torch.equal(got, o[:, : P // 2])


def test_shard_dsa_projections():
    args = _args()
    q_b = torch.arange(MLA_H * QK_DIM * Q_LORA, dtype=torch.float32).view(MLA_H * QK_DIM, Q_LORA)
    got = _shard_dense_weight(
        "model.layers.1.self_attn.q_b_proj.weight", q_b, rank=1, size=2, args=args
    )
    assert torch.equal(got, q_b[MLA_H * QK_DIM // 2 :])
    kv_b = torch.arange(MLA_H * (QK_DIM + V_DIM) * KV_LORA, dtype=torch.float32).view(
        MLA_H * (QK_DIM + V_DIM), KV_LORA
    )
    got = _shard_dense_weight(
        "model.layers.1.self_attn.kv_b_proj.weight", kv_b, rank=0, size=2, args=args
    )
    assert torch.equal(got, kv_b[: MLA_H * (QK_DIM + V_DIM) // 2])
    o = torch.arange(HIDDEN * MLA_H * V_DIM, dtype=torch.float32).view(HIDDEN, MLA_H * V_DIM)
    got = _shard_dense_weight(
        "model.layers.1.self_attn.o_proj.weight", o, rank=1, size=2, args=args
    )
    assert torch.equal(got, o[:, MLA_H * V_DIM // 2 :])


def test_shard_mlp_and_vocab():
    args = _args()
    gate = torch.arange(INTER * HIDDEN, dtype=torch.float32).view(INTER, HIDDEN)
    for name in (
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.up_proj.weight",
    ):
        got = _shard_dense_weight(name, gate, rank=1, size=2, args=args)
        assert torch.equal(got, gate[INTER // 2 :]), name
    down = torch.arange(HIDDEN * INTER, dtype=torch.float32).view(HIDDEN, INTER)
    for name in (
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
    ):
        got = _shard_dense_weight(name, down, rank=0, size=2, args=args)
        assert torch.equal(got, down[:, : INTER // 2]), name
    # vocab shard is div_ceil-contiguous (VocabParallelEmbedding's contract), so an
    # odd vocab still covers every row exactly once.
    emb = torch.arange(11 * 3, dtype=torch.float32).view(11, 3)
    assert torch.equal(
        _shard_dense_weight("model.embed_tokens.weight", emb, rank=0, size=2, args=args), emb[:6]
    )
    assert torch.equal(
        _shard_dense_weight("lm_head.weight", emb, rank=1, size=2, args=args), emb[6:]
    )


def test_shard_replicated_passthrough():
    args = _args()
    for name in (
        # MLA latents + norms (shared by all heads)
        "model.layers.1.self_attn.q_a_proj.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.1.self_attn.q_a_layernorm.weight",
        "model.layers.1.self_attn.kv_a_layernorm.weight",
        # indexer (identical DSA selection on every rank)
        "model.layers.1.self_attn.indexer.wq_b.weight",
        "model.layers.1.self_attn.indexer.wk.weight",
        "model.layers.1.self_attn.indexer.weights_proj.weight",
        "model.layers.1.self_attn.indexer.k_norm.weight",
        "model.layers.1.self_attn.indexer.index_kpool_compress_gate",
        "model.layers.1.self_attn.indexer.index_kpool_compress_ape",
        # KDA per-head-width output norm + norms + mHC + router
        "model.layers.0.self_attn.o_norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.hc_attn_fn",
        "model.layers.0.hc_ffn_scale",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.e_score_correction_bias",
        "model.norm.weight",
    ):
        t = torch.randn(4, 3) if name.endswith("weight") and "norm" not in name else torch.randn(4)
        assert _shard_dense_weight(name, t, rank=1, size=2, args=args) is t, name


# ======================================================================================
# iter_weights at tp=2 over a synthetic checkpoint
# ======================================================================================


def _raw_checkpoint() -> dict[str, torch.Tensor]:
    """The checkpoint-layout (``model.language_model.*``) tensors iter_weights reads."""
    torch.manual_seed(0)
    raw: dict[str, torch.Tensor] = {}

    def add(key, shape, dtype=torch.bfloat16):
        raw[key] = (torch.randn(shape, dtype=torch.float32) * 0.05).to(dtype)

    L = "model.language_model"
    # layer 0: KDA + dense MLP
    s = f"{L}.layers.0.self_attn"
    for proj in ("q_proj", "k_proj", "v_proj"):
        add(f"{s}.{proj}.weight", (P, HIDDEN))
    for proj in ("q", "k", "v"):
        add(f"{s}.{proj}_conv1d.weight", (P, 1, CONV_K))
    add(f"{s}.b_proj.weight", (KDA_H, HIDDEN))
    add(f"{s}.f_a_proj.weight", (KDA_D, HIDDEN))
    add(f"{s}.g_a_proj.weight", (KDA_D, HIDDEN))
    add(f"{s}.f_b_proj.weight", (P, KDA_D))
    add(f"{s}.g_b_proj.weight", (P, KDA_D))
    add(f"{s}.o_proj.weight", (HIDDEN, P))
    add(f"{s}.A_log", (KDA_H,), torch.float32)
    add(f"{s}.dt_bias", (P,), torch.float32)
    add(f"{s}.o_norm.weight", (KDA_D,))
    # layer 1: DSA + sparse MoE
    s = f"{L}.layers.1.self_attn"
    add(f"{s}.q_a_proj.weight", (Q_LORA, HIDDEN))
    add(f"{s}.q_a_layernorm.weight", (Q_LORA,))
    add(f"{s}.q_b_proj.weight", (MLA_H * QK_DIM, Q_LORA))
    add(f"{s}.kv_a_proj_with_mqa.weight", (KV_LORA, HIDDEN))
    add(f"{s}.kv_a_layernorm.weight", (KV_LORA,))
    add(f"{s}.kv_b_proj.weight", (MLA_H * (QK_DIM + V_DIM), KV_LORA))
    add(f"{s}.o_proj.weight", (HIDDEN, MLA_H * V_DIM))
    add(f"{s}.indexer.wq_b.weight", (IDX_H * IDX_D, Q_LORA))
    add(f"{s}.indexer.wk.weight", (IDX_D, HIDDEN))
    add(f"{s}.indexer.weights_proj.weight", (IDX_H, HIDDEN))
    add(f"{s}.indexer.k_norm.weight", (IDX_D,))
    add(f"{s}.indexer.k_norm.bias", (IDX_D,))
    add(f"{s}.indexer.index_kpool_compress_gate", (IDX_D, HIDDEN))
    add(f"{s}.indexer.index_kpool_compress_ape", (KPOOL, IDX_D), torch.float32)
    # per-layer shared keys
    for layer in (0, 1):
        d = f"{L}.layers.{layer}"
        for hc, shape in (
            ("hc_attn_fn", (_MIX, N_STREAMS * HIDDEN)),
            ("hc_attn_base", (_MIX,)),
            ("hc_attn_scale", (3,)),
            ("hc_ffn_fn", (_MIX, N_STREAMS * HIDDEN)),
            ("hc_ffn_base", (_MIX,)),
            ("hc_ffn_scale", (3,)),
        ):
            add(f"{d}.{hc}", shape, torch.float32)
        add(f"{d}.input_layernorm.weight", (HIDDEN,))
        add(f"{d}.post_attention_layernorm.weight", (HIDDEN,))
    for proj, shape in (
        ("gate_proj", (INTER, HIDDEN)),
        ("up_proj", (INTER, HIDDEN)),
        ("down_proj", (HIDDEN, INTER)),
    ):
        add(f"{L}.layers.0.mlp.{proj}.weight", shape)
    add(f"{L}.layers.1.mlp.gate.weight", (EXPERTS, HIDDEN))
    add(f"{L}.layers.1.mlp.gate.e_score_correction_bias", (EXPERTS,), torch.float32)
    for proj, shape in (
        ("gate_proj", (MOE_INTER, HIDDEN)),
        ("up_proj", (MOE_INTER, HIDDEN)),
        ("down_proj", (HIDDEN, MOE_INTER)),
    ):
        add(f"{L}.layers.1.mlp.shared_experts.{proj}.weight", shape)
    add(f"{L}.embed_tokens.weight", (VOCAB, HIDDEN))
    add(f"{L}.norm.weight", (HIDDEN,))
    add("lm_head.weight", (VOCAB, HIDDEN))
    return raw


@pytest.fixture(scope="module")
def tp_checkpoint(tmp_path_factory):
    import json

    from safetensors.torch import save_file

    folder = tmp_path_factory.mktemp("glm5_next_tp_ckpt")
    raw = _raw_checkpoint()
    names = sorted(raw)
    shards = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    save_file({n: raw[n] for n in names[::2]}, str(folder / shards[0]))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / shards[1]))
    weight_map = {n: shards[i % 2] for i, n in enumerate(names)}
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    (folder / "config.json").write_text(json.dumps(_text_config()))
    return str(folder)


def _iter_all(folder, monkeypatch, rank, size):
    import freetoken.models.glm5_next.weight as weight_mod

    hf_cfg = _hf_config()
    monkeypatch.setattr(weight_mod, "cached_load_hf_config", lambda path: hf_cfg)
    _patch_tp(monkeypatch, rank, size)
    return {
        name: tensor.clone()
        for name, tensor in weight_mod.iter_weights(
            folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True
        )
    }


def test_iter_weights_tp2_shards_match_the_tp1_pass(tp_checkpoint, monkeypatch):
    args = _args()
    full = _iter_all(tp_checkpoint, monkeypatch, rank=0, size=1)
    rank0 = _iter_all(tp_checkpoint, monkeypatch, rank=0, size=2)
    rank1 = _iter_all(tp_checkpoint, monkeypatch, rank=1, size=2)
    assert set(rank0) == set(full) == set(rank1)
    for name, tensor in full.items():
        want0 = _shard_dense_weight(name, tensor, rank=0, size=2, args=args)
        want1 = _shard_dense_weight(name, tensor, rank=1, size=2, args=args)
        assert torch.equal(rank0[name], want0), name
        assert torch.equal(rank1[name], want1), name
    # spot-check the shapes the layers declare at tp=2
    assert rank0["model.layers.0.self_attn.in_proj.weight"].shape == (
        (3 * P + KDA_H) // 2 + 2 * KDA_D,
        HIDDEN,
    )
    assert rank0["model.layers.0.self_attn.conv1d.weight"].shape == (3 * P // 2, 1, CONV_K)
    assert rank0["model.layers.0.self_attn.o_proj.weight"].shape == (HIDDEN, P // 2)
    assert rank0["model.layers.1.self_attn.q_b_proj.weight"].shape == (
        MLA_H * QK_DIM // 2,
        Q_LORA,
    )
    assert rank0["model.layers.1.self_attn.o_proj.weight"].shape == (HIDDEN, MLA_H * V_DIM // 2)
    assert rank0["model.layers.0.mlp.down_proj.weight"].shape == (HIDDEN, INTER // 2)
    assert rank0["model.embed_tokens.weight"].shape == (VOCAB // 2, HIDDEN)
    # replicated tensors keep the full shape
    assert rank0["model.layers.1.self_attn.q_a_proj.weight"].shape == (Q_LORA, HIDDEN)


# ======================================================================================
# Gated MLP, two simulated ranks vs the full layer (pure torch -> CPU)
# ======================================================================================


def test_gated_mlp_tp2_matches_full(monkeypatch, noop_comm):
    from freetoken.models.glm5_next.mlp import Glm5NextGatedMLP

    args = _args()
    _patch_tp(monkeypatch, 0, 1)
    full = Glm5NextGatedMLP(HIDDEN, INTER, quant="none", swiglu_limit=None)
    gen = torch.Generator().manual_seed(5)
    for tensor in full.state_dict().values():
        tensor.normal_(0.0, 0.05, generator=gen)
    full_sd = {k: v.clone() for k, v in full.state_dict().items()}

    outs = []
    for rank in (0, 1):
        _patch_tp(monkeypatch, rank, 2)
        op = Glm5NextGatedMLP(HIDDEN, INTER, quant="none", swiglu_limit=None)
        op.load_state_dict(
            {
                key: _shard_dense_weight(
                    f"model.layers.0.mlp.{key}", tensor, rank=rank, size=2, args=args
                )
                for key, tensor in full_sd.items()
            }
        )
        outs.append(op)

    gen = torch.Generator().manual_seed(11)
    x = torch.randn(7, HIDDEN, generator=gen)
    got = outs[0].forward(x) + outs[1].forward(x)
    torch.testing.assert_close(got, full.forward(x), rtol=1e-5, atol=1e-6)
    # the rank layers really are sharded
    assert outs[0].gate_proj.weight.shape == (INTER // 2, HIDDEN)
    assert outs[0].down_proj.weight.shape == (HIDDEN, INTER // 2)


# ======================================================================================
# KDA op, two simulated ranks vs the full layer (kernels -> CUDA)
# ======================================================================================


def _kda_group():
    from freetoken.models.config import LinearGatedDeltaGroupConfig

    return LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,),
        num_key_heads=KDA_H, num_value_heads=KDA_H,
        key_head_dim=KDA_D, value_head_dim=KDA_D,
        conv_kernel_dim=CONV_K, output_gate="sigmoid", variant="kda",
    )


def _fla(cu, indices, has_init=None, fresh=None):
    from freetoken.attention.linear import FLAMetadata

    return FLAMetadata(
        cu_seqlens=cu, cache_indices=indices,
        has_initial_state=has_init, fresh_state_indices=fresh,
    )


def _kda_prefill_ctx(monkeypatch, pool, total, device):
    batch = SimpleNamespace(
        is_decode=False,
        fla_metadata=_fla(
            torch.tensor([0, total], dtype=torch.int32, device=device),
            torch.tensor([1], dtype=torch.int32, device=device),
            torch.tensor([False], device=device),
            torch.tensor([1], dtype=torch.int64, device=device),
        ),
    )
    ctx = SimpleNamespace(batch=batch, linear_state_pool=pool)
    monkeypatch.setattr("freetoken.models.glm5_next.kda.get_global_ctx", lambda: ctx)


def _kda_decode_ctx(monkeypatch, pool, device):
    batch = SimpleNamespace(
        is_decode=True,
        fla_metadata=_fla(
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            torch.tensor([1], dtype=torch.int32, device=device),
        ),
    )
    ctx = SimpleNamespace(batch=batch, linear_state_pool=pool)
    monkeypatch.setattr("freetoken.models.glm5_next.kda.get_global_ctx", lambda: ctx)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kda_tp2_prefill_and_decode_match_full(monkeypatch, noop_comm):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.glm5_next.kda import Glm5NextKDA
    from freetoken.utils.torch_utils import torch_dtype

    DEV = torch.device("cuda")
    args = _args()
    cfg = _kda_op_config()
    group = _kda_group()

    # The full (tp=1) layer; its weights drive both rank shards.
    _patch_tp(monkeypatch, 0, 1)
    with torch.device(DEV), torch_dtype(torch.bfloat16):
        full = Glm5NextKDA(cfg, layer_id=0)
    gen = torch.Generator(device=DEV).manual_seed(5)
    for name, tensor in full.state_dict().items():
        if name.endswith("o_norm.weight"):
            tensor.normal_(0.0, 0.1, generator=gen).add_(1.0)
        else:
            tensor.normal_(0.0, 0.05, generator=gen)
    full_sd = {k: v.clone() for k, v in full.state_dict().items()}

    def build_rank(rank: int):
        _patch_tp(monkeypatch, rank, 2)
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            op = Glm5NextKDA(cfg, layer_id=0)
        op.load_state_dict(
            {
                key: _shard_dense_weight(
                    f"model.layers.0.self_attn.{key}", tensor, rank=rank, size=2, args=args
                )
                for key, tensor in full_sd.items()
            }
        )
        return op, LinearStatePool(group, 4, torch.bfloat16, DEV, tp_size=2)

    ops = [build_rank(r) for r in (0, 1)]
    assert ops[0][0].num_heads == KDA_H // 2
    assert ops[0][0].in_proj.weight.shape[0] == (3 * P + KDA_H) // 2 + 2 * KDA_D
    assert ops[0][0].o_proj.weight.shape == (HIDDEN, P // 2)

    total = 64
    gen = torch.Generator(device=DEV).manual_seed(11)
    x = torch.randn(total, HIDDEN, device=DEV, dtype=torch.bfloat16, generator=gen)

    full_pool = LinearStatePool(group, 4, torch.bfloat16, DEV, tp_size=1)
    _kda_prefill_ctx(monkeypatch, full_pool, total, DEV)
    ref = full.forward(x)
    partials = []
    for op, pool in ops:
        _kda_prefill_ctx(monkeypatch, pool, total, DEV)
        partials.append(op.forward(x))
    got = partials[0] + partials[1]
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)

    # one decode step off the carried per-rank state
    gen = torch.Generator(device=DEV).manual_seed(13)
    nxt = torch.randn(1, HIDDEN, device=DEV, dtype=torch.bfloat16, generator=gen)
    _kda_decode_ctx(monkeypatch, full_pool, DEV)
    ref = full.forward(nxt)
    partials = []
    for op, pool in ops:
        _kda_decode_ctx(monkeypatch, pool, DEV)
        partials.append(op.forward(nxt))
    got = partials[0] + partials[1]
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


# ======================================================================================
# DSA/MLA attention, two simulated ranks vs a dense latent oracle (CUDA)
# ======================================================================================


class _MlaLatentOracle:
    """Dense MLA-latent oracle driven by the LOCAL tensor shapes (TP-aware): q arrives
    as [T, local_heads, kv_lora_rank], c_kv as [T, kv_lora_rank]; the per-request
    latent history is kept per layer. Attention in latent space: softmax(q . c) @ c.
    """

    dsa_enabled = False  # the layer skips the indexer entirely

    def __init__(self, scale: float):
        self._scale = scale
        self._kv: dict[int, list[torch.Tensor]] = {}

    def mla_forward(self, q_nope, q_pe, c_kv, k_rope, layer_id, batch, indexer_inputs=None):
        assert indexer_inputs is None
        hist = self._kv.setdefault(layer_id, [])
        hist.append(c_kv.float())
        c = torch.cat(hist)  # [L, lora]
        q = q_nope.float()  # [T, H_local, lora]
        t = q.shape[0]
        scores = torch.einsum("thl,kl->thk", q, c) * self._scale
        visible = torch.arange(c.shape[0], device=q.device) <= (
            c.shape[0] - t + torch.arange(t, device=q.device)
        ).unsqueeze(-1)
        scores = scores.masked_fill(~visible.unsqueeze(1), float("-inf"))
        return torch.einsum("thk,kl->thl", scores.softmax(-1), c).to(q_nope.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dsa_attention_tp2_matches_full(monkeypatch, noop_comm):
    from freetoken.models.glm5_next.attention import Glm5NextAttention
    from freetoken.utils.torch_utils import torch_dtype

    DEV = torch.device("cuda")
    args = _args()
    cfg = _kda_op_config()  # same SimpleNamespace: glm5_args + attn_quant="none"
    scale = QK_DIM**-0.5

    _patch_tp(monkeypatch, 0, 1)
    with torch.device(DEV), torch_dtype(torch.bfloat16):
        full = Glm5NextAttention(cfg, layer_id=1)
    gen = torch.Generator(device=DEV).manual_seed(7)
    for tensor in full.state_dict().values():
        tensor.normal_(0.0, 0.05, generator=gen)
    full_sd = {k: v.clone() for k, v in full.state_dict().items()}

    def build_rank(rank: int):
        _patch_tp(monkeypatch, rank, 2)
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            op = Glm5NextAttention(cfg, layer_id=1)
        op.load_state_dict(
            {
                key: _shard_dense_weight(
                    f"model.layers.1.self_attn.{key}", tensor, rank=rank, size=2, args=args
                )
                for key, tensor in full_sd.items()
            }
        )
        return op

    ops = [build_rank(r) for r in (0, 1)]
    assert ops[0].num_heads == MLA_H // 2
    assert ops[0].q_b_proj.weight.shape == (MLA_H * QK_DIM // 2, Q_LORA)
    assert ops[0].kv_b_proj.weight.shape == (MLA_H * (QK_DIM + V_DIM) // 2, KV_LORA)
    assert ops[0].o_proj.weight.shape == (HIDDEN, MLA_H * V_DIM // 2)

    t = 16
    gen = torch.Generator(device=DEV).manual_seed(3)
    x = torch.randn(t, HIDDEN, device=DEV, dtype=torch.bfloat16, generator=gen)
    batch = SimpleNamespace(
        padded_reqs=[SimpleNamespace(extend_len=t, table_idx=0, cached_len=0)]
    )

    def run(op):
        ctx = SimpleNamespace(attn_backend=_MlaLatentOracle(scale), batch=batch)
        monkeypatch.setattr(
            "freetoken.models.glm5_next.attention.get_global_ctx", lambda: ctx
        )
        return op.forward(x)

    ref = run(full)
    got = run(ops[0]) + run(ops[1])
    torch.testing.assert_close(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


# ======================================================================================
# Full-model state-dict shape contract at tp=2 (meta device, CPU)
# ======================================================================================


def test_full_model_state_dict_shapes_match_tp2_shards(monkeypatch):
    """Every key of the tp=2 model's state dict has exactly the shape _shard_dense_weight
    produces for that rank -- the contract load_state_dict enforces at boot."""
    import dataclasses

    from freetoken.models.glm5_next.config import parse_config
    from freetoken.models.glm5_next.model import Glm5NextForCausalLM

    config = parse_config(_hf_config())
    config = dataclasses.replace(config, moe_backend="offload")  # experts live in banks
    _patch_tp(monkeypatch, 0, 1)
    with torch.device("meta"):
        full_sd = Glm5NextForCausalLM(config).state_dict()
    for rank in (0, 1):
        _patch_tp(monkeypatch, rank, 2)
        with torch.device("meta"):
            rank_sd = Glm5NextForCausalLM(config).state_dict()
        assert set(rank_sd) == set(full_sd)
        for name, tensor in full_sd.items():
            probe = torch.zeros(tensor.shape, dtype=tensor.dtype)
            want = _shard_dense_weight(name, probe, rank=rank, size=2, args=config.glm5_args)
            assert tuple(rank_sd[name].shape) == tuple(want.shape), (
                name, tuple(rank_sd[name].shape), tuple(want.shape)
            )


def test_indivisible_tp_size_raises(monkeypatch):
    from freetoken.models.glm5_next.attention import Glm5NextAttention
    from freetoken.models.glm5_next.kda import Glm5NextKDA

    cfg = _kda_op_config()
    _patch_tp(monkeypatch, 0, 3)  # 4 heads on 3 ranks
    with pytest.raises(NotImplementedError, match="linear_num_heads % tp_size"):
        Glm5NextKDA(cfg, layer_id=0)
    with pytest.raises(NotImplementedError, match="num_heads % tp_size"):
        Glm5NextAttention(cfg, layer_id=1)
