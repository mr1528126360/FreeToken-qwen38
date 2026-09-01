"""TP=2 (expert/head sharding) tests for qwen4_exp.

Single-process simulations: each rank's layer/loader is built under a patched
``freetoken.distributed.info._TP_INFO`` and the all-reduce is a no-op stub, so the two
partials are summed by hand -- exactly what the real all-reduce computes. Covered:

* ``_shard_dense_weight`` row/col splits for every sharded dense tensor (and the
  replicated pass-through), plus an end-to-end ``iter_weights`` run over the synthetic
  checkpoint from ``test_weight`` at tp=2;
* the GDN layer: two TP halves (local head dims, local state pool) reproduce the full
  fp32 reference;
* the QSA attention layer: two TP halves against a dense oracle driven by the LOCAL
  tensor shapes (per-rank q/kv head counts).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.models.qwen4_exp.weight import _shard_dense_weight

from .common import fill_weights, hf_config, toy_hf_config  # noqa: F401  (shared builders)

H, HD = 256, 128  # hidden / head dim for the GDN cases (test_gdn geometry)
KH, VH = 16, 48  # GDN key/value heads (Qwen3.8 3:1 ratio)
QH, KVH, AHD = 4, 2, 16  # QSA q / kv heads, head dim (test_weight geometry)
CONV_K = 4


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


def _gdn_group() -> LinearGatedDeltaGroupConfig:
    return LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=KH, num_value_heads=VH,
        key_head_dim=HD, value_head_dim=HD, conv_kernel_dim=CONV_K, output_gate="sigmoid",
    )


def _cfg() -> SimpleNamespace:
    group = _gdn_group()
    return SimpleNamespace(
        num_qo_heads=QH,
        num_kv_heads=KVH,
        head_dim=AHD,
        linear_attention_group=lambda: group,
    )


# ======================================================================================
# _shard_dense_weight
# ======================================================================================


def test_shard_qkv_proj_is_head_aligned():
    cfg = _cfg()
    full = torch.arange(2 * QH * AHD + 2 * KVH * AHD, dtype=torch.float32).unsqueeze(1).expand(-1, 3)
    qo, kv = QH * AHD, KVH * AHD
    for rank in (0, 1):
        got = _shard_dense_weight("model.layers.3.self_attn.qkv_proj.weight", full, rank=rank, size=2, config=cfg)
        q, k, v = full[: 2 * qo], full[2 * qo : 2 * qo + kv], full[2 * qo + kv :]
        want = torch.cat([
            q[rank * qo : (rank + 1) * qo],  # q is 2x wide (gate); half of it per rank
            k[rank * kv // 2 : (rank + 1) * kv // 2],
            v[rank * kv // 2 : (rank + 1) * kv // 2],
        ])
        assert torch.equal(got, want)


def test_shard_gdn_in_proj_splits_every_head_block():
    cfg = _cfg()
    key, value = KH * HD, VH * HD
    conv = 2 * key + value
    full = torch.arange(conv + value + 2 * VH, dtype=torch.float32).unsqueeze(1)
    for rank in (0, 1):
        got = _shard_dense_weight("model.layers.0.linear_attn.in_proj.weight", full, rank=rank, size=2, config=cfg)
        want = torch.cat([
            full[rank * key // 2 : (rank + 1) * key // 2],                       # q heads
            full[key + rank * key // 2 : key + (rank + 1) * key // 2],           # k heads
            full[2 * key + rank * value // 2 : 2 * key + (rank + 1) * value // 2],  # v heads
            full[conv + rank * value // 2 : conv + (rank + 1) * value // 2],     # z
            full[conv + value + rank * VH // 2 : conv + value + (rank + 1) * VH // 2],  # b
            full[conv + value + VH + rank * VH // 2 : conv + value + VH + (rank + 1) * VH // 2],  # a
        ])
        assert torch.equal(got, want)
        # the local layout the layer splits by: [conv_local | z_local | b_local | a_local]
        assert got.shape[0] == (2 * key + value) // 2 + value // 2 + VH


def test_shard_gdn_conv1d_matches_in_proj_conv_segment():
    cfg = _cfg()
    key, value = KH * HD, VH * HD
    conv = torch.arange(2 * key + value, dtype=torch.float32).view(-1, 1, CONV_K)
    for rank in (0, 1):
        got = _shard_dense_weight("model.layers.0.linear_attn.conv1d.weight", conv, rank=rank, size=2, config=cfg)
        want = torch.cat([
            conv[rank * key // 2 : (rank + 1) * key // 2],
            conv[key + rank * key // 2 : key + (rank + 1) * key // 2],
            conv[2 * key + rank * value // 2 : 2 * key + (rank + 1) * value // 2],
        ])
        assert torch.equal(got, want)


def test_shard_row_parallel_and_vector_splits():
    cfg = _cfg()
    o = torch.arange(8 * 12, dtype=torch.float32).view(8, 12)
    got = _shard_dense_weight("model.layers.3.self_attn.o_proj.weight", o, rank=1, size=2, config=cfg)
    assert torch.equal(got, o[:, 6:])
    a_log = torch.arange(VH, dtype=torch.float32)
    got = _shard_dense_weight("model.layers.0.linear_attn.A_log", a_log, rank=0, size=2, config=cfg)
    assert torch.equal(got, a_log[: VH // 2])
    down = torch.arange(8 * 12, dtype=torch.float32).view(8, 12)
    got = _shard_dense_weight("model.layers.0.mlp.shared_expert.down_proj.weight", down, rank=0, size=2, config=cfg)
    assert torch.equal(got, down[:, :6])


def test_shard_shared_expert_gate_up_splits_each_half():
    cfg = _cfg()
    full = torch.arange(2 * 8, dtype=torch.float32).unsqueeze(1)
    for rank in (0, 1):
        got = _shard_dense_weight(
            "model.layers.5.mlp.shared_expert.gate_up_proj.weight", full, rank=rank, size=2, config=cfg
        )
        want = torch.cat([full[rank * 4 : (rank + 1) * 4], full[8 + rank * 4 : 8 + (rank + 1) * 4]])
        assert torch.equal(got, want)


def test_shard_embedding_vocab_and_replicated_passthrough():
    cfg = _cfg()
    emb = torch.arange(11 * 3, dtype=torch.float32).view(11, 3)
    assert torch.equal(
        _shard_dense_weight("model.embed_tokens.weight", emb, rank=0, size=2, config=cfg), emb[:6]
    )
    assert torch.equal(
        _shard_dense_weight("lm_head.weight", emb, rank=1, size=2, config=cfg), emb[6:]
    )
    for name in (
        "model.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight",
        "model.layers.0.attn_hyper_connection.hc_norm.weight",
        "model.layers.3.self_attn.indexer.index_qk_proj.weight",
        "model.layers.3.self_attn.q_norm.weight",
        "model.layers.5.mlp.gate.weight",
        "model.layers.0.ple.key_proj.weight",
    ):
        t = torch.randn(4, 3)
        assert _shard_dense_weight(name, t, rank=1, size=2, config=cfg) is t


# ======================================================================================
# iter_weights at tp=2 over the synthetic checkpoint
# ======================================================================================


@pytest.fixture(scope="module")
def tp_checkpoint(tmp_path_factory):
    """The test_weight synthetic checkpoint + a matching HF config for parse_config."""
    from safetensors.torch import save_file

    from .test_weight import _raw_checkpoint

    torch.manual_seed(0)
    folder = tmp_path_factory.mktemp("qwen4_exp_tp_ckpt")
    raw = _raw_checkpoint()
    names = sorted(raw)
    save_file({n: raw[n] for n in names[::2]}, str(folder / "model-bf16-00001.safetensors"))
    save_file({n: raw[n] for n in names[1::2]}, str(folder / "model-bf16-00002.safetensors"))
    cfg = hf_config(
        num_layers=2, head_dim=AHD, num_q=QH, num_kv=KVH, index_head_dim=8, index_heads=4,
        hidden=32, ple_layer_ids=[1],
        linear_num_key_heads=2, linear_num_value_heads=6,
        linear_key_head_dim=8, linear_value_head_dim=8,
    )
    return str(folder), raw, cfg


def _iter_all(folder, monkeypatch, rank, size, hf_cfg):
    import freetoken.models.qwen4_exp.weight as weight_mod

    monkeypatch.setattr(weight_mod, "cached_load_hf_config", lambda path: hf_cfg)
    _patch_tp(monkeypatch, rank, size)
    return {
        name: tensor.clone()
        for name, tensor in weight_mod.iter_weights(
            folder, torch.device("cpu"), include_moe_experts=True, include_non_moe=True
        )
    }


def test_iter_weights_tp2_shards_match_the_tp1_pass(tp_checkpoint, monkeypatch):
    from freetoken.models.qwen4_exp.config import parse_config

    folder, _raw, hf_cfg = tp_checkpoint
    full = _iter_all(folder, monkeypatch, rank=0, size=1, hf_cfg=hf_cfg)
    cfg = parse_config(hf_cfg)
    rank0 = _iter_all(folder, monkeypatch, rank=0, size=2, hf_cfg=hf_cfg)
    rank1 = _iter_all(folder, monkeypatch, rank=1, size=2, hf_cfg=hf_cfg)
    assert set(rank0) == set(full) == set(rank1)
    for name, tensor in full.items():
        want0 = _shard_dense_weight(name, tensor, rank=0, size=2, config=cfg)
        want1 = _shard_dense_weight(name, tensor, rank=1, size=2, config=cfg)
        assert torch.equal(rank0[name], want0), name
        assert torch.equal(rank1[name], want1), name
    # spot-check the shapes the layers declare at tp=2
    assert rank0["model.layers.1.self_attn.qkv_proj.weight"].shape == (QH * AHD + KVH * AHD, 32)
    assert rank0["model.layers.1.self_attn.o_proj.weight"].shape == (32, QH * AHD // 2)
    assert rank0["model.layers.0.linear_attn.in_proj.weight"].shape == (
        (2 * 2 * 8 + 6 * 8) // 2 + (6 * 8) // 2 + 6, 32
    )


# ======================================================================================
# GDN layer, two simulated ranks vs the full fp32 reference
# ======================================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_gdn_tp2_prefill_and_decode_match_reference(monkeypatch, noop_comm):
    from freetoken.core import Batch, Context, Req, SamplingParams
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.utils import torch_dtype

    from .test_gdn import _make_layer, _ref_out, _state_dict

    DEV = torch.device("cuda")
    group = _gdn_group()
    cfg = _cfg()
    # The full reference + its (TP=1-layout) fused state dict.
    _op_full, ref = _make_layer(3, seed=5)  # ratio 3 -> KH/VH = 16/48; builds at ambient tp=1
    full_sd = _state_dict(ref)

    def build_rank(rank: int):
        from freetoken.models.qwen4_exp.gdn import Qwen4ExpGatedDeltaNet

        _patch_tp(monkeypatch, rank, 2)
        with torch.device("meta"), torch_dtype(torch.bfloat16):
            op = Qwen4ExpGatedDeltaNet(
                hidden_size=H, num_k_heads=KH, num_v_heads=VH, head_k_dim=HD,
                head_v_dim=HD, conv_kernel_size=CONV_K, rms_norm_eps=1e-6, layer_id=0,
                output_gate="sigmoid",
            )
        sharded = {
            key: _shard_dense_weight(
                f"model.layers.0.linear_attn.{key}", tensor, rank=rank, size=2, config=cfg
            )
            for key, tensor in full_sd.items()
        }
        op.load_state_dict(sharded)
        import freetoken.core as core

        core._GLOBAL_CTX = None
        ctx = Context(page_size=64)
        ctx.linear_state_pool = LinearStatePool(group, 8, torch.bfloat16, DEV, tp_size=2)
        core.set_global_ctx(ctx)
        return op, ctx

    ops = [build_rank(r) for r in (0, 1)]

    torch.manual_seed(11)
    hidden = torch.randn(128, H, device=DEV, dtype=torch.bfloat16)
    partials = []
    for i, (op, ctx) in enumerate(ops):
        import freetoken.core as core

        # rank 1's build left ITS ctx global; point back to ours
        core._GLOBAL_CTX = None
        core.set_global_ctx(ctx)
        req = Req(input_ids=torch.zeros(128, dtype=torch.int32), table_idx=1, cached_len=0,
                  output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = [req]
        with ctx.forward_batch(batch):
            partials.append(op.forward(hidden))
    got = partials[0] + partials[1]
    torch.testing.assert_close(got.float(), _ref_out(ref, hidden), rtol=2e-2, atol=2e-2)

    # one decode step off the carried per-rank state
    nxt = torch.randn(1, H, device=DEV, dtype=torch.bfloat16)
    partials = []
    for op, ctx in ops:
        import freetoken.core as core

        core._GLOBAL_CTX = None
        core.set_global_ctx(ctx)
        # a fresh Req standing in for the decode step: len(input_ids) == device_len
        req = Req(input_ids=torch.zeros(129, dtype=torch.int32), table_idx=1, cached_len=128,
                  output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
        batch = Batch(reqs=[req], phase="decode")
        batch.padded_reqs = [req]
        batch.linear_table_idx = torch.tensor([1], dtype=torch.int32, device=DEV)
        with ctx.forward_batch(batch):
            partials.append(op.forward(nxt))
    got = partials[0] + partials[1]
    torch.testing.assert_close(
        got[0].float(), _ref_out(ref, torch.cat([hidden, nxt], dim=0))[-1], rtol=2e-2, atol=2e-2
    )


# ======================================================================================
# QSA attention layer, two simulated ranks vs a dense oracle
# ======================================================================================


class _LocalDenseOracle:
    """Dense-attention oracle driven by the LOCAL tensor shapes (TP-aware): k/v arrive as
    [T, local_kv*head_dim], q as [T, local_q, head_dim]; per-request KV kept per layer."""

    def __init__(self, head_dim: int):
        self.head_dim = head_dim
        self._kv: dict[int, dict[int, list]] = {}

    def qsa_forward(self, q, k, v, index, layer_id, batch):
        del index
        hd = self.head_dim
        nkv = k.shape[1] // hd
        k = k.view(-1, nkv, hd)
        v = v.view(-1, nkv, hd)
        layers = self._kv.setdefault(layer_id, {})
        out = torch.empty_like(q)
        offset = 0
        for r in batch.padded_reqs:
            n, slot, prefix = r.extend_len, r.table_idx, r.cached_len
            rows = slice(offset, offset + n)
            ks, vs = layers.setdefault(slot, ([], []))
            ks.append(k[rows].float())
            vs.append(v[rows].float())
            kk, vv = torch.cat(ks), torch.cat(vs)
            rep = q.shape[1] // nkv
            total = kk.shape[0]
            scores = torch.einsum(
                "qhd,khd->hqk", q[rows].float(), kk.repeat_interleave(rep, dim=1)
            ) * hd**-0.5
            visible = torch.arange(total, device=q.device) <= (
                prefix + torch.arange(n, device=q.device)
            ).unsqueeze(-1)
            scores = scores.masked_fill(~visible, float("-inf"))
            out[rows] = torch.einsum(
                "hqk,khd->qhd", scores.softmax(-1), vv.repeat_interleave(rep, dim=1)
            ).to(q.dtype)
            offset += n
        return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qsa_attention_tp2_matches_dense_oracle(monkeypatch, noop_comm):
    import freetoken.core as core
    from freetoken.core import Batch, Context, Req, SamplingParams
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention
    from freetoken.utils.torch_utils import torch_dtype

    DEV = torch.device("cuda")
    config = parse_config(hf_config(num_layers=4))  # layer 3 is the QSA layer
    cfg = SimpleNamespace(
        num_qo_heads=config.num_qo_heads,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        linear_attention_group=config.linear_attention_group,
    )

    def build(rank: int):
        _patch_tp(monkeypatch, rank, 2)
        with torch.device(DEV), torch_dtype(torch.bfloat16):
            return Qwen4ExpAttention(config, layer_id=3)

    # Reference: the full (tp=1) layer. Its weights drive both rank shards.
    import freetoken.distributed.info as dist_info

    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(rank=0, size=1))
    with torch.device(DEV), torch_dtype(torch.bfloat16):
        ref = Qwen4ExpAttention(config, layer_id=3)
    fill_weights(ref, seed=7, device=DEV)
    full_sd = {k: v.clone() for k, v in ref.state_dict().items()}

    ops = []
    for rank in (0, 1):
        op = build(rank)
        sharded = {
            key: _shard_dense_weight(
                f"model.layers.3.self_attn.{key}", tensor, rank=rank, size=2, config=cfg
            )
            for key, tensor in full_sd.items()
        }
        op.load_state_dict(sharded)
        ops.append(op)

    torch.manual_seed(3)
    x = torch.randn(64, config.hidden_size, device=DEV, dtype=torch.bfloat16)
    outs = []
    for op in [ref, *ops]:
        core._GLOBAL_CTX = None
        ctx = Context(page_size=64)
        ctx.attn_backend = _LocalDenseOracle(config.head_dim)
        core.set_global_ctx(ctx)
        req = Req(input_ids=torch.zeros(64, dtype=torch.int32), table_idx=1, cached_len=0,
                  output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = [req]
        batch.positions = torch.arange(64, dtype=torch.int32, device=DEV)
        with ctx.forward_batch(batch):
            outs.append(op.forward(x, batch))
    got = outs[1] + outs[2]
    torch.testing.assert_close(got.float(), outs[0].float(), rtol=2e-2, atol=2e-2)
    # the rank layers really are sharded
    assert ops[0].num_q == config.num_qo_heads // 2
    assert ops[0].num_kv == config.num_kv_heads // 2
    assert ops[0].qkv_proj.weight.shape[0] == full_sd["qkv_proj.weight"].shape[0] // 2
    assert ops[0].o_proj.weight.shape[1] == full_sd["o_proj.weight"].shape[1] // 2


def test_full_model_state_dict_shapes_match_tp2_shards(monkeypatch):
    """Every key of the tp=2 model's state dict has exactly the shape _shard_dense_weight
    produces for that rank -- the contract load_state_dict enforces at boot."""
    import dataclasses

    from freetoken.layers import rotary
    from freetoken.models.qwen4_exp.config import parse_config
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    config = parse_config(hf_config(num_layers=4))
    config = dataclasses.replace(config, moe_backend="offload")  # experts live in banks
    saved = rotary._ROPE_DEVICE
    rotary.set_rope_device(torch.device("cpu"))  # get_rope refuses to build on meta
    rotary.get_rope.cache_clear()
    try:
        _patch_tp(monkeypatch, 0, 1)
        with torch.device("meta"):
            full_sd = Qwen4ExpForCausalLM(config).state_dict()
        for rank in (0, 1):
            _patch_tp(monkeypatch, rank, 2)
            with torch.device("meta"):
                rank_sd = Qwen4ExpForCausalLM(config).state_dict()
            assert set(rank_sd) == set(full_sd)
            for name, tensor in full_sd.items():
                probe = torch.zeros(tensor.shape, dtype=tensor.dtype)
                want = _shard_dense_weight(name, probe, rank=rank, size=2, config=config)
                assert tuple(rank_sd[name].shape) == tuple(want.shape), (
                    name, tuple(rank_sd[name].shape), tuple(want.shape)
                )
    finally:
        rotary.set_rope_device(saved)
        rotary.get_rope.cache_clear()
