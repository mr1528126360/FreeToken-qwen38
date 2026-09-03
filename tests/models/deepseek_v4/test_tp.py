"""TP=2 (expert-parallel + vocab sharding) tests for deepseek_v4 (DeepSeek-V4-Flash).

Single-process simulations, same pattern as qwen4_exp/test_tp.py and
glm5_next/test_tp.py: each rank's layer/loader is built under a patched
``freetoken.distributed.info._TP_INFO`` and the collectives are stubs, so the
two partials are reassembled by hand -- exactly what the real all-reduce /
all-gather compute. DSV4's dense projections are block-fp8 (128x128 scale
blocks, not splittable), so TP=2 shards only the routed experts
(``id % 2 == rank``) and the vocab rows (embed/head); attention and the shared
expert replicate. Covered:

* ``_shard_vocab_rows`` splits (+ the indivisible-vocab error) and an
  end-to-end ``iter_weights`` run over a synthetic checkpoint at tp=2;
* the ds_fp4 expert-bank loaders' ``expert_shard`` filtering (owned experts at
  local row ``id // size``) for the serial, parallel and dummy paths;
* ``DSV4OffloadMoELayer._prefill_routed`` at tp=2 -- the on-demand slot path
  (the pre-TP bug: it routed GLOBAL expert ids into a local cache) and the
  streaming path, two halves summed vs the tp=1 full layer (CUDA);
* the Transformer embed/head wiring: two-rank embedding partials sum to the
  full embedding; per-rank head logits match the full-logit slices (CUDA);
* the full-model state-dict shape + load contract at tp=2 (meta device);
* divisibility errors (experts % tp_size, vocab % tp_size).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import DistributedInfo
from freetoken.models.deepseek_v4.args import DeepseekV4Args
from freetoken.models.deepseek_v4.weight import _shard_vocab_rows

# Toy geometry: every fp8 block-dim a multiple of 128 (the loader's scale blocks).
V, D = 130, 256  # vocab (even: TP-divisible), hidden
QL, OL, OG = 128, 128, 1  # q_lora_rank, o_lora_rank, o_groups
NH, HD = 2, 128  # heads, head_dim
MI, E, TOPK = 128, 8, 2  # moe_inter_dim, routed experts, top-k
HC = 2  # hc_mult


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


def _toy_args(**over) -> DeepseekV4Args:
    base = dict(
        vocab_size=V, dim=D, moe_inter_dim=MI, n_layers=1, n_hash_layers=0,
        n_mtp_layers=0, n_heads=NH, n_routed_experts=E, n_activated_experts=TOPK,
        q_lora_rank=QL, head_dim=HD, rope_head_dim=64, o_groups=OG,
        o_lora_rank=OL, window_size=16, compress_ratios=(0,), hc_mult=HC,
        index_n_heads=2, index_head_dim=32, swiglu_limit=7.0,
    )
    base.update(over)
    return DeepseekV4Args(**base)


def _toy_model(monkeypatch, rank: int, size: int):
    from freetoken.models.deepseek_v4.model import DeepseekV4ForCausalLM

    _patch_tp(monkeypatch, rank, size)
    with torch.device("meta"):
        return DeepseekV4ForCausalLM(SimpleNamespace(dsv4_args=_toy_args()))


# ======================================================================================
# _shard_vocab_rows
# ======================================================================================


def test_shard_vocab_rows_split_and_identity():
    t = torch.arange(10 * 3, dtype=torch.float32).view(10, 3)
    assert _shard_vocab_rows(t, rank=0, size=1) is t
    assert torch.equal(_shard_vocab_rows(t, 0, 2), t[:5])
    assert torch.equal(_shard_vocab_rows(t, 1, 2), t[5:])
    with pytest.raises(ValueError, match="vocab_size % tp_size"):
        _shard_vocab_rows(t, 0, 4)  # 10 % 4 != 0


# ======================================================================================
# iter_weights at tp=2 over a synthetic checkpoint
# ======================================================================================


def _fp8(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    w = (torch.randn(n, k, dtype=torch.float32) * 0.05).to(torch.float8_e4m3fn)
    s = torch.randint(120, 127, (n // 128, k // 128), dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    return w, s


def _raw_checkpoint() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    raw: dict[str, torch.Tensor] = {}

    def add(key, shape, dtype=torch.bfloat16):
        raw[key] = (torch.randn(shape, dtype=torch.float32) * 0.05).to(dtype)

    def add_fp8(prefix, n, k):
        w, s = _fp8(n, k)
        raw[f"{prefix}.weight"] = w
        raw[f"{prefix}.scale"] = s

    add("embed.weight", (V, D))
    add("norm.weight", (D,))
    add("head.weight", (V, D))
    add("hc_head_fn", (HC, HC * D), torch.float32)
    add("hc_head_base", (HC,), torch.float32)
    add("hc_head_scale", (1,), torch.float32)

    a = "layers.0.attn"
    add_fp8(f"{a}.wq_a", QL, D)
    add(f"{a}.q_norm.weight", (QL,), torch.float32)
    add_fp8(f"{a}.wq_b", NH * HD, QL)
    add_fp8(f"{a}.wkv", HD, D)
    add(f"{a}.kv_norm.weight", (HD,), torch.float32)
    add_fp8(f"{a}.wo_a", OG * OL, NH * HD // OG)  # dequantized to bf16 by the loader
    add_fp8(f"{a}.wo_b", D, OG * OL)
    add(f"{a}.attn_sink", (NH,), torch.float32)
    add("layers.0.attn_norm.weight", (D,), torch.float32)
    add("layers.0.ffn_norm.weight", (D,), torch.float32)
    add("layers.0.ffn.gate.weight", (E, D))
    add("layers.0.ffn.gate.bias", (E,), torch.float32)
    add_fp8("layers.0.ffn.shared_experts.w1", MI, D)
    add_fp8("layers.0.ffn.shared_experts.w2", D, MI)
    add_fp8("layers.0.ffn.shared_experts.w3", MI, D)
    mix = (2 + HC) * HC
    for nm, shape in (
        ("hc_attn_fn", (mix, HC * D)), ("hc_ffn_fn", (mix, HC * D)),
        ("hc_attn_base", (mix,)), ("hc_ffn_base", (mix,)),
        ("hc_attn_scale", (3,)), ("hc_ffn_scale", (3,)),
    ):
        add(f"layers.0.{nm}", shape, torch.float32)
    return raw


@pytest.fixture(scope="module")
def tp_checkpoint(tmp_path_factory):
    from safetensors.torch import save_file

    folder = tmp_path_factory.mktemp("dsv4_tp_ckpt")
    raw = _raw_checkpoint()
    save_file(raw, str(folder / "model-00001-of-00001.safetensors"))
    (folder / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {n: "model-00001-of-00001.safetensors" for n in raw}}
    ))
    (folder / "inference").mkdir()
    (folder / "inference" / "config.json").write_text(json.dumps(vars(_toy_args())))
    return str(folder)


def _iter_all(folder, monkeypatch, rank, size):
    from freetoken.models.deepseek_v4.weight import iter_weights

    _patch_tp(monkeypatch, rank, size)
    return {
        name: tensor.clone()
        for name, tensor in iter_weights(
            folder, torch.device("cpu"), include_moe_experts=False, include_non_moe=True
        )
    }


def test_iter_weights_tp2_shards_match_the_tp1_pass(tp_checkpoint, monkeypatch):
    full = _iter_all(tp_checkpoint, monkeypatch, rank=0, size=1)
    rank0 = _iter_all(tp_checkpoint, monkeypatch, rank=0, size=2)
    rank1 = _iter_all(tp_checkpoint, monkeypatch, rank=1, size=2)
    assert set(rank0) == set(full) == set(rank1)
    for name, tensor in full.items():
        if name in ("embed.weight", "head"):
            assert torch.equal(rank0[name], _shard_vocab_rows(tensor, 0, 2)), name
            assert torch.equal(rank1[name], _shard_vocab_rows(tensor, 1, 2)), name
        else:
            # block-fp8 dense projections / norms / router / mHC all replicate
            assert torch.equal(rank0[name], tensor), name
            assert torch.equal(rank1[name], tensor), name
    assert rank0["embed.weight"].shape == (V // 2, D)
    assert rank1["head"].shape == (V // 2, D)
    assert rank0["layers.0.attn.wq_a.weight"].shape == (QL, D)  # replicated


def test_iter_weights_tp2_indivisible_vocab_raises(tp_checkpoint, monkeypatch):
    _patch_tp(monkeypatch, 0, 4)  # 130 % 4 != 0
    from freetoken.models.deepseek_v4.weight import iter_weights

    with pytest.raises(ValueError, match="vocab_size % tp_size"):
        list(iter_weights(
            tp_checkpoint, torch.device("cpu"),
            include_moe_experts=False, include_non_moe=True,
        ))


# ======================================================================================
# ds_fp4 expert-bank loaders: expert_shard filtering
# ======================================================================================

_EI, _EH = 32, 64  # expert-bank toy intermediate / hidden (H % 32 == 0 for e8m0 scales)


@pytest.fixture(scope="module")
def expert_checkpoint(tmp_path_factory):
    """4 experts x 1 layer (+ one MTP-layer tensor the loader must skip), fingerprinted:
    every byte of expert e's payload is e + 1."""
    from safetensors.torch import save_file

    folder = tmp_path_factory.mktemp("dsv4_tp_experts")
    raw: dict[str, torch.Tensor] = {}
    for layer in (0, 1):  # layer 1 == the MTP layer (args.n_layers == 1)
        for e in range(4):
            base = f"layers.{layer}.ffn.experts.{e}"
            raw[f"{base}.w1.weight"] = torch.full((_EI, _EH // 2), e + 1, dtype=torch.uint8)
            raw[f"{base}.w3.weight"] = torch.full((_EI, _EH // 2), e + 1, dtype=torch.uint8)
            raw[f"{base}.w2.weight"] = torch.full((_EH, _EI // 2), e + 1, dtype=torch.uint8)
            for proj, shape in (("w1", (_EI, _EH // 32)), ("w3", (_EI, _EH // 32)),
                                ("w2", (_EH, _EI // 32))):
                raw[f"{base}.{proj}.scale"] = torch.full(
                    shape, 120 + e, dtype=torch.uint8
                ).view(torch.float8_e8m0fnu)
    save_file(raw, str(folder / "model.safetensors"))
    (folder / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {}, "weight_map": {n: "model.safetensors" for n in raw}}
    ))
    return str(folder)


def _expert_args() -> DeepseekV4Args:
    return _toy_args(n_routed_experts=4, dim=_EH, moe_inter_dim=_EI)


def _check_rank_banks(banks, rank: int, size: int, E: int = 4):
    e_local = E // size
    assert banks["gate_up_packed"][0].shape == (e_local, 2 * _EI, _EH // 2)
    assert banks["down_packed"][0].shape == (e_local, _EH, _EI // 2)
    assert banks["gate_up_scale"][0].shape == (e_local, 2 * _EI, _EH // 32)
    assert banks["down_scale"][0].shape == (e_local, _EH, _EI // 32)
    for local in range(e_local):
        glob = local * size + rank
        assert (banks["gate_up_packed"][0][local] == glob + 1).all()
        assert (banks["down_packed"][0][local] == glob + 1).all()
        assert (banks["gate_up_scale"][0][local].view(torch.uint8) == 120 + glob).all()
        assert (banks["down_scale"][0][local].view(torch.uint8) == 120 + glob).all()


def test_load_dsfp4_expert_sources_tp_shard(expert_checkpoint):
    from freetoken.models.deepseek_v4.weight import load_dsfp4_expert_sources

    args = _expert_args()
    for rank in (0, 1):
        # layer_sink: no pinning (CPU-only test); the sink owns nothing here
        banks = load_dsfp4_expert_sources(
            expert_checkpoint, args, layer_sink=lambda layer_id, b: None,
            expert_shard=(rank, 2),
        )
        _check_rank_banks(banks, rank, 2)


def test_load_dsfp4_expert_sources_parallel_tp_shard(expert_checkpoint):
    from freetoken.models.deepseek_v4.weight import load_dsfp4_expert_sources_parallel

    args = _expert_args()
    for rank in (0, 1):
        banks = load_dsfp4_expert_sources_parallel(
            expert_checkpoint, args, layer_sink=lambda layer_id, b: None,
            expert_shard=(rank, 2),
        )
        _check_rank_banks(banks, rank, 2)


def test_load_dsfp4_expert_sources_ambient_tp_shard(expert_checkpoint, monkeypatch):
    """No explicit expert_shard -> the ambient TP info drives the split."""
    from freetoken.models.deepseek_v4.weight import load_dsfp4_expert_sources

    args = _expert_args()
    _patch_tp(monkeypatch, 1, 2)
    banks = load_dsfp4_expert_sources(
        expert_checkpoint, args, layer_sink=lambda layer_id, b: None
    )
    _check_rank_banks(banks, 1, 2)


def test_load_dsfp4_expert_sources_indivisible_raises(expert_checkpoint):
    from freetoken.models.deepseek_v4.weight import load_dsfp4_expert_sources

    with pytest.raises(ValueError, match="n_routed_experts % tp_size"):
        load_dsfp4_expert_sources(
            expert_checkpoint, _expert_args(), layer_sink=lambda layer_id, b: None,
            expert_shard=(0, 3),  # 4 % 3 != 0
        )


def test_dummy_dsfp4_expert_sources_tp_shard(monkeypatch):
    from freetoken.models.deepseek_v4.weight import dummy_dsfp4_expert_sources

    monkeypatch.setenv("FREETOKEN_SKIP_BANK_PIN", "1")
    args = _expert_args()
    banks = dummy_dsfp4_expert_sources(args, expert_shard=(0, 2))
    assert banks["gate_up_packed"][0].shape == (2, 2 * _EI, _EH // 2)
    assert banks["down_scale"][0].shape == (2, _EH, _EI // 32)
    # ambient resolution (dummy-weight boot under TP)
    _patch_tp(monkeypatch, 1, 2)
    banks = dummy_dsfp4_expert_sources(args)
    assert banks["gate_up_packed"][0].shape == (2, 2 * _EI, _EH // 2)
    with pytest.raises(ValueError, match="n_routed_experts % tp_size"):
        dummy_dsfp4_expert_sources(args, expert_shard=(0, 3))


# ======================================================================================
# Offload MoE prefill at tp=2, two simulated ranks vs the full layer (CUDA)
# ======================================================================================


def _torch_copy_missing(cache) -> None:
    """copy_missing without the fast_index_copy JIT: the same staged rows via index_copy_
    (fp8 dtypes go through their uint8 views -- CUDA index_copy_ is not implemented for them)."""
    n = int(cache.num_indices.sum().item())
    if n == 0:
        return
    dst = cache.evict_slots[:n].long()
    src = cache.src_indices[:n].long()
    for per_layer, cache_tensor in cache.banks:
        rows = per_layer[cache._pending_src_layer]
        if cache_tensor.dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn, torch.float8_e5m2):
            cache_tensor.view(torch.uint8).index_copy_(0, dst, rows.view(torch.uint8)[src])
        else:
            cache_tensor.index_copy_(0, dst, rows[src])


def _full_dsfp4_banks(dev, E, H, I, seed=0):
    g = torch.Generator().manual_seed(seed)

    def rows(out, inp):
        packed = torch.randint(0, 256, (E, out, inp // 2), dtype=torch.uint8, generator=g)
        scale = torch.randint(122, 128, (E, out, inp // 32), dtype=torch.uint8, generator=g)
        return packed, scale.view(torch.float8_e8m0fnu).contiguous()

    gup, gus = rows(2 * I, H)
    dnp, dns = rows(H, I)
    return {  # one [E, ...] tensor per layer (num_layers == 1)
        "gate_up_packed": [gup.to(dev)], "gate_up_scale": [gus.to(dev)],
        "down_packed": [dnp.to(dev)], "down_scale": [dns.to(dev)],
    }


def _rank_moe_layer_and_cache(monkeypatch, rank, size, banks, E, H, I, dev):
    """One rank's DSV4OffloadMoELayer + slot cache over its owned experts (id % size == rank)."""
    from freetoken.models.deepseek_v4.moe import DSV4OffloadMoELayer
    from freetoken.moe.offload_cache import OffloadMoeCache

    _patch_tp(monkeypatch, rank, size)
    layer = DSV4OffloadMoELayer(
        layer_id=0,
        args=_toy_args(dim=H, moe_inter_dim=I, n_routed_experts=E, n_activated_experts=3),
    )
    n_local = E // size
    cache = OffloadMoeCache(
        num_layers=1, num_experts=n_local, cache_size=n_local, device=dev,
        quant_format="ds_fp4",
    )
    cache.set_bank_sources({k: [v[0][rank::size].contiguous()] for k, v in banks.items()})
    monkeypatch.setattr(cache, "copy_missing", lambda c=cache: _torch_copy_missing(c))
    layer.offload_cache = cache
    return layer, cache


_MH, _MI2, _ME = 256, 256, 8  # MoE toy hidden / intermediate / experts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offload_prefill_on_demand_tp2_matches_full(monkeypatch, noop_comm):
    """The on-demand slot path (T*top_k < num_experts_global): the pre-TP code routed
    GLOBAL expert ids into the per-rank cache; the fix partitions first (local ids,
    non-owned -> weight 0 / dump row 0), and the two halves sum to the full layer."""
    dev = torch.device("cuda")
    banks = _full_dsfp4_banks(dev, _ME, _MH, _MI2)
    torch.manual_seed(7)
    hidden = torch.randn(2, _MH, dtype=torch.bfloat16, device=dev)  # 2*3=6 < 8: on-demand
    weights = torch.tensor(
        [[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]], dtype=torch.float32, device=dev
    )
    ids = torch.tensor([[0, 3, 5], [1, 2, 7]], dtype=torch.int32, device=dev)

    ref_layer, _ = _rank_moe_layer_and_cache(monkeypatch, 0, 1, banks, _ME, _MH, _MI2, dev)
    ref = ref_layer._prefill_routed(hidden, weights, ids.clone())

    partials = []
    seen: list[tuple[int, torch.Tensor]] = []
    for rank in (0, 1):
        layer, cache = _rank_moe_layer_and_cache(monkeypatch, rank, 2, banks, _ME, _MH, _MI2, dev)
        assert layer.num_experts == 4 and layer.num_experts_global == 8
        orig_ensure = cache.ensure_experts

        def rec(layer_id, eids, orig=orig_ensure, rank=rank):
            seen.append((rank, eids.clone()))  # before the in-place slot remap
            return orig(layer_id, eids)

        monkeypatch.setattr(cache, "ensure_experts", rec)
        partials.append(layer._prefill_routed(hidden, weights, ids.clone()))
    # ensure_experts saw only LOCAL ids (owned experts + the clamped dump row 0)
    for rank, eids in seen:
        assert int(eids.min()) >= 0 and int(eids.max()) < _ME // 2
    # bf16 partials sum with large cancelling terms, so compare relative to the
    # output magnitude (the repo-wide ds_fp4 convention, cf. tests/moe/test_cpu_moe)
    s = (partials[0] + partials[1]).float()
    r = ref.float()
    rel = (s - r).abs().max() / (r.abs().max() + 1e-6)
    assert rel < 4e-3, f"on-demand tp2 rel err {rel.item()}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offload_prefill_streaming_tp2_matches_full(monkeypatch, noop_comm):
    """The whole-layer streaming path (T*top_k >= num_experts_global): the base
    materialize path with the partitioned routing (-1 clamped onto row 0)."""
    dev = torch.device("cuda")
    banks = _full_dsfp4_banks(dev, _ME, _MH, _MI2)
    torch.manual_seed(9)
    hidden = torch.randn(3, _MH, dtype=torch.bfloat16, device=dev)  # 3*3=9 >= 8: streaming
    weights = torch.full((3, 3), 1.0 / 3, dtype=torch.float32, device=dev)
    ids = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 0]], dtype=torch.int32, device=dev)

    ref_layer, _ = _rank_moe_layer_and_cache(monkeypatch, 0, 1, banks, _ME, _MH, _MI2, dev)
    ref = ref_layer._prefill_routed(hidden, weights, ids.clone())

    partials = []
    for rank in (0, 1):
        layer, _cache = _rank_moe_layer_and_cache(monkeypatch, rank, 2, banks, _ME, _MH, _MI2, dev)
        partials.append(layer._prefill_routed(hidden, weights, ids.clone()))
    s = (partials[0] + partials[1]).float()
    r = ref.float()
    rel = (s - r).abs().max() / (r.abs().max() + 1e-6)
    assert rel < 4e-3, f"streaming tp2 rel err {rel.item()}"


def test_moe_layer_indivisible_experts_raise(monkeypatch):
    from freetoken.models.deepseek_v4.moe import DSV4OffloadMoELayer

    _patch_tp(monkeypatch, 0, 2)
    with pytest.raises(AssertionError, match="num_experts % tp_size"):
        DSV4OffloadMoELayer(layer_id=0, args=_toy_args(n_routed_experts=7))


# ======================================================================================
# Transformer embed / head wiring (CUDA: the indexing kernel is CUDA-only)
# ======================================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_embed_tp2_partials_sum_to_full(monkeypatch, noop_comm):
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    full_w = torch.randn(V, D, dtype=torch.bfloat16, generator=g).to(dev)
    ids = torch.randint(0, V, (17,), generator=g).to(dev)

    m_full = _toy_model(monkeypatch, 0, 1)
    m_full._transformer.embed.weight = full_w
    ref = m_full._transformer._embed_tokens(ids.view(1, -1))

    parts = []
    for rank in (0, 1):
        m = _toy_model(monkeypatch, rank, 2)
        m._transformer.embed.weight = _shard_vocab_rows(full_w, rank, 2).contiguous()
        out = m._transformer._embed_tokens(ids.view(1, -1))
        assert out.shape == ref.shape  # [1, T, dim] on every rank
        parts.append(out)
    # embedding gather is exact; the mask zeroes non-owned rows, so the sum is exact
    torch.testing.assert_close((parts[0] + parts[1]).float(), ref.float(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_head_tp2_local_logits_match_full(monkeypatch):
    """Each rank's vocab shard produces exactly the corresponding full-logit slice;
    the dup-gather stub places it at its own slot of the reassembled [B, vocab]."""
    from freetoken.distributed import DistributedCommunicator
    from freetoken.distributed.impl import DistributedImpl

    class _DupGather(DistributedImpl):
        def all_reduce(self, x):
            return x

        def all_gather(self, x):  # both slots = this rank's local logits
            return torch.cat([x, x], dim=0)

    monkeypatch.setattr(DistributedCommunicator, "plugins", [_DupGather()])
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(1)
    full_w = torch.randn(V, D, dtype=torch.bfloat16, generator=g).to(dev)
    h = torch.randn(5, D, dtype=torch.bfloat16, generator=g).to(dev)
    ref = torch.nn.functional.linear(h, full_w)

    vt = V // 2
    for rank in (0, 1):
        m = _toy_model(monkeypatch, rank, 2)
        m._transformer.head.weight = _shard_vocab_rows(full_w, rank, 2).contiguous()
        out = m._transformer._head_logits(h)
        assert out.shape == (5, V)
        lo = rank * vt
        torch.testing.assert_close(out[:, lo:lo + vt], ref[:, lo:lo + vt])


# ======================================================================================
# Full-model state-dict shape + load contract at tp=2 (meta device, CPU)
# ======================================================================================


def test_full_model_state_dict_tp2_contract(monkeypatch):
    """Every tp=2 state-dict key matches the tp=1 key set; only embed/head change
    shape (vocab halves), and load_state_dict accepts exactly the sharded dict --
    the contract the engine's weight load enforces at boot."""
    full = _toy_model(monkeypatch, 0, 1).state_dict()
    assert full["embed.weight"].shape == (V, D)
    assert full["head"].shape == (V, D)
    for rank in (0, 1):
        m = _toy_model(monkeypatch, rank, 2)
        sd = m.state_dict()
        assert set(sd) == set(full)
        for name, t in full.items():
            want = (V // 2, D) if name in ("embed.weight", "head") else tuple(t.shape)
            assert tuple(sd[name].shape) == tuple(want), name
        sharded = {
            name: (_shard_vocab_rows(t, rank, 2) if name in ("embed.weight", "head") else t)
            for name, t in full.items()
        }
        m.load_state_dict(sharded)  # raises on missing/unexpected/shape mismatch


def test_full_model_tp1_state_dict_unchanged(monkeypatch):
    """TP=1: the key set and shapes are exactly the pre-TP contract."""
    sd = _toy_model(monkeypatch, 0, 1).state_dict()
    assert "embed.weight" in sd and "head" in sd
    assert "layers.0.attn.wq_a.weight" in sd
    assert sd["embed.weight"].shape == (V, D)
    assert sd["head"].shape == (V, D)
    assert sd["layers.0.ffn.gate.weight"].shape == (E, D)
