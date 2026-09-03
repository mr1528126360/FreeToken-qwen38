"""GLM-5.3-Flash MoE / dense-MLP site (``freetoken.models.glm5_next.moe``).

The reference is the module this port was made against,
``transformers.models.glm5_next.modeling_glm5_next`` -- the real HF classes are used
wherever they are buildable without a checkpoint (``Glm5NextTextTopkRouter`` and
``Glm5NextTextMLP`` only read plain scalars off their config), so the checks run against
the reference and not against a re-transcription of the same code. What is pinned:

* the ``noaux_tc`` routing decision (sigmoid scores, ``e_score_correction_bias`` used to
  PICK but never to WEIGHT, renormalize-then-scale) is id-exact against HF on a non-round
  batch of 17, with and without group limiting, with and without renormalization, and on a
  knife-edge near-tie at the top-k boundary;
* the layer split (dense 0/1/2, sparse elsewhere) and the *compacted* expert-bank rows
  0..41 the sparse blocks must hand to ``make_moe_layer``;
* the clamped SwiGLU of the shared/dense MLP equals ``Glm5NextTextMLP`` where the clamp
  bites, and a resident-bf16 block reproduces the reference expert math (``_apply_gate``
  + per-route weighting + the UNGATED shared-expert add).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("torch")

from freetoken.utils import torch_dtype  # noqa: E402  (must follow the importorskip)

# Shrunk-but-real geometry: experts 8 / top-2 / moe_inter 16 keeps the resident bf16 fused
# kernels exerciseable; swiglu_limit 10 and routed_scaling_factor 2.5 are the checkpoint's.
HIDDEN, N_EXP, TOP_K, MOE_INTER, DENSE_INTER = 32, 8, 2, 16, 64
LIMIT, RSF = 10.0, 2.5

CFG = {
    "hidden_size": HIDDEN,
    "vocab_size": 64,
    "num_hidden_layers": 6,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "num_attention_heads": 4,
    "q_lora_rank": 16,
    "kv_lora_rank": 16,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 0,
    "v_head_dim": 16,
    "max_position_embeddings": 256,
    "index_head_dim": 0,
    "index_topk": 0,
    "linear_attn_config": {"kda_layers": [], "full_attn_layers": []},
    "n_routed_experts": N_EXP,
    "num_experts_per_tok": TOP_K,
    "moe_intermediate_size": MOE_INTER,
    "n_shared_experts": 1,
    "routed_scaling_factor": RSF,
    "n_group": 1,
    "topk_group": 1,
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "swiglu_limit": LIMIT,
    "intermediate_size": DENSE_INTER,
    "mlp_layer_types": ["sparse"] * 6,
}


def _config(**over):
    """A real ``ModelConfig`` (``glm_dsa_args`` included) from a toy HF dict, so the
    payload and the ModelConfig fields agree exactly as ``parse_config`` makes them."""
    from freetoken.models.glm5_next.config import parse_config

    return parse_config({**CFG, **over})


def _sparse_block(cfg, layer_id=3, dtype=torch.bfloat16, device="cpu", seed=1):
    """Build + fill the block (its weights are ``torch.empty`` placeholders)."""
    from freetoken.models.glm5_next.moe import Glm5NextSparseBlock

    with torch.device(device), torch_dtype(dtype):
        block = Glm5NextSparseBlock(cfg, layer_id)
    gen = torch.Generator(device=device).manual_seed(seed)
    for t in block.state_dict().values():
        if t.is_floating_point():
            t.normal_(0.0, 0.05, generator=gen)
    return block


def _hf_router(block, cfg):
    """The HF router carrying THIS block's weights -- the strictest reference available."""
    mod = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    router = mod.Glm5NextTextTopkRouter(
        SimpleNamespace(
            num_experts_per_tok=cfg.num_experts_per_tok,
            num_local_experts=cfg.num_experts,
            hidden_size=cfg.hidden_size,
            routed_scaling_factor=cfg.routed_scaling_factor,
            n_group=cfg.n_group,
            topk_group=cfg.topk_group,
            norm_topk_prob=cfg.norm_topk_prob,
        )
    )
    with torch.no_grad():
        router.weight.copy_(block.gate.weight)
        router.e_score_correction_bias.copy_(block.e_score_correction_bias)
    return router.to(block.gate.weight.dtype)


def _by_id(ids, weights):
    """Canonical (id-sorted) view of a routing. HF asks ``topk`` for an UNSORTED answer
    (modeling_glm5_next.py:177) and we take the sorted one; a route is identified by its
    expert, not by the slot it lands in, so the comparison is order-insensitive."""
    order = ids.long().argsort(-1)
    return ids.long().gather(-1, order), weights.float().gather(-1, order)


# --------------------------------------------------------------------------------------
# (a) routing parity with HF Glm5NextTextTopkRouter
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("norm_topk_prob", [True, False])
@pytest.mark.parametrize("n_group,topk_group", [(1, 1), (4, 2), (2, 1)])
def test_route_matches_hf_router(n_group, topk_group, norm_topk_prob):
    """The checkpoint routes grouplessly (``n_group = topk_group = 1``, where the
    reference's group mask keeps everything and we skip it); the group-limited branches
    (which ``_group_limited`` implements for other GLM geometries) must agree too."""
    cfg = _config(n_group=n_group, topk_group=topk_group, norm_topk_prob=norm_topk_prob)
    block = _sparse_block(cfg, seed=100 * n_group + topk_group)
    x = torch.randn(17, HIDDEN, dtype=torch.bfloat16) * 0.5

    weights, ids = block._route(x)
    _, ref_weights, ref_ids = _hf_router(block, cfg)(x)

    assert weights.dtype is torch.float32 and ids.dtype is torch.int32
    assert weights.shape == ids.shape == (17, TOP_K)
    assert weights.is_contiguous() and ids.is_contiguous()

    mine_ids, mine_w = _by_id(ids, weights)
    ref_ids, ref_w = _by_id(ref_ids, ref_weights)
    assert torch.equal(mine_ids, ref_ids), "topk_ids must match HF exactly"
    torch.testing.assert_close(mine_w, ref_w, rtol=1e-3, atol=1e-6)


def test_selection_bias_picks_but_never_scales():
    """A boundary fixture on a 17-token batch, in fp32 so the gaps survive: expert 1 is a
    literal clone of expert 0 (their scores are bitwise equal) and the boosted pair is
    parked within 1e-5 of the strongest rival, i.e. ON the top-k line. Any deviation in
    the bias handling -- or in the fp32 logits -- flips a pick here. The weights must
    still be the raw sigmoid scores, renormalized and only then scaled (:162/:178-182)."""
    cfg = _config()
    block = _sparse_block(cfg, dtype=torch.float32, seed=7)
    with torch.no_grad():  # 0.2-scale gate: scores spread wide enough for a real rival
        block.gate.weight.normal_(0.0, 0.2)
        block.gate.weight[1] = block.gate.weight[0]
        block.e_score_correction_bias.zero_()
    x = torch.randn(17, HIDDEN, dtype=torch.float32) * 0.8
    scores = torch.sigmoid(F.linear(x, block.gate.weight))
    rival = scores[0].clone()
    rival[[0, 1]] = -1.0
    top = rival.max()
    with torch.no_grad():  # e0 a hair above e1, e1 a hair above the rival: 5e-6 apart
        block.e_score_correction_bias[0] = float(top - scores[0, 0] + 1e-5)
        block.e_score_correction_bias[1] = float(top - scores[0, 0] + 5e-6)

    weights, ids = block._route(x)
    _, ref_weights, ref_ids = _hf_router(block, cfg)(x)

    mine_ids, mine_w = _by_id(ids, weights)
    assert torch.equal(mine_ids, _by_id(ref_ids, ref_weights)[0])
    assert mine_ids[0].tolist() == [0, 1], "the boosted pair must clear the top-k line"
    # The fixture really is a boundary case: e0 > e1 > the best rival, 5e-6 apart.
    chosen = scores[0] + block.e_score_correction_bias
    rival = chosen.clone()
    rival[[0, 1]] = -1.0
    assert chosen[0] - chosen[1] == pytest.approx(5e-6, rel=0.3)
    assert chosen[1] - rival.max() == pytest.approx(5e-6, rel=0.3)

    expected = scores.gather(1, mine_ids)  # the UNBIASED scores are the weights
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * RSF
    torch.testing.assert_close(mine_w, expected, rtol=1e-4, atol=1e-7)
    assert (mine_w <= RSF * (1 + 1e-6)).all(), "a biased weight could exceed the 2.5x ceiling"


# --------------------------------------------------------------------------------------
# (b) dense/sparse split and the compacted expert-bank rows
# --------------------------------------------------------------------------------------


def test_layer_split_and_compacted_bank_rows(monkeypatch):
    """``make_mlp`` follows ``mlp_layer_types`` (dense 0/1/2, scattered -- not a prefix),
    so the experts are numbered by BANK ROW ``args.moe_bank_index(layer_id)`` and NOT by
    ``layer_id - first_k_dense_replace``, which ``parse_config`` pins to 0."""
    from freetoken.models.glm5_next import moe as moe_module
    from freetoken.models.glm5_next.moe import Glm5NextDenseMLP, Glm5NextSparseBlock

    cfg = _config(
        num_hidden_layers=45,
        mlp_layer_types=["dense", "dense", "dense"] + ["sparse"] * 42,
    )
    args = cfg.glm_dsa_args
    assert args.dense_mlp_layer_ids == (0, 1, 2)
    assert args.num_moe_layers == 42 and args.moe_layer_ids == tuple(range(3, 45))
    assert cfg.first_k_dense_replace == 0  # a prefix count cannot describe this split

    asked, real = [], moe_module.make_moe_layer

    def spy(config, **kwargs):
        asked.append(kwargs["layer_id"])
        return real(config, **kwargs)

    monkeypatch.setattr(moe_module, "make_moe_layer", spy)
    with torch.device("cpu"), torch_dtype(torch.bfloat16):
        blocks = [moe_module.make_mlp(cfg, lid) for lid in range(45)]

    assert [isinstance(b, Glm5NextDenseMLP) for b in blocks[:3]] == [True] * 3
    assert all(isinstance(b, Glm5NextDenseMLP) == (lid in (0, 1, 2)) for lid, b in enumerate(blocks))
    assert all(isinstance(b, Glm5NextSparseBlock) for b in blocks[3:])
    # One bank per sparse layer, in sparse order: 0..41 with no gaps and no duplicates.
    assert asked == list(range(42))
    assert asked[0] == 3 - 3 and asked[-1] == 41
    # The layer -> bank map is the compressed one (decoder id 3 is bank 0, id 44 is bank 41)
    assert args.moe_bank_index(3) == 0 and args.moe_bank_index(44) == 41
    assert args.moe_bank_index(0) is None
    assert all(getattr(b.experts, "swiglu_limit", None) == LIMIT for b in blocks[3:])

    dense = blocks[0]
    assert tuple(dense.gate_proj.weight.shape) == (DENSE_INTER, HIDDEN)
    assert tuple(dense.up_proj.weight.shape) == (DENSE_INTER, HIDDEN)
    assert tuple(dense.down_proj.weight.shape) == (HIDDEN, DENSE_INTER)
    assert dense.swiglu_limit == LIMIT
    sparse = blocks[3]
    assert tuple(sparse.gate.weight.shape) == (N_EXP, HIDDEN)
    assert sparse.e_score_correction_bias.shape == (N_EXP,)
    # shared expert: moe_intermediate_size * n_shared_experts (reference :197), NOT the
    # unused shared_expert_intermediate_size config field.
    assert tuple(sparse.shared_experts.gate_proj.weight.shape) == (MOE_INTER, HIDDEN)
    assert sparse.shared_experts.swiglu_limit == LIMIT


# --------------------------------------------------------------------------------------
# (c) dense MLP math + the resident expert forward
# --------------------------------------------------------------------------------------


def test_dense_mlp_matches_hf_text_mlp_where_the_clamp_bites():
    """``Glm5NextTextMLP.forward`` (:98-104) clamps ``gate`` from above and ``up``
    symmetrically BEFORE the plain silu -- and the SAME class serves both the dense-MLP
    layers and every sparse layer's shared expert (:196, :1271), so both carry the limit.
    Activations far past +-10 make a missing clamp observable, not silently tolerable."""
    mod = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from freetoken.models.glm5_next.moe import Glm5NextDenseMLP

    with torch.device("cpu"), torch_dtype(torch.float32):
        mlp = Glm5NextDenseMLP(HIDDEN, DENSE_INTER, swiglu_limit=LIMIT)
        naked = Glm5NextDenseMLP(HIDDEN, DENSE_INTER, swiglu_limit=None)
    gen = torch.Generator().manual_seed(3)
    for proj in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        proj.weight.normal_(0.0, 0.3, generator=gen)
    with torch.no_grad():  # the unclamped twin, same weights: the clamp's fingerprint
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(naked, name).weight.copy_(getattr(mlp, name).weight)

    ref = mod.Glm5NextTextMLP(
        SimpleNamespace(
            hidden_size=HIDDEN, intermediate_size=DENSE_INTER, swiglu_limit=LIMIT, hidden_act="silu"
        )
    )
    with torch.no_grad():  # the checkpoint ships these projections bias-free
        for name in ("gate_proj", "up_proj", "down_proj"):
            getattr(ref, name).weight.copy_(getattr(mlp, name).weight)

    x = torch.randn(9, HIDDEN, dtype=torch.float32) * 60.0
    assert (F.linear(x, mlp.gate_proj.weight) > LIMIT).any(), "fixture must clamp"
    assert F.linear(x, mlp.up_proj.weight).abs().gt(LIMIT).any(), "fixture must clamp"
    got = mlp.forward(x)
    assert got.shape == (9, HIDDEN)
    torch.testing.assert_close(got, ref(x), rtol=1e-5, atol=1e-3)
    assert (got - naked.forward(x)).abs().max() > 1.0, "the clamp changed nothing: bad fixture"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sparse_block_forward_matches_reference_experts():
    """The full block on resident bf16 experts (8 experts, top-2, inter 16, tp=1): routed
    output + shared expert, against an fp32 restatement of ``Glm5NextTextExperts.forward``
    (:120-142). Also the guard that the shared expert reads the UNROUTED activations --
    the fused expert kernels overwrite ``hidden_states`` in place."""
    from freetoken.models.glm5_next.moe import Glm5NextSparseBlock

    block = _sparse_block(_config(), device="cuda", seed=11)
    x = (torch.randn(17, HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.3).contiguous()
    out = block.forward(x.clone())
    assert out.shape == x.shape and out.dtype is torch.bfloat16
    assert torch.isfinite(out).all(), "resident expert forward produced NaN"

    weights, ids = block._route(x)
    gate_up = block.experts.gate_up_proj.float()[ids.long()]  # [T, k, 2I, H]
    down = block.experts.down_proj.float()[ids.long()]  # [T, k, H, I]
    gu = torch.einsum("th,ekth->ekt", x.float(), gate_up)
    act = (
        F.silu(gu[..., :MOE_INTER].clamp(max=LIMIT))
        * gu[..., MOE_INTER:].clamp(min=-LIMIT, max=LIMIT)
    )
    routed = (torch.matmul(act, down.transpose(-1, -2)) * weights[..., None]).sum(1)
    ref = routed + block.shared_experts.forward(x).float()

    err = (out.float() - ref).abs().max().item()
    assert err < 5e-2, f"block output drifted from the reference experts ({err})"
