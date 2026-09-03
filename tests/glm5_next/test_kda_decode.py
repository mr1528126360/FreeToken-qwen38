"""GLM-5.3-Flash KDA decode (the raw-g branch of the vendored fla kernel) vs a torch oracle.

The oracle is a packed-buffer transcription of
``transformers/models/glm5_next/modeling_glm5_next.py::recurrent_kimi_delta_attention``:
fp32 casts, ``l2norm(q, k, eps=1e-6)``, ``scale = K**-0.5`` on q, then per token
``state *= exp(g); kv = (state*k).sum(-2); state += k (outer) (v-kv)*beta; out = (state*q).sum(-2)``
with ``state`` semantically ``[K, V]``. ``g`` is the bounded-sigmoid forget gate
(``Glm5NextTextForgetGate``, ``lower_bound = -5``) computed model-side and handed to the kernel
VERBATIM -- which is exactly what ``USE_RAW_G`` buys. Random tensors only, no checkpoint.

The last test pins the legacy path: with the new flags left False, ``gdn_decode_fla`` must still
reproduce the in-kernel ``g = -exp(A_log)*softplus(a + dt_bias)`` / ``beta = sigmoid(b)`` GDN
math, so no other model can be affected by the switch.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = torch.device("cuda")
LOWER_BOUND = -5.0  # text_config linear_attn_config.gate_lower_bound
REL_TOL = 2e-2
# The pool's last two dims are addressed as [V, K] by BOTH fla kernels (offset ``v*K + k``),
# i.e. the torch view of a ``[slots, HV, K, V]`` pool holds the kernel state TRANSPOSED.
# Every config below therefore keeps K == V, as the serving shapes do.
POOL_TRANSPOSED = True


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """HF glm5_next ``l2norm`` (sqrt + eps, not max(..., eps)) -- same formula as the kernel."""
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _forget_gate(hidden, f_a_w, f_b_w, a_log, dt_bias, head_dim) -> torch.Tensor:
    """``Glm5NextTextForgetGate.forward``: [B, T, H, K] fp32 log-decrement in [-5, 0)."""
    lin = torch.nn.functional.linear
    gate = lin(lin(hidden, f_a_w), f_b_w).float()
    b, t = hidden.shape[:2]
    gate = (gate + dt_bias.float()).view(b, t, -1, head_dim)
    decay = torch.exp(a_log.float()).view(1, 1, -1, 1)
    return LOWER_BOUND * torch.sigmoid(decay * gate)


def _ref_kda(q, k, v, g, beta, state0, lengths, scale):
    """Single/multi-token KDA recurrence over a packed token axis.

    q, k: [total, H, K] (H == HV here, as glm5_next has no KDA GQA); v: [total, HV, V];
    g: [total, HV, K] per-channel log-decrement, or [total, HV] for GDN's scalar-per-head
    gate (it broadcasts over K); beta: [total, HV]; state0: [N, HV, K, V] fp32.
    Returns ([total, HV, V], [N, HV, K, V]) in fp32.
    """
    hv, head_k, head_v = state0.shape[1:]
    outs, finals, off = [], [], 0
    for i, n in enumerate(lengths):
        state = state0[i].clone()
        for t in range(off, off + n):
            q_i = _l2norm(q[t].float()) * scale               # [HV, K]
            k_i = _l2norm(k[t].float())                       # [HV, K]
            v_i = v[t].float()                                # [HV, V]
            g_i = g[t].reshape(hv, -1).float().exp().unsqueeze(-1)   # [HV, K|1, 1]
            b_i = beta[t].float().unsqueeze(-1)               # [HV, 1]
            state = state * g_i
            kv_mem = (state * k_i.unsqueeze(-1)).sum(dim=-2)  # [HV, V]
            delta = (v_i - kv_mem) * b_i
            state = state + k_i.unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((state * q_i.unsqueeze(-1)).sum(dim=-2))
        finals.append(state)
        off += n
    return torch.stack(outs), torch.stack(finals)


def _make_case(num_heads, head_dim, lengths, slots, seed, num_slots=8):
    """Random bf16 qkv + fp32 gates/state for one packed decode batch."""
    total = sum(lengths)
    torch.manual_seed(seed)
    qkv = torch.randn(total, 3 * num_heads * head_dim, device=DEV, dtype=torch.bfloat16)
    qf, kf, vf = qkv.split([num_heads * head_dim] * 3, dim=-1)
    hidden = torch.randn(1, total, 64, device=DEV, dtype=torch.bfloat16)
    f_a_w = torch.randn(head_dim, 64, device=DEV, dtype=torch.bfloat16) * 0.2
    f_b_w = torch.randn(num_heads * head_dim, head_dim, device=DEV, dtype=torch.bfloat16) * 0.2
    a_log = torch.empty(num_heads, device=DEV).uniform_(0.01, 16.0).log_()
    dt_bias = torch.randn(num_heads * head_dim, device=DEV) * 2.0
    g = _forget_gate(hidden, f_a_w, f_b_w, a_log, dt_bias, head_dim)   # [1, total, HV, K]
    beta = torch.sigmoid(torch.randn(total, num_heads, device=DEV) * 2.0)
    state0 = torch.randn(len(lengths), num_heads, head_dim, head_dim, device=DEV) * 0.5
    pool = torch.zeros(num_slots, num_heads, head_dim, head_dim, device=DEV)
    pool[slots] = state0.transpose(-1, -2) if POOL_TRANSPOSED else state0
    return dict(
        q=qf.view(total, num_heads, head_dim),
        k=kf.view(total, num_heads, head_dim),
        v=vf.view(total, num_heads, head_dim),
        g=g, beta=beta, state0=state0, pool=pool, lengths=list(lengths),
        slots=list(slots), total=total,
        indices=torch.tensor(slots, device=DEV, dtype=torch.int32),
        cu_seqlens=torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                                device=DEV, dtype=torch.int64),
    )


def _gate(case, layout):
    """Hand the log-decrement over in one of the layouts the kernel wrapper has to accept."""
    g = case["g"]
    if layout == "4d":
        return g                                  # [B, T, HV, K] -> wrapper flattens the tail
    if layout == "3d":
        return g.reshape(1, case["total"], -1)    # [B, T, HV*K]
    return g.reshape(case["total"], -1)           # [T, HV*K], the KDA serving layout


def _run(case, num_heads, head_dim, layout):
    from freetoken.models.glm5_next.kda_kernels import kda_decode_fla

    total = case["total"]
    out = kda_decode_fla(
        case["q"].reshape(1, total, num_heads, head_dim),
        case["k"].reshape(1, total, num_heads, head_dim),
        case["v"].reshape(1, total, num_heads, head_dim),
        _gate(case, layout), case["beta"],
        state_source=case["pool"], indices=case["indices"],
        cu_seqlens=case["cu_seqlens"], scale=head_dim ** -0.5,
    )
    ref_out, ref_state = _ref_kda(
        case["q"], case["k"], case["v"],
        case["g"].reshape(total, num_heads, head_dim), case["beta"],
        case["state0"], case["lengths"], head_dim ** -0.5,
    )
    return out, ref_out, ref_state


def _rel_err(a, b) -> float:
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("layout", ["4d", "3d", "2d"])
@pytest.mark.parametrize("lengths,slots", [
    ([1], [0]), ([1], [5]), ([1, 1, 1, 1], [0, 1, 2, 3]),
    ([1, 1, 1], [7, 2, 5]), ([2, 5, 1], [6, 1, 4]), ([3, 4], [0, 7]),
])
@pytest.mark.parametrize("num_heads,head_dim", [(4, 128), (4, 64), (8, 128)])
def test_kda_decode_matches_reference(num_heads, head_dim, lengths, slots, layout):
    """T=1 decode batches plus multi-token varlen, contiguous and scattered slot ids."""
    case = _make_case(num_heads, head_dim, lengths, slots, seed=len(lengths) * 31 + num_heads)
    out, ref_out, ref_state = _run(case, num_heads, head_dim, layout)

    assert out.shape == (case["total"], num_heads, head_dim)
    assert out.dtype == torch.bfloat16
    assert _rel_err(out, ref_out) < REL_TOL, "decode output drifted from the KDA oracle"
    got = case["pool"][case["slots"]]
    got = got.transpose(-1, -2) if POOL_TRANSPOSED else got
    assert _rel_err(got, ref_state) < REL_TOL, "state read/write-by-slot drifted"
    for s in set(range(case["pool"].shape[0])) - set(case["slots"]):
        assert torch.count_nonzero(case["pool"][s]) == 0, f"slot {s} clobbered"


def test_kda_decode_realistic_head_count():
    """The glm5_next KDA shape (64 heads x 128) on 3 non-contiguous slots."""
    num_heads, head_dim, lengths, slots = 64, 128, [1, 1, 1], [3, 0, 11]
    case = _make_case(num_heads, head_dim, lengths, slots, seed=101, num_slots=16)
    out, ref_out, ref_state = _run(case, num_heads, head_dim, "2d")
    assert _rel_err(out, ref_out) < REL_TOL
    assert _rel_err(case["pool"][slots].transpose(-1, -2), ref_state) < REL_TOL


def test_legacy_gdn_path_is_unchanged():
    """``USE_RAW_G`` / ``USE_RAW_BETA`` default to False: GDN decode still derives the gate
    in-kernel from A_log/dt_bias and sigmoids ``b``. If the flag leaked into that path (or the
    softplus branch broke), the oracle below stops matching."""
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_decode_fla

    torch.manual_seed(3)
    num_k, num_v, head_dim = 4, 8, 128  # GQA: v-head groups share a k head, as in qwen4_exp
    slots, head_k, head_v = [2, 5, 0], head_dim, head_dim
    qkv = torch.randn(1, 3, (2 * num_k + num_v) * head_dim, device=DEV, dtype=torch.bfloat16)
    qf, kf, vf = qkv.split([num_k * head_k, num_k * head_k, num_v * head_v], dim=-1)
    q = qf.reshape(1, 3, num_k, head_k)
    k = kf.reshape(1, 3, num_k, head_k)
    v = vf.reshape(1, 3, num_v, head_v)
    a_raw = torch.randn(3, num_v, device=DEV) * 2.0
    b_raw = torch.randn(3, num_v, device=DEV) * 2.0
    a_log = torch.empty(num_v, device=DEV).uniform_(0.01, 4.0).log_()
    dt_bias = torch.randn(num_v, device=DEV)
    pool = torch.zeros(8, num_v, head_k, head_v, device=DEV)
    scale = head_k ** -0.5

    out = gdn_decode_fla(
        q, k, v, a_raw, b_raw, A_log=a_log, dt_bias=dt_bias, state_source=pool,
        indices=torch.tensor(slots, device=DEV, dtype=torch.int32),
        cu_seqlens=torch.arange(4, device=DEV, dtype=torch.int64), scale=scale,
    )
    # Legacy gate math, spelled out: scalar-per-head g, beta = sigmoid(b), GQA-expanded q/k.
    g = -a_log.exp() * torch.nn.functional.softplus(a_raw + dt_bias)   # [3, num_v]
    rep = num_v // num_k
    ref_out, ref_state = _ref_kda(
        q[0].repeat_interleave(rep, dim=1), k[0].repeat_interleave(rep, dim=1), v[0],
        g, torch.sigmoid(b_raw),
        torch.zeros(3, num_v, head_k, head_v, device=DEV), [1, 1, 1], scale,
    )
    assert _rel_err(out, ref_out) < REL_TOL
    assert _rel_err(pool[slots].transpose(-1, -2), ref_state) < REL_TOL
