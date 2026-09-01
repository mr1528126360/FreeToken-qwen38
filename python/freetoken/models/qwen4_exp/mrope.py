"""mRoPE support for Qwen3.8-Flash-Next (images only).

HF computes per-token 3D (t, h, w) positions (``Qwen4ExpModel.get_rope_index``): text tokens
get the same running position on all three channels; each image's ``image_token_id`` span gets
a (t, h, w) grid offset by the running position, which then advances by
``max(grid_h, grid_w) // spatial_merge_size`` instead of the span length. At decode time all
three channels are equal to ``seq position + rope_delta`` where ``rope_delta =
max(prompt positions) + 1 - prompt_len`` -- so decode needs only the scalar delta.

Engine wiring (see ``Batch.rope_positions`` / ``Batch.mrope_cos_sin``): ``batch.positions``
stays the LOGICAL sequential position everywhere (the QSA backend derives ring addressing and
group boundaries from its consecutiveness), and rope lookups alone are redirected:

* prefill with an image request: ``mrope_cos_sin`` is a per-token cos/sin table (same row
  layout as the standard rope cache: ``cat(cos, sin)`` over the rotary width) covering every
  padded request's ``[cached_len - key_margin, device_len)`` window, and ``rope_positions``
  holds each extend token's row key into it. Text rows of the table equal the standard
  cache rows (all three channels equal), so mixed batches share one table.
* decode: the 3 channels coincide, so ``rope_positions = positions + delta`` with the
  standard cache is exact; no table is built (keeps CUDA-graph decode on the usual buffers).

The QSA indexer's fused norm+rope takes the same ``rope_positions`` for its query rows, and
``first + (rope_positions[row] - positions[row])`` as the rope key for a pooled key group --
the group closes on the same request as its first token, so the per-row shift carries over.
"""

from __future__ import annotations

import itertools

import torch


def compute_prompt_mrope(
    input_ids: torch.Tensor,
    image_grid_thw: list[tuple[int, int, int]],
    *,
    image_token_id: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, int]:
    """3D mRoPE positions for one whole prompt, plus the decode-time rope delta.

    ``input_ids`` is the EXPANDED prompt (each image already occupying ``prod(grid) //
    merge**2`` consecutive ``image_token_id`` slots). Returns ``(positions[3, L] int64 cpu,
    delta)``. Port of HF ``Qwen4ExpModel.get_rope_index`` reduced to text+image modalities.
    """
    ids = input_ids.tolist()
    types = [1 if t == image_token_id else 0 for t in ids]
    grids = iter(image_grid_thw)
    chunks: list[torch.Tensor] = []
    pos = 0
    for modality, group in itertools.groupby(enumerate(types), lambda x: x[1]):
        n = len(list(group))
        if modality == 0:
            chunks.append(
                torch.arange(n, dtype=torch.int64).view(1, -1).expand(3, -1) + pos
            )
            pos += n
        else:
            t, h, w = next(grids)
            if t != 1:
                raise ValueError(f"video grids (t={t}) are not supported")
            if n != (t * h * w) // spatial_merge_size**2:
                raise ValueError(
                    f"image token span ({n}) != grid tokens ({(t * h * w) // spatial_merge_size**2})"
                )
            gh, gw = h // spatial_merge_size, w // spatial_merge_size
            tt = torch.zeros(1, gh * gw, dtype=torch.int64) + pos
            hh = torch.arange(gh, dtype=torch.int64).repeat_interleave(gw).view(1, -1) + pos
            ww = torch.arange(gw, dtype=torch.int64).repeat(gh).view(1, -1) + pos
            chunks.append(torch.cat([tt, hh, ww], dim=0))
            pos += max(gh, gw)
    positions = torch.cat(chunks, dim=1)
    assert positions.shape[1] == len(ids)
    delta = int(positions.max()) + 1 - len(ids)
    return positions, delta


def _cos_sin_rows(
    positions_3d: torch.Tensor, inv_freq: torch.Tensor, mrope_section: tuple[int, int, int]
) -> torch.Tensor:
    """``[L, 2 * half]`` rows in the rope-cache layout (cat(cos, sin)) for 3D positions.

    ``positions_3d``: ``[3, L]`` on the target device. The section interleave follows HF
    ``apply_interleaved_mrope``: channels 1/2 overwrite the strides-3 slots of channel 0.
    """
    freqs = positions_3d.float().unsqueeze(-1) * inv_freq  # [3, L, half]
    out = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = mrope_section[dim] * 3
        out[:, offset:length:3] = freqs[dim][:, offset:length:3]
    return torch.cat((out.cos(), out.sin()), dim=-1)


def build_batch_rope_positions(
    batch,
    device: torch.device,
    *,
    rope_base: float,
    rotary_dim: int,
    mrope_section: tuple[int, int, int],
    key_margin: int,
) -> None:
    """Set ``batch.rope_positions`` (and ``batch.mrope_cos_sin`` on prefill) for a batch
    that contains at least one request carrying mrope data. No-op otherwise."""
    reqs = batch.padded_reqs
    has_mm = any(getattr(r, "mrope_positions", None) is not None for r in reqs)
    has_delta = any(getattr(r, "mrope_delta", 0) for r in reqs)
    if not (has_mm or has_delta):
        return
    half = rotary_dim // 2
    inv_freq = 1.0 / (
        rope_base
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
    )
    assert len(mrope_section) == 3 and sum(mrope_section) == half

    if batch.is_decode:
        # 1 token per (padded) request; all 3 channels equal -> shift into the standard cache.
        rope_positions = batch.positions.clone()
        offset = 0
        for req in reqs:
            delta = getattr(req, "mrope_delta", 0)
            if delta:
                rope_positions[offset : offset + req.extend_len] += delta
            offset += req.extend_len
        batch.rope_positions = rope_positions
        return

    # Prefill: one cos/sin table covering every request's [base, device_len) window.
    parts: list[torch.Tensor] = []
    keys: list[torch.Tensor] = []
    offset = 0
    for req in reqs:
        pos3d = getattr(req, "mrope_positions", None)
        if pos3d is not None:
            base = 0  # image requests are never prefix-cached nor chunked
            pos3d = pos3d.to(device)
            assert pos3d.shape[1] == req.device_len
        else:
            base = max(0, req.cached_len - key_margin)
            pos3d = (
                torch.arange(base, req.device_len, dtype=torch.int64, device=device)
                .view(1, -1)
                .expand(3, -1)
            )
        rows = _cos_sin_rows(pos3d, inv_freq, mrope_section)
        parts.append(rows)
        keys.append(
            torch.arange(req.cached_len, req.device_len, dtype=torch.int64, device=device)
            + (offset - base)
        )
        offset += rows.shape[0]
    batch.mrope_cos_sin = torch.cat(parts, dim=0).contiguous()
    batch.rope_positions = torch.cat(keys).to(torch.int32)


__all__ = ["build_batch_rope_positions", "compute_prompt_mrope"]
