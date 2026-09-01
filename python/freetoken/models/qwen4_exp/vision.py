"""Qwen3.8-Flash-Next vision tower (Qwen3-VL-style ViT), bf16 pure-torch port of HF
``Qwen4ExpVisionModel`` (transformers/models/qwen4_exp/modeling_qwen4_exp.py).

Consumes the PIL image processor's output directly: ``pixel_values`` is
``[total_patches, in_channels * temporal_patch_size * patch_size**2]`` with each image's
patches in spatial-merge-block-major order, and ``grid_thw`` is ``[num_images, 3]``
(temporal=1 for images). Per-image token count after the merger is
``prod(grid) // spatial_merge_size**2``.

The tower is replicated on every TP rank: both ranks run the same pixels through the same
bf16 weights, so the scattered soft-token embeddings agree without any cross-rank traffic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, OPList

if TYPE_CHECKING:
    from .config import Qwen4ExpVisionArgs

_VISION_ROPE_THETA = 10000.0  # HF Qwen4ExpVisionRotaryEmbedding default; not configurable


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _VisionLayerNorm(BaseOP):
    """Plain LayerNorm with weight + bias (the BaseOP layer zoo only has RMSNorm variants)."""

    def __init__(self, size: int, eps: float = 1e-6) -> None:
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self._eps)


class Qwen4ExpVisionPatchEmbed(BaseOP):
    """Conv3d patch embedding; the patch dim flattening matches the processor's
    ``(C, T, ph, pw)`` per-row layout, so a plain view restores the conv input."""

    def __init__(self, vc: Qwen4ExpVisionArgs) -> None:
        self.proj_weight = torch.empty(
            vc.hidden_size, vc.in_channels, vc.temporal_patch_size, vc.patch_size, vc.patch_size
        )
        self.proj_bias = torch.empty(vc.hidden_size)
        self._shape = (vc.in_channels, vc.temporal_patch_size, vc.patch_size, vc.patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = pixel_values.view(-1, *self._shape)
        return F.conv3d(
            x.to(self.proj_weight.dtype), self.proj_weight, self.proj_bias, stride=self._shape[1:]
        ).view(-1, self.proj_weight.shape[0])


class Qwen4ExpVisionMLP(BaseOP):
    def __init__(self, vc: Qwen4ExpVisionArgs) -> None:
        self.linear_fc1 = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.linear_fc2 = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=True)
        assert vc.hidden_act == "gelu_pytorch_tanh", f"unexpected vision act {vc.hidden_act}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2.forward(
            F.gelu(self.linear_fc1.forward(x), approximate="tanh")
        )


class Qwen4ExpVisionAttention(BaseOP):
    """Full (non-causal) attention over one image's patches; images are attended
    independently via per-image slices (HF's flash path uses cu_seqlens, identical math)."""

    def __init__(self, vc: Qwen4ExpVisionArgs) -> None:
        self.num_heads = vc.num_heads
        self.head_dim = vc.hidden_size // vc.num_heads
        self.qkv = LinearReplicated(vc.hidden_size, vc.hidden_size * 3, has_bias=True)
        self.proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=True)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlens: list[int]
    ) -> torch.Tensor:
        n = x.shape[0]
        q, k, v = (
            self.qkv.forward(x)
            .reshape(n, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        # fp32 rope like HF's apply_rotary_pos_emb_vision; cos/sin [N, head_dim]
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # [N, 1, head_dim]
        q = (q.float() * cos + _rotate_half(q.float()) * sin).to(q.dtype)
        k = (k.float() * cos + _rotate_half(k.float()) * sin).to(k.dtype)

        chunks = []
        for q_i, k_i, v_i in zip(
            q.split(seqlens), k.split(seqlens), v.split(seqlens)
        ):
            o = F.scaled_dot_product_attention(
                q_i.transpose(0, 1).unsqueeze(0),
                k_i.transpose(0, 1).unsqueeze(0),
                v_i.transpose(0, 1).unsqueeze(0),
            )
            chunks.append(o.squeeze(0).transpose(0, 1))
        return self.proj.forward(torch.cat(chunks).reshape(n, -1))


class Qwen4ExpVisionBlock(BaseOP):
    def __init__(self, vc: Qwen4ExpVisionArgs) -> None:
        self.norm1 = _VisionLayerNorm(vc.hidden_size)
        self.norm2 = _VisionLayerNorm(vc.hidden_size)
        self.attn = Qwen4ExpVisionAttention(vc)
        self.mlp = Qwen4ExpVisionMLP(vc)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlens: list[int]
    ) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin, seqlens)
        return x + self.mlp.forward(self.norm2.forward(x))


class Qwen4ExpVisionPatchMerger(BaseOP):
    def __init__(self, vc: Qwen4ExpVisionArgs) -> None:
        merged = vc.hidden_size * vc.spatial_merge_size**2
        self.norm = _VisionLayerNorm(vc.hidden_size)
        self.linear_fc1 = LinearReplicated(merged, merged, has_bias=True)
        self.linear_fc2 = LinearReplicated(merged, vc.out_hidden_size, has_bias=True)
        self._merged = merged

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use_postshuffle_norm=False: norm per patch, THEN group the merge block's patches
        # (the processor's block-major patch order makes each block contiguous).
        x = self.norm.forward(x).view(-1, self._merged)
        return self.linear_fc2.forward(F.gelu(self.linear_fc1.forward(x)))


class Qwen4ExpVisionModel(BaseOP):
    """Pixels -> text-space soft tokens: ``[sum(prod(grid) // merge**2), out_hidden_size]``."""

    def __init__(self, vc: Qwen4ExpVisionArgs) -> None:
        self.patch_embed = Qwen4ExpVisionPatchEmbed(vc)
        self.pos_embed = torch.empty(vc.num_position_embeddings, vc.hidden_size)
        self.blocks = OPList([Qwen4ExpVisionBlock(vc) for _ in range(vc.depth)])
        self.merger = Qwen4ExpVisionPatchMerger(vc)
        self._merge = vc.spatial_merge_size
        self._num_grid_per_side = int(vc.num_position_embeddings**0.5)
        # Built lazily on the runtime device: the model is constructed on the meta device,
        # and _-prefixed attributes are outside the state dict (never materialized by load).
        self._head_dim = vc.hidden_size // vc.num_heads
        self._inv_freq: torch.Tensor | None = None

    def _pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        from transformers.vision_utils import get_vision_interpolation_indices_and_weights

        indices, weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self._num_grid_per_side,
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=self._merge,
        )
        return (self.pos_embed[indices] * weights[:, :, None]).sum(1)

    def _rotary_cos_sin(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers.vision_utils import get_vision_position_ids

        pos_ids = get_vision_position_ids(grid_thw, self._merge)  # [N, 2] (h, w), block-major
        if self._inv_freq is None or self._inv_freq.device != pos_ids.device:
            half = self._head_dim // 2
            inv = 1.0 / (
                _VISION_ROPE_THETA
                ** (torch.arange(0, half, 2, dtype=torch.float32, device=pos_ids.device) / half)
            )
            # HF casts the whole tower to the model dtype, inv_freq buffer included, and
            # computes the freqs in that dtype (bf16) before cos/sin. Match it exactly.
            self._inv_freq = inv.to(self.pos_embed.dtype)
        emb = (pos_ids.unsqueeze(-1) * self._inv_freq).flatten(1)  # [N, head_dim/2]
        emb = torch.cat((emb, emb), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw = grid_thw.to(device=pixel_values.device, dtype=torch.long)
        pos_embeds = self._pos_embed_interpolate(grid_thw)
        cos, sin = self._rotary_cos_sin(grid_thw)
        seqlens = [int(t * h * w) for t, h, w in grid_thw.tolist()]

        hidden = self.patch_embed.forward(pixel_values) + pos_embeds.to(
            self.patch_embed.proj_weight.dtype
        )
        for block in self.blocks.op_list:
            hidden = block.forward(hidden, cos, sin, seqlens)
        return self.merger.forward(hidden)


__all__ = ["Qwen4ExpVisionModel"]
