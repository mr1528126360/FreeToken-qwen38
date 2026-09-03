"""GLM-5.3-Flash (``glm5_next``) vision tower -- bf16 pure-torch port of HF
``Glm5NextVisionModel`` (transformers/models/glm5_next/modeling_glm5_next.py:1733).

Consumes the PIL image processor's output directly: ``pixel_values`` is
``[total_patches, in_channels * temporal_patch_size * patch_size**2]`` with each image's
patches in spatial-merge-block-major order, and ``grid_thw`` is ``[num_images, 3]``
(temporal=1 for images). Per-image token count after the tower is
``prod(grid) // spatial_merge_size**2``.

Structure (1.13 GB of bf16 weights, 24 blocks, hidden 1024):

    Conv3d patch_embed (k=stride=(2,14,14)) -> 24 x [RMSNorm -> fused qkv attention
    (per-head q/k RMSNorm + 2D rotary) -> residual, RMSNorm -> clamped-SwiGLU MLP]
    -> post_layernorm (RMSNorm) -> Conv2d downsample (1024 -> 4096, k=stride=2)
    -> PatchMerger (proj 4096 -> GELU(LayerNorm) -> SwiGLU(intermediate 10240))

Three things differ from the Qwen3.8 tower next door, and each is load-bearing:

* No learned ``pos_embed`` table -- GLM5 places 2D rotary phases only (``init_pos_height_size``
  is in the config but no ``pos_embed`` weight ships), so the interpolation path is absent.
* Block norms are RMSNorm (weight only, ``rms_norm_eps``); ``merger.post_projection_norm``
  is the only true LayerNorm in the tower.
* Both MLPs clamp their SwiGLU inputs at ``swiglu_limit`` (10.0 here), and attention
  normalizes q/k per head *before* rotary -- skipping either changes the outputs
  materially on real images.

The tower is replicated on every TP rank (irrelevant today: glm5_next weights are
TP=1-only upstream, see weight.py). Both ranks would run the same pixels through the same
bf16 weights, so no cross-rank traffic is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP, LinearReplicated, OPList

if TYPE_CHECKING:
    from .config import Glm5NextVisionArgs

_VISION_ROPE_THETA = 10000.0  # HF Glm5NextVisionRotaryEmbedding default; not configurable


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class _VisionRMSNorm(BaseOP):
    """RMSNorm with weight only, eps from ``vision_config.rms_norm_eps``.

    Deliberately not ``F.rms_norm``: that kernel folds the weight multiply in fp32 and rounds
    once, while HF normalizes in fp32, casts back to the activation dtype and *then* multiplies
    by the (dtype) weight. The extra rounding is worth ~1 ulp per norm here, which compounds to
    cos 0.98 vs the reference over 24 blocks -- bit-matching HF costs nothing.
    """

    def __init__(self, size: int, eps: float) -> None:
        self.weight = torch.empty(size)
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self._eps)
        return self.weight * x.to(dtype)


class _VisionLayerNorm(BaseOP):
    """LayerNorm with weight + bias (used only by ``merger.post_projection_norm``, which
    HF builds as a bare ``torch.nn.LayerNorm(dim)`` -- hence the default eps of 1e-5, not the
    1e-6 the language side uses)."""

    def __init__(self, size: int, eps: float = 1e-5) -> None:
        self.weight = torch.empty(size)
        self.bias = torch.empty(size)
        self._eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], self.weight, self.bias, self._eps)


class Glm5NextVisionPatchEmbed(BaseOP):
    """Conv3d(k=stride=(T,P,P)); the processor's per-row ``(C, T, ph, pw)`` flattening means
    a plain view restores the conv input."""

    def __init__(self, vc: Glm5NextVisionArgs) -> None:
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


class Glm5NextVisionMLP(BaseOP):
    """SwiGLU with both inputs clamped at ``swiglu_limit`` (HF Glm5NextVisionMLP:1508).

    All three projections carry a bias: HF builds this MLP with ``bias=config.attention_bias``
    which is True for this checkpoint.
    """

    def __init__(self, vc: Glm5NextVisionArgs) -> None:
        self.gate_proj = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.up_proj = LinearReplicated(vc.hidden_size, vc.intermediate_size, has_bias=True)
        self.down_proj = LinearReplicated(vc.intermediate_size, vc.hidden_size, has_bias=True)
        assert vc.hidden_act == "silu", f"unexpected vision act {vc.hidden_act}"
        self._limit = vc.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj.forward(x).clamp(max=self._limit)
        up = self.up_proj.forward(x).clamp(min=-self._limit, max=self._limit)
        return self.down_proj.forward(F.silu(gate) * up)


class Glm5NextVisionAttention(BaseOP):
    """Non-causal attention over each frame's patch window. HF packs windows with
    ``cu_seqlens`` for flash-attn; the windows are disjoint, so slicing per window and
    running SDPA is the same math. Images have temporal=1 -> one window per image."""

    def __init__(self, vc: Glm5NextVisionArgs) -> None:
        self.num_heads = vc.num_heads
        self.head_dim = vc.hidden_size // vc.num_heads
        self.qkv = LinearReplicated(vc.hidden_size, vc.hidden_size * 3, has_bias=vc.attention_bias)
        self.proj = LinearReplicated(vc.hidden_size, vc.hidden_size, has_bias=vc.attention_bias)
        # Per-head norms, applied to the split heads *before* rotary (HF Glm5NextVisionAttention:1611).
        self.q_norm = _VisionRMSNorm(self.head_dim, vc.rms_norm_eps)
        self.k_norm = _VisionRMSNorm(self.head_dim, vc.rms_norm_eps)

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
        q = self.q_norm.forward(q)
        k = self.k_norm.forward(k)
        # fp32 rope like HF's apply_rotary_pos_emb_vision; cos/sin [N, head_dim]
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # [N, 1, head_dim]
        q = (q.float() * cos + _rotate_half(q.float()) * sin).to(q.dtype)
        k = (k.float() * cos + _rotate_half(k.float()) * sin).to(k.dtype)

        chunks = []
        for q_i, k_i, v_i in zip(q.split(seqlens), k.split(seqlens), v.split(seqlens)):
            o = F.scaled_dot_product_attention(
                q_i.transpose(0, 1).unsqueeze(0),
                k_i.transpose(0, 1).unsqueeze(0),
                v_i.transpose(0, 1).unsqueeze(0),
            )
            chunks.append(o.squeeze(0).transpose(0, 1))
        return self.proj.forward(torch.cat(chunks).reshape(n, -1))


class Glm5NextVisionBlock(BaseOP):
    def __init__(self, vc: Glm5NextVisionArgs) -> None:
        self.norm1 = _VisionRMSNorm(vc.hidden_size, vc.rms_norm_eps)
        self.norm2 = _VisionRMSNorm(vc.hidden_size, vc.rms_norm_eps)
        self.attn = Glm5NextVisionAttention(vc)
        self.mlp = Glm5NextVisionMLP(vc)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlens: list[int]
    ) -> torch.Tensor:
        x = x + self.attn.forward(self.norm1.forward(x), cos, sin, seqlens)
        return x + self.mlp.forward(self.norm2.forward(x))


class Glm5NextVisionDownsample(BaseOP):
    """Conv2d(hidden -> out_hidden, k=stride=spatial_merge_size): folds each 2x2 merge block
    into one vector, so ``out_hidden_size`` must equal ``hidden_size * merge**2``."""

    def __init__(self, vc: Glm5NextVisionArgs) -> None:
        self.weight = torch.empty(
            vc.out_hidden_size,
            vc.hidden_size,
            vc.spatial_merge_size,
            vc.spatial_merge_size,
        )
        self.bias = torch.empty(vc.out_hidden_size)
        self._k = vc.spatial_merge_size
        assert vc.out_hidden_size == vc.hidden_size * vc.spatial_merge_size**2, (
            f"downsample expects out_hidden_size {vc.out_hidden_size} == hidden_size "
            f"{vc.hidden_size} * merge**2 {vc.spatial_merge_size**2}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, C] -> [n_merged, C, k, k] -> conv -> [n_merged, out_hidden]
        x = x.view(-1, self._k, self._k, x.shape[-1]).permute(0, 3, 1, 2)
        return F.conv2d(x, self.weight, self.bias, stride=self._k).view(-1, self.weight.shape[0])


class Glm5NextVisionPatchMerger(BaseOP):
    """HF Glm5NextVisionPatchMerger:1517 -- ``dim`` is already the *merged* width
    (``out_hidden_size``), unlike Qwen3.8 where the merger itself does the 4x concat.
    No biases anywhere (HF builds it with ``bias=False``)."""

    def __init__(self, vc: Glm5NextVisionArgs) -> None:
        dim = vc.out_hidden_size
        self.proj = LinearReplicated(dim, dim, has_bias=False)
        self.post_projection_norm = _VisionLayerNorm(dim)
        self.gate_proj = LinearReplicated(dim, vc.projection_intermediate_size, has_bias=False)
        self.up_proj = LinearReplicated(dim, vc.projection_intermediate_size, has_bias=False)
        self.down_proj = LinearReplicated(vc.projection_intermediate_size, dim, has_bias=False)
        assert vc.hidden_act == "silu", f"unexpected vision act {vc.hidden_act}"
        self._limit = vc.swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj.forward(x)
        x = F.gelu(self.post_projection_norm.forward(x))  # act1: exact GELU, not tanh
        gate = self.gate_proj.forward(x).clamp(max=self._limit)
        up = self.up_proj.forward(x).clamp(min=-self._limit, max=self._limit)
        return self.down_proj.forward(F.silu(gate) * up)


class Glm5NextVisionModel(BaseOP):
    """Pixels -> text-space soft tokens: ``[sum(prod(grid) // merge**2), out_hidden_size]``."""

    def __init__(self, vc: Glm5NextVisionArgs) -> None:
        self.patch_embed = Glm5NextVisionPatchEmbed(vc)
        self.blocks = OPList([Glm5NextVisionBlock(vc) for _ in range(vc.depth)])
        self.post_layernorm = _VisionRMSNorm(vc.hidden_size, vc.rms_norm_eps)
        self.downsample = Glm5NextVisionDownsample(vc)
        self.merger = Glm5NextVisionPatchMerger(vc)
        self._merge = vc.spatial_merge_size
        self._head_dim = vc.hidden_size // vc.num_heads
        # Built lazily on the runtime device: the model is constructed on the meta device,
        # and _-prefixed attributes are outside the state dict (never materialized by load).
        self._inv_freq: torch.Tensor | None = None

    def _rotary_cos_sin(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from transformers.vision_utils import get_vision_position_ids

        pos_ids = get_vision_position_ids(grid_thw, self._merge)  # [N, 2] (h, w), block-major
        if self._inv_freq is None or self._inv_freq.device != pos_ids.device:
            # HF: Glm5NextVisionRotaryEmbedding(dim=head_dim // 2) -> head_dim/4 freqs,
            # then cat((emb, emb)) back up to head_dim.
            dim = self._head_dim // 2
            inv = 1.0 / (
                _VISION_ROPE_THETA
                ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos_ids.device) / dim)
            )
            # HF casts the whole tower to the model dtype, inv_freq buffer included, and
            # computes the freqs in that dtype (bf16) before cos/sin. Match it exactly --
            # taken from the weights, since the pixel buffer's dtype is not the tower's.
            self._inv_freq = inv.to(self.patch_embed.proj_weight.dtype)
        emb = (pos_ids.unsqueeze(-1) * self._inv_freq).flatten(1)  # [N, head_dim/2]
        emb = torch.cat((emb, emb), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw = grid_thw.to(device=pixel_values.device, dtype=torch.long)
        cos, sin = self._rotary_cos_sin(grid_thw)
        # Attention windows are per frame, not per image (HF get_vision_attention_seqlens).
        seqlens = [int(h * w) for t, h, w in grid_thw.tolist() for _ in range(int(t))]

        hidden = self.patch_embed.forward(pixel_values)
        for block in self.blocks.op_list:
            hidden = block.forward(hidden, cos, sin, seqlens)
        hidden = self.post_layernorm.forward(hidden)
        return self.merger.forward(self.downsample.forward(hidden))


__all__ = ["Glm5NextVisionModel"]
