"""glm5_next vision tower: numerical alignment against the HF reference.

Two levels:

* ``test_vision_encoder_matches_hf`` -- a scaled-down tower with random weights, so the
  structure (norm placement, rotary, windowed attention, clamped SwiGLU, the Conv2d
  downsample + PatchMerger order) is checked without 1.13 GB of checkpoint reads.
* ``test_vision_encoder_matches_hf_on_the_real_checkpoint`` -- the shipped 24-block tower
  with the real bf16 weights, which is what actually validates the ``model.visual.*``
  name mapping in weight.py.
"""

from __future__ import annotations

import os

import pytest
import torch

_REAL_MODEL = os.path.expanduser("~/GLM-5.3-Flash-NVFP4")

# FreeToken holds the patch Conv3d as plain .proj_weight/.proj_bias tensors (nn.Conv3d would
# otherwise register a module the state-dict names differ from); everything else matches HF.
_FT_RENAMES = {
    "patch_embed.proj.weight": "patch_embed.proj_weight",
    "patch_embed.proj.bias": "patch_embed.proj_bias",
}
_FT_RENAMES_INV = {v: k for k, v in _FT_RENAMES.items()}


def _vision_cfgs(**over):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig

    from freetoken.models.glm5_next.config import Glm5NextVisionArgs

    kwargs = dict(
        depth=3,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        out_hidden_size=256,  # hidden_size * merge**2
        projection_intermediate_size=512,
        swiglu_limit=10.0,
        rms_norm_eps=1e-5,
        attention_bias=True,
    )
    kwargs.update(over)
    hf = Glm5NextVisionConfig(**kwargs, hidden_act="silu")
    ft = Glm5NextVisionArgs(
        hidden_act="silu",
        **{k: v for k, v in kwargs.items() if k != "rms_norm_eps"},
        rms_norm_eps=kwargs["rms_norm_eps"],
    )
    return hf, ft


def _inputs(seed: int = 0):
    """Two images with non-square grids, in the processor's block-major patch order."""
    grid = torch.tensor([[1, 2, 4], [1, 4, 2]])
    n = int(grid.prod(-1).sum())
    gen = torch.Generator().manual_seed(seed)
    pixels = torch.randn(n, 3 * 2 * 14 * 14, generator=gen).to(torch.bfloat16)
    return pixels, grid


def _compare(hf, ft, pixels, grid, expect_rows: int, width: int, exact: bool) -> None:
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    assert isinstance(hf, Glm5NextVisionModel)
    with torch.no_grad():
        out_hf = hf(pixels, grid_thw=grid).pooler_output.float()
        out_ft = ft.forward(pixels, grid).float()
    assert out_ft.shape == out_hf.shape == (expect_rows, width)
    cos = torch.nn.functional.cosine_similarity(out_hf, out_ft, dim=-1)
    max_diff = (out_hf - out_ft).abs().max().item()
    if exact:
        # CPU + bf16: the tower reproduces HF bit for bit (see the RMSNorm note in vision.py).
        assert max_diff == 0.0, f"vision outputs differ: max abs diff {max_diff}"
    assert cos.min().item() > 0.999, f"vision outputs diverge: min cos {cos.min().item():.6f}"


def test_vision_encoder_matches_hf():
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel as HFVision

    from freetoken.models.glm5_next.vision import Glm5NextVisionModel
    from freetoken.utils.torch_utils import torch_dtype

    torch.manual_seed(0)
    hf_cfg, ft_cfg = _vision_cfgs()
    hf = HFVision(hf_cfg).eval().to(torch.bfloat16)
    with torch_dtype(torch.bfloat16):
        ft = Glm5NextVisionModel(ft_cfg)
    # BaseOP.load_state_dict is strict and consumes the dict, so hand it a copy
    ft.load_state_dict(
        {_FT_RENAMES.get(k, k): v for k, v in hf.state_dict().items()}
    )

    pixels, grid = _inputs()
    # (1*2*4 + 1*4*2) = 16 patches / merge**2 4 -> 4 soft tokens, merger emits out_hidden_size
    _compare(hf, ft, pixels, grid, expect_rows=4, width=hf_cfg.out_hidden_size, exact=True)


def test_rotary_position_ids_are_block_major():
    """No learned pos_embed in this tower: the 2D rotary phases must land on the same
    (h, w) grid HF builds for block-major patches."""
    from transformers.vision_utils import get_vision_position_ids

    from freetoken.models.glm5_next.vision import Glm5NextVisionModel
    from freetoken.utils.torch_utils import torch_dtype

    _, ft_cfg = _vision_cfgs()
    with torch_dtype(torch.bfloat16):
        ft = Glm5NextVisionModel(ft_cfg)
    grid = torch.tensor([[1, 2, 4], [1, 4, 2]])
    cos, sin = ft._rotary_cos_sin(grid)
    assert cos.shape == (16, ft._head_dim)
    ref = get_vision_position_ids(grid, ft_cfg.spatial_merge_size)
    head_dim = ft._head_dim
    dim = head_dim // 2
    inv = (
        1.0
        / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ).to(torch.bfloat16)
    # cat((emb, emb), -1) == repeat(1, 2) on the flattened [N, head_dim/2] phases
    emb = (ref.unsqueeze(-1) * inv).flatten(1).repeat(1, 2)
    assert torch.equal(cos, emb.cos()) and torch.equal(sin, emb.sin())


@pytest.mark.skipif(not os.path.isdir(_REAL_MODEL), reason="real checkpoint not available")
def test_vision_encoder_matches_hf_on_the_real_checkpoint(monkeypatch):
    """Full-depth tower (24 blocks) with the shipped bf16 weights vs the HF reference."""
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel as HFVision

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.glm5_next.config import Glm5NextVisionArgs
    from freetoken.models.glm5_next.vision import Glm5NextVisionModel
    from freetoken.models.glm5_next.weight import iter_visual_weights
    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    # Set it here rather than skipping: this is the assertion worth always running.
    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")

    vc = AutoConfig.from_pretrained(_REAL_MODEL).vision_config
    hf = HFVision(vc).eval().to(torch.bfloat16)
    ft_sd = {
        k[len("model.visual."):]: v
        for k, v in iter_visual_weights(_REAL_MODEL, torch.device("cpu"))
    }
    hf.load_state_dict(
        {_FT_RENAMES_INV.get(k, k): v for k, v in ft_sd.items()}, strict=True
    )

    ft_cfg = Glm5NextVisionArgs(
        depth=vc.depth,
        hidden_size=vc.hidden_size,
        intermediate_size=vc.intermediate_size,
        num_heads=vc.num_heads,
        patch_size=vc.patch_size,
        temporal_patch_size=vc.temporal_patch_size,
        spatial_merge_size=vc.spatial_merge_size,
        in_channels=vc.in_channels,
        out_hidden_size=vc.out_hidden_size,
        projection_intermediate_size=vc.projection_intermediate_size,
        hidden_act=vc.hidden_act,
        swiglu_limit=vc.swiglu_limit,
        rms_norm_eps=vc.rms_norm_eps,
        attention_bias=vc.attention_bias,
    )
    with torch_dtype(torch.bfloat16):
        ft = Glm5NextVisionModel(ft_cfg)
    # iter_visual_weights yields fresh tensors; the HF model has already copied what it needs
    # out of the same set, so load the HF side first (above) and the FT side into a copy.
    ft.load_state_dict(dict(ft_sd))

    import numpy as np
    from PIL import Image

    from transformers.models.glm5_next.image_processing_pil_glm5_next import (
        Glm5NextImageProcessorPil,
    )

    rng = np.random.RandomState(0)
    imgs = [Image.fromarray((rng.rand(224, 336, 3) * 255).astype(np.uint8))]
    out = Glm5NextImageProcessorPil.from_pretrained(_REAL_MODEL)(
        images=imgs, return_tensors="pt"
    )
    grid = out["image_grid_thw"]
    n_rows = int(grid.prod(-1).sum() // vc.spatial_merge_size**2)
    _compare(
        hf, ft, out["pixel_values"], grid, expect_rows=n_rows, width=vc.out_hidden_size,
        exact=True,
    )


def test_config_gates_the_tower_behind_load_vision(monkeypatch):
    """vision_config (hence is_multimodal, hence the whole image path) exists only when
    FREETOKEN_LOAD_VISION=1; text-only deployments keep paying nothing."""
    import json

    from freetoken.utils.hf import RawConfigShim

    if not os.path.isdir(_REAL_MODEL):
        pytest.skip("real checkpoint not available")
    with open(os.path.join(_REAL_MODEL, "config.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    from freetoken.models.glm5_next.config import parse_config

    monkeypatch.delenv("FREETOKEN_LOAD_VISION", raising=False)
    assert parse_config(RawConfigShim(raw)).vision_config is None
    assert not parse_config(RawConfigShim(raw)).is_multimodal
    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    cfg = parse_config(RawConfigShim(raw))
    assert cfg.vision_config is not None and cfg.image_token_id == 154854
    assert cfg.is_multimodal


def test_image_processor_resolves_glm5():
    """The checkpoint ships processor_config.json (no preprocessor_config.json); the
    per-model_type lookup must land on the GLM5 PIL processor."""
    from freetoken.tokenizer.images import ImageProcessor

    if not os.path.isdir(_REAL_MODEL):
        pytest.skip("real checkpoint not available")
    proc = ImageProcessor(_REAL_MODEL)
    assert proc.image_token_id == 154854
    loaded = proc._load()
    assert type(loaded).__name__ == "Glm5NextImageProcessorPil"
    assert (loaded.patch_size, loaded.temporal_patch_size, loaded.merge_size) == (14, 2, 2)


def test_visual_weight_iteration_filters_and_renames(tmp_path, monkeypatch):
    """iter_visual_weights yields only model.visual.*, with the Conv3d pair renamed to the
    FreeToken flat-tensor attributes."""
    import json

    from safetensors.torch import save_file

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.glm5_next.weight import iter_visual_weights

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    tensors = {
        "model.visual.patch_embed.proj.weight": torch.zeros(4, 3, 2, 14, 14),
        "model.visual.patch_embed.proj.bias": torch.zeros(4),
        "model.visual.blocks.0.attn.qkv.bias": torch.zeros(12),
        "model.language_model.layers.0.input_layernorm.weight": torch.zeros(4),
        "lm_head.weight": torch.zeros(4, 4),
    }
    save_file(tensors, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors},
            }
        ),
        encoding="utf-8",
    )
    got = dict(iter_visual_weights(str(tmp_path), torch.device("cpu")))
    assert set(got) == {
        "model.visual.patch_embed.proj_weight",
        "model.visual.patch_embed.proj_bias",
        "model.visual.blocks.0.attn.qkv.bias",
    }


def test_language_model_scatters_image_embeddings():
    """``_merge_multimodal`` replaces exactly the image_token_id rows, in order, and is a
    no-op when the batch carries no vision features."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from freetoken.models.glm5_next.model import Glm5NextModel

    self = SimpleNamespace(_image_token_id=7)
    merge = Glm5NextModel._merge_multimodal
    ids = torch.tensor([1, 7, 7, 2, 7, 3])
    base = torch.arange(6 * 4, dtype=torch.bfloat16).reshape(6, 4)
    feats = torch.full((3, 4), 9.0, dtype=torch.bfloat16)

    ctx = SimpleNamespace(batch=SimpleNamespace(mm_embeds=None))
    with patch("freetoken.models.glm5_next.model.get_global_ctx", return_value=ctx):
        assert torch.equal(merge(self, ids, base), base)
    ctx = SimpleNamespace(batch=SimpleNamespace(mm_embeds=feats))
    with patch("freetoken.models.glm5_next.model.get_global_ctx", return_value=ctx):
        merged = merge(self, ids, base)
    assert torch.equal(merged[[0, 3, 5]], base[[0, 3, 5]])
    assert torch.equal(merged[[1, 2, 4]], feats)


def test_multimodal_slot_count_mismatch_is_loud():
    """A truncated prompt (``--max-request-token-len`` cut the placeholders) or a chunked
    prefill would otherwise scatter silently into the wrong rows."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from freetoken.models.glm5_next.model import Glm5NextModel

    self = SimpleNamespace(_image_token_id=7)
    ids = torch.tensor([1, 7, 7, 2])
    base = torch.zeros(4, 4, dtype=torch.bfloat16)
    feats = torch.ones(5, 4, dtype=torch.bfloat16)
    ctx = SimpleNamespace(batch=SimpleNamespace(mm_embeds=feats))
    with patch("freetoken.models.glm5_next.model.get_global_ctx", return_value=ctx):
        with pytest.raises(AssertionError, match="image-token slots"):
            Glm5NextModel._merge_multimodal(self, ids, base)
