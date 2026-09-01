"""qwen4_exp vision tests: image preprocessing, the ViT vs HF, weight loading, mRoPE.

The vision tower is compared against HF ``Qwen4ExpVisionModel`` on a scaled-down config
(same block structure, fewer layers/heads); the mRoPE position/cos-sin math is compared
against HF ``Qwen4ExpModel.get_rope_index`` and ``Qwen4ExpTextRotaryEmbedding``.
"""

from __future__ import annotations

import base64
import io
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .common import requires_cuda

IMG = 248056
MERGE = 2

_PREPROCESSOR_CONFIG = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}


def _png_b64(arr: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.fixture()
def model_dir(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"image_token_id": IMG}))
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(_PREPROCESSOR_CONFIG))
    return str(tmp_path)


def _images():
    rng = np.random.RandomState(0)
    return [
        (rng.rand(64, 96, 3) * 255).astype(np.uint8),   # 4x6 grid
        (rng.rand(480, 640, 3) * 255).astype(np.uint8),  # resized by smart_resize
    ]


# --------------------------------------------------------------------------------------
# preprocessing + token expansion
# --------------------------------------------------------------------------------------


def test_preprocess_token_counts_match_hf_processor(model_dir):
    from freetoken.tokenizer.images import ImageProcessor

    proc = ImageProcessor(model_dir)
    assert proc.available and proc.image_token_id == IMG
    payload = proc.process([_png_b64(img) for img in _images()])

    # Ground truth: the HF processor's own grid output.
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )

    hf = Qwen2VLImageProcessorPil.from_pretrained(model_dir)
    out = hf(images=[img for img in _images()], return_tensors="pt")
    grid = out["image_grid_thw"]
    assert payload["grid_thw"] == grid.tolist()
    expected = (grid.prod(-1) // MERGE**2).tolist()
    assert payload["token_counts"] == expected
    # pixel payload round-trips to the processor's tensor (modulo the bf16 wire format)
    pv = (
        torch.from_numpy(np.frombuffer(payload["pixel_values"], dtype=np.uint16).copy())
        .view(torch.bfloat16)
        .reshape(payload["shape"])
    )
    assert payload["shape"] == list(out["pixel_values"].shape)
    assert torch.equal(pv, out["pixel_values"].to(torch.bfloat16))


def test_expand_image_tokens():
    from freetoken.tokenizer.images import expand_image_tokens

    ids = torch.tensor([10, 11, IMG, 12, 13, IMG, 14], dtype=torch.int32)
    out = expand_image_tokens(ids, IMG, [6, 1])
    assert out.tolist() == [10, 11] + [IMG] * 6 + [12, 13, IMG, 14]
    with pytest.raises(ValueError, match="placeholders"):
        expand_image_tokens(ids, IMG, [6])


def test_extract_image_urls_skips_text_only():
    from freetoken.tokenizer.images import extract_image_urls

    assert extract_image_urls("plain prompt") == []
    assert extract_image_urls([{"role": "user", "content": "hi"}]) == []
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        }
    ]
    assert extract_image_urls(msgs) == ["data:image/png;base64,AA"]
    with pytest.raises(ValueError, match="missing its url"):
        extract_image_urls(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
        )


def test_render_content_parts_passes_images_through():
    from freetoken.server.generation import render_messages

    text_only = render_messages(
        [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    )
    assert text_only[0]["content"] == "ab"  # all-text still flattens to a string

    mixed = render_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                ],
            }
        ]
    )
    assert mixed[0]["content"] == [
        {"type": "text", "text": "what"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
    ]
    with pytest.raises(ValueError, match="Unsupported content part"):
        render_messages([{"role": "user", "content": [{"type": "audio_url", "audio_url": {}}]}])


def test_tokenize_expands_image_placeholders(model_dir):
    """End of the tokenizer-worker path: chat template + expansion == HF token count."""
    from freetoken.message import TokenizeMsg
    from freetoken.core import SamplingParams
    from freetoken.tokenizer.tokenize import TokenizeManager
    from transformers import AutoTokenizer

    import shutil
    import os

    src = os.path.expanduser("~/models/Qwen3.8-Flash-Next-NVFP4")
    if not os.path.isdir(src):
        pytest.skip("real model dir not available")
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "chat_template.jinja"):
        shutil.copy(os.path.join(src, name), os.path.join(model_dir, name))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    manager = TokenizeManager(tokenizer)
    msg = TokenizeMsg(
        uid=0,
        text=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "图里有什么？"},
                    {"type": "image_url", "image_url": {"url": _png_b64(_images()[0])}},
                ],
            }
        ],
        sampling_params=SamplingParams(),
    )
    tokens, mm = manager.tokenize_with_mm([msg])[0]
    assert mm is not None
    n_img = int((tokens == IMG).sum())
    assert n_img == mm["grid_thw"][0][1] * mm["grid_thw"][0][2] // MERGE**2
    # the placeholder span keeps its <|vision_start|>/<|vision_end|> frame
    pos = (tokens == IMG).nonzero().view(-1)
    assert tokens[pos[0] - 1].item() == 248053 and tokens[pos[-1] + 1].item() == 248054
    # text-only requests carry no payload and are byte-identical to the old path
    text_msg = TokenizeMsg(uid=1, text=[{"role": "user", "content": "hello"}],
                           sampling_params=SamplingParams())
    tokens_text, mm_text = manager.tokenize_with_mm([text_msg])[0]
    assert mm_text is None and tokens_text.numel() > 0


# --------------------------------------------------------------------------------------
# vision tower vs HF
# --------------------------------------------------------------------------------------

_HF_RENAMES = {
    "patch_embed.proj.weight": "patch_embed.proj_weight",
    "patch_embed.proj.bias": "patch_embed.proj_bias",
    "pos_embed.weight": "pos_embed",
}


def _vision_cfgs():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig

    from freetoken.models.qwen4_exp.config import Qwen4ExpVisionArgs

    kwargs = dict(
        depth=3,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        num_position_embeddings=256,
        out_hidden_size=32,
    )
    hf = Qwen4ExpVisionConfig(**kwargs)
    ft = Qwen4ExpVisionArgs(**kwargs, hidden_act="gelu_pytorch_tanh")
    return hf, ft


def test_vision_encoder_matches_hf():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpVisionModel as HFVision,
    )

    from freetoken.models.qwen4_exp.vision import Qwen4ExpVisionModel
    from freetoken.utils.torch_utils import torch_dtype

    torch.manual_seed(0)
    hf_cfg, ft_cfg = _vision_cfgs()
    hf = HFVision(hf_cfg).eval().to(torch.bfloat16)
    with torch_dtype(torch.bfloat16):
        ft = Qwen4ExpVisionModel(ft_cfg)
    ft.load_state_dict({_HF_RENAMES.get(k, k): v for k, v in hf.state_dict().items()})

    # two images, non-square grids, packed in the processor's block-major order
    grid = torch.tensor([[1, 2, 4], [1, 4, 2]])
    n = int(grid.prod(-1).sum())
    pixel_values = torch.randn(n, 3 * 2 * 16 * 16).to(torch.bfloat16)
    with torch.no_grad():
        out_hf = hf(pixel_values, grid_thw=grid).pooler_output
        out_ft = ft.forward(pixel_values, grid)
    assert out_ft.shape == out_hf.shape == (2 + 2, 32)
    cos = torch.nn.functional.cosine_similarity(out_hf.float(), out_ft.float(), dim=-1)
    assert cos.min().item() > 0.999


_HF_RENAMES_INV = {v: k for k, v in _HF_RENAMES.items()}
_REAL_MODEL = os.path.expanduser("~/models/Qwen3.8-Flash-Next-NVFP4")


@pytest.mark.skipif(not os.path.isdir(_REAL_MODEL), reason="real checkpoint not available")
def test_vision_encoder_matches_hf_on_the_real_checkpoint(monkeypatch):
    """Full-depth tower (27 layers) with the shipped bf16 weights vs the HF reference."""
    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpVisionModel as HFVision,
    )

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.models.qwen4_exp.config import Qwen4ExpVisionArgs
    from freetoken.models.qwen4_exp.vision import Qwen4ExpVisionModel
    from freetoken.models.qwen4_exp.weight import iter_visual_weights
    from freetoken.utils.torch_utils import torch_dtype

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)

    hf_cfg = AutoConfig.from_pretrained(_REAL_MODEL).vision_config
    hf = HFVision(hf_cfg).eval().to(torch.bfloat16)
    # iter_visual_weights renames to the FreeToken layout; map back for the HF model
    ft_sd = {
        k[len("model.visual."):]: v
        for k, v in iter_visual_weights(_REAL_MODEL, torch.device("cpu"))
    }
    hf.load_state_dict({_HF_RENAMES_INV.get(k, k): v for k, v in ft_sd.items()}, strict=True)

    ft_cfg = Qwen4ExpVisionArgs(
        depth=hf_cfg.depth,
        hidden_size=hf_cfg.hidden_size,
        intermediate_size=hf_cfg.intermediate_size,
        num_heads=hf_cfg.num_heads,
        patch_size=hf_cfg.patch_size,
        temporal_patch_size=hf_cfg.temporal_patch_size,
        spatial_merge_size=hf_cfg.spatial_merge_size,
        in_channels=hf_cfg.in_channels,
        num_position_embeddings=hf_cfg.num_position_embeddings,
        out_hidden_size=hf_cfg.out_hidden_size,
        hidden_act="gelu_pytorch_tanh",
    )
    with torch_dtype(torch.bfloat16):
        ft = Qwen4ExpVisionModel(ft_cfg)
    ft.load_state_dict(ft_sd)

    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )

    imgs = [Image.fromarray(img) for img in _images()]
    out = Qwen2VLImageProcessorPil.from_pretrained(_REAL_MODEL)(
        images=imgs, return_tensors="pt"
    )
    with torch.no_grad():
        out_hf = hf(out["pixel_values"], grid_thw=out["image_grid_thw"]).pooler_output.float()
        out_ft = ft.forward(out["pixel_values"], out["image_grid_thw"]).float()
    cos = torch.nn.functional.cosine_similarity(out_hf, out_ft, dim=-1)
    assert cos.min().item() > 0.999


# --------------------------------------------------------------------------------------
# weight loading
# --------------------------------------------------------------------------------------


def test_iter_visual_weights(model_dir, tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from freetoken.distributed import set_tp_info, try_get_tp_info

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    folder = tmp_path / "ckpt"
    folder.mkdir()
    raw = {
        "model.visual.patch_embed.proj.weight": torch.randn(4, 3, 2, 2, 2).to(torch.bfloat16),
        "model.visual.patch_embed.proj.bias": torch.randn(4).to(torch.bfloat16),
        "model.visual.pos_embed.weight": torch.randn(8, 4).to(torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(12, 4).to(torch.bfloat16),
        "model.language_model.embed_tokens.weight": torch.randn(8, 4).to(torch.bfloat16),
    }
    save_file(raw, str(folder / "model-bf16-00001.safetensors"))

    from freetoken.models.qwen4_exp.weight import iter_visual_weights

    monkeypatch.delenv("FREETOKEN_LOAD_VISION", raising=False)
    assert list(iter_visual_weights(str(folder), torch.device("cpu"))) == []

    monkeypatch.setenv("FREETOKEN_LOAD_VISION", "1")
    loaded = dict(iter_visual_weights(str(folder), torch.device("cpu")))
    assert set(loaded) == {
        "model.visual.patch_embed.proj_weight",
        "model.visual.patch_embed.proj_bias",
        "model.visual.pos_embed",
        "model.visual.blocks.0.attn.qkv.weight",
    }
    assert torch.equal(
        loaded["model.visual.patch_embed.proj_weight"],
        raw["model.visual.patch_embed.proj.weight"],
    )


# --------------------------------------------------------------------------------------
# mRoPE
# --------------------------------------------------------------------------------------


def _hf_rope_index(input_ids, grid_thw):
    import types as pytypes

    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel as HFModel

    fake = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=MERGE))
    )
    fake.get_vision_position_ids = pytypes.MethodType(HFModel.get_vision_position_ids, fake)
    mm_types = (input_ids == IMG).long().unsqueeze(0)
    return HFModel.get_rope_index(
        fake, input_ids.view(1, -1), mm_types, image_grid_thw=grid_thw
    )


def test_compute_prompt_mrope_matches_hf():
    from freetoken.models.qwen4_exp.mrope import compute_prompt_mrope

    # text | image 1x4x6 (6 tokens) | text | image 1x2x2 (1 token) | text
    ids = torch.tensor([1] * 5 + [IMG] * 6 + [2] * 3 + [IMG] + [3] * 4)
    grids = [(1, 4, 6), (1, 2, 2)]
    pos_hf, delta_hf = _hf_rope_index(ids, torch.tensor(grids))
    pos_ft, delta_ft = compute_prompt_mrope(
        ids, grids, image_token_id=IMG, spatial_merge_size=MERGE
    )
    assert torch.equal(pos_hf[:, 0], pos_ft)
    assert delta_ft == int(delta_hf.item())


def test_mrope_cos_sin_matches_hf():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

    from freetoken.models.qwen4_exp.mrope import _cos_sin_rows, compute_prompt_mrope

    rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
    }
    cfg = Qwen4ExpTextConfig(
        hidden_size=2560, num_attention_heads=10, head_dim=256,
        rope_parameters=rope_parameters,
    )
    rot = Qwen4ExpTextRotaryEmbedding(cfg)
    ids = torch.tensor([1] * 5 + [IMG] * 6 + [2] * 3)
    pos3d, _ = compute_prompt_mrope(ids, [(1, 4, 6)], image_token_id=IMG, spatial_merge_size=MERGE)
    # fp32 x so HF does not downcast cos/sin (the engine's rope cache is fp32 too)
    cos_hf, sin_hf = rot.forward(torch.zeros(1, dtype=torch.float32), pos3d[:, None, :])

    inv = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    table = _cos_sin_rows(pos3d, inv, (11, 11, 10))
    assert table.shape == (ids.numel(), 64)
    assert torch.equal(cos_hf[0][:, :32], table[:, :32])
    assert torch.equal(sin_hf[0][:, :32], table[:, 32:])


@requires_cuda
def test_build_batch_rope_positions_prefill_and_decode():
    """The batch-level redirect: table keys land on the right rows for mixed batches."""
    from freetoken.core import Batch, Req
    from freetoken.models.qwen4_exp.mrope import (
        _cos_sin_rows,
        build_batch_rope_positions,
        compute_prompt_mrope,
    )

    device = torch.device("cuda")
    rope_base, rotary_dim, section = 10000000.0, 64, (11, 11, 10)

    def make_req(ids, grids=None, cached=0):
        ids_t = torch.tensor(ids, dtype=torch.int32)
        req = Req(
            input_ids=ids_t,
            table_idx=0,
            cached_len=cached,
            output_len=8,
            uid=0,
            sampling_params=SimpleNamespace(max_tokens=8),
            cache_handle=None,
        )
        if grids is not None:
            req.mrope_positions, req.mrope_delta = compute_prompt_mrope(
                ids_t, grids, image_token_id=IMG, spatial_merge_size=MERGE
            )
        return req

    mm_ids = [1] * 5 + [IMG] * 6 + [2] * 3
    text_ids = [7] * 10
    mm_req = make_req(mm_ids, grids=[(1, 4, 6)])
    text_req = make_req(text_ids, cached=4)  # prefix-cached text request shares the batch

    batch = Batch(reqs=[mm_req, text_req], phase="prefill")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.cat(
        [torch.arange(r.cached_len, r.device_len, dtype=torch.int32) for r in batch.reqs]
    ).to(device)
    build_batch_rope_positions(
        batch, device,
        rope_base=rope_base, rotary_dim=rotary_dim, mrope_section=section, key_margin=3,
    )
    assert batch.mrope_cos_sin is not None
    inv = 1.0 / (rope_base ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    # the mm request's rows start at table offset 0 and cover its whole prompt
    mm_table = _cos_sin_rows(mm_req.mrope_positions.to(device), inv.to(device), section)
    n_mm = len(mm_ids)
    torch.testing.assert_close(batch.mrope_cos_sin[:n_mm], mm_table)
    # the text request's window starts at cached_len - key_margin
    base = 4 - 3
    text_pos = torch.arange(base, 10).view(1, -1).expand(3, -1)
    text_table = _cos_sin_rows(text_pos.to(device), inv.to(device), section)
    torch.testing.assert_close(batch.mrope_cos_sin[n_mm:], text_table)
    # rope keys: extend positions offset into the table
    keys = batch.rope_positions.tolist()
    assert keys[:n_mm] == list(range(n_mm))
    assert keys[n_mm:] == [n_mm + (p - base) for p in range(4, 10)]

    # decode: the mm request's delta shifts its rope positions, text stays put
    mm_req.cached_len = len(mm_ids)
    mm_req.device_len = len(mm_ids) + 1
    text_req.cached_len = 10
    text_req.device_len = 11
    batch = Batch(reqs=[mm_req, text_req], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.positions = torch.tensor([len(mm_ids), 10], dtype=torch.int32, device=device)
    build_batch_rope_positions(
        batch, device,
        rope_base=rope_base, rotary_dim=rotary_dim, mrope_section=section, key_margin=3,
    )
    assert batch.mrope_cos_sin is None
    assert batch.rope_positions.tolist() == [len(mm_ids) + mm_req.mrope_delta, 10]
