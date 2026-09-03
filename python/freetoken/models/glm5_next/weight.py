"""Weight loading for GLM-5.3-Flash (``glm5_next``).

Supported checkpoints: NVFP4 exports of GLM-5.3-Flash in the multimodal-wrapper
layout (``model.language_model.*``) -- ModelOpt tensor kinds (LibertAIDAI) or
compressed-tensors kinds (RedHatAI), selected by ``quantization_config``. Not
supported: bf16-expert originals (zai-org), text-only key layouts.

Routed experts go to the offload cache via ``load_nvfp4_expert_sources``;
everything else loads bf16 with keys renamed ``model.language_model.X`` ->
``model.X``. ``model.visual.*`` and the trailing MTP layer are never read.

Load-time fusions (must mirror the module split orders):

* KDA ``in_proj``  = q|k|v|b|f_a|g_a projections concatenated on the output axis
* KDA ``conv1d``   = q|k|v depthwise conv weights concatenated on the channel axis

fp32-kept tensors: ``A_log`` / ``dt_bias``, the mHC ``hc_*`` tensors, the indexer
APE, and the router ``e_score_correction_bias``. Optional W8A16 fp8-at-load
follows ``ModelConfig.attn_quant`` / ``dense_quant`` / ``lm_head_quant``
(defaults and env opt-ins: see config.py); the fp8 resident modes are TP=1-only.

Tensor parallelism (TP > 1): every fused tensor is emitted pre-sharded for this
rank (``_shard_dense_weight``) and the routed experts are sharded by expert id
(``expert_shard``). The head-sharded blocks (KDA q|k|v|b + conv channels, MLA
q_b/kv_b, o/down projections, MLP gate/up) are head-major, so a per-segment chunk
stays head-aligned; KDA's low-rank f_a|g_a bottlenecks, the latents
(q_a/kv_a), the indexer, the mHC tensors, the norms and the router replicate --
the residual stream is identical on every rank because each block's output is
all-reduced.
"""

from __future__ import annotations

import json
import os
import re
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.glm_moe_dsa.weight import _ShardReader, _quant_fp8_per_row
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

from .args import Glm5NextArgs
from .config import parse_config

# Checkpoint prefix (multimodal wrapper) -> model prefix.
_CKPT = "model.language_model"
_MODEL = "model"

# MTP-layer experts (layer == num_layers under the full checkpoint) map to None
# alongside the dense prefix; the bank loader skips them.
def _layer_to_bank(layer, config):
    return (
        None
        if layer < config.first_k_dense_replace or layer >= config.num_layers
        else layer - config.first_k_dense_replace
    )


# ModelOpt export (LibertAIDAI/GLM-5.3-Flash-NVFP4): weight | weight_scale |
# weight_scale_2 (dequant-side global).
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts",
)

# llm-compressor export (RedHatAI/GLM-5.3-Flash-NVFP4): weight_packed |
# weight_scale | weight_global_scale (quant-side global -> reciprocal at ingest).
# ``input_global_scale`` (the calibrated W4A4 activation scale) deliberately does
# not match: our routed-expert paths are W4A16 and never quantize activations.
_NVFP4_CT_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=re.compile(
        r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
        r"(?P<proj>gate_proj|up_proj|down_proj)\."
        r"(?P<kind>weight_packed|weight_global_scale|weight_scale)$"
    ),
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=_layer_to_bank,
    desc="GLM-5.3 NVFP4 experts (compressed-tensors)",
    kind_map={"weight_packed": "weight", "weight_global_scale": "weight_scale_2"},
    global_reciprocal=True,
)


def _select_expert_source_spec(model_path: str) -> Nvfp4ExpertSourceSpec:
    quant = getattr(cached_load_hf_config(model_path), "quantization_config", None) or {}
    get = quant.get if isinstance(quant, dict) else (lambda k, d=None: getattr(quant, k, d))
    method = str(get("quant_method") or "").lower()
    return _NVFP4_CT_SOURCE_SPEC if method == "compressed-tensors" else _NVFP4_SOURCE_SPEC

# KDA in_proj fusion order; MUST match Glm5NextKDA._in_proj_split.
_KDA_IN_PROJ = ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")


def _expert_shard() -> tuple[int, int] | None:
    """This rank's expert-parallel shard (experts ``id % size == rank``), None at TP=1."""
    tp_info = get_tp_info()
    return (tp_info.rank, tp_info.size) if tp_info.size > 1 else None


def load_nvfp4_expert_sources(model_path: str, config, layer_sink=None):
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _select_expert_source_spec(model_path),
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
        expert_shard=_expert_shard(),
    )


def _maybe_fp8(key: str, w: torch.Tensor, fp8: bool):
    if fp8:
        q, scale = _quant_fp8_per_row(w)
        yield f"{key}.weight", q
        yield f"{key}.weight_scale", scale
    else:
        yield f"{key}.weight", w.to(torch.bfloat16)


def _iter_kda_layer(reader, layer: int, attn_fp8: bool) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    if attn_fp8:
        # fp8 resident: q|k|v (the 201 MB/layer read) as one W8A16 GEMM with
        # per-row scales; the small gate projections b|f_a|g_a stay bf16.
        qkv = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in ("q_proj", "k_proj", "v_proj")],
            dim=0,
        )
        q, scale = _quant_fp8_per_row(qkv)
        yield f"{dst}.in_proj_qkv.weight", q
        yield f"{dst}.in_proj_qkv.weight_scale", scale
        bfg = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in ("b_proj", "f_a_proj", "g_a_proj")],
            dim=0,
        )
        yield f"{dst}.in_proj_bfg.weight", bfg
    else:
        # One fused input GEMM: q|k|v|b|f_a|g_a (output-axis concat).
        fused = torch.cat(
            [reader.get(f"{src}.{p}.weight").to(torch.bfloat16) for p in _KDA_IN_PROJ], dim=0
        )
        yield f"{dst}.in_proj.weight", fused
    # One merged depthwise conv over the q|k|v stream (channel-axis concat).
    conv = torch.cat(
        [reader.get(f"{src}.{p}_conv1d.weight").to(torch.bfloat16) for p in ("q", "k", "v")],
        dim=0,
    )
    yield f"{dst}.conv1d.weight", conv
    for p in ("f_b_proj", "g_b_proj"):
        yield f"{dst}.{p}.weight", reader.get(f"{src}.{p}.weight").to(torch.bfloat16)
    yield from _maybe_fp8(f"{dst}.o_proj", reader.get(f"{src}.o_proj.weight"), attn_fp8)
    # Gate params stay fp32 (the recurrent kernels read them as fp32).
    yield f"{dst}.A_log", reader.get(f"{src}.A_log").to(torch.float32)
    yield f"{dst}.dt_bias", reader.get(f"{src}.dt_bias").to(torch.float32)
    yield f"{dst}.o_norm.weight", reader.get(f"{src}.o_norm.weight").to(torch.bfloat16)


def _iter_dsa_layer(reader, layer: int, attn_fp8: bool) -> Iterator[tuple[str, torch.Tensor]]:
    src = f"{_CKPT}.layers.{layer}.self_attn"
    dst = f"{_MODEL}.layers.{layer}.self_attn"
    fp8_projs = ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj") if attn_fp8 else ()
    for proj in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"):
        w = reader.get(f"{src}.{proj}.weight")
        yield from _maybe_fp8(f"{dst}.{proj}", w, proj in fp8_projs)
    for norm in ("q_a_layernorm", "kv_a_layernorm"):
        yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(torch.bfloat16)
    # kpool indexer (every DSA layer owns one). Kept bf16; the APE is fp32.
    for proj in ("wq_b", "wk", "weights_proj"):
        yield f"{dst}.indexer.{proj}.weight", reader.get(
            f"{src}.indexer.{proj}.weight"
        ).to(torch.bfloat16)
    for part, dtype in (
        ("k_norm.weight", torch.bfloat16),
        ("k_norm.bias", torch.bfloat16),
        ("index_kpool_compress_gate", torch.bfloat16),
        ("index_kpool_compress_ape", torch.float32),
    ):
        yield f"{dst}.indexer.{part}", reader.get(f"{src}.indexer.{part}").to(dtype)


# ======================================================================================
# TP sharding of the dense (non-expert) weights
# ======================================================================================
#
# Applied to the FUSED tensors (KDA in_proj / conv1d), i.e. after the load-time
# concatenation, so the emitted local layout matches the modules' sharded buffers
# exactly. Everything not listed here (the MLA latents q_a/kv_a, the indexer, the
# mHC hc_* tensors, norms, the router gate and its bias, KDA o_norm) is replicated:
# the residual stream is identical on every rank because each block's output is
# all-reduced (o_proj / down_proj / the offload MoE's _maybe_all_reduce).


def _head_rows(t: torch.Tensor, offset: int, width: int, rank: int, size: int) -> torch.Tensor:
    """This rank's row range of a head-major block starting at ``offset`` (width % size == 0)."""
    local = width // size
    return t[offset + rank * local : offset + (rank + 1) * local]


def _shard_dense_weight(
    name: str, tensor: torch.Tensor, *, rank: int, size: int, args: Glm5NextArgs
) -> torch.Tensor:
    """TP shard of one fused dense weight; identity at size 1 (never called then)."""
    h, d = args.linear_num_heads, args.linear_head_dim
    p = h * d  # one KDA head-major projection block (q/k/v)
    if name.endswith(".self_attn.in_proj.weight"):
        # [q|k|v|b | f_a|g_a] rows: q|k|v (p each) and b (h, the per-head beta)
        # shard by head; f_a|g_a (d each) are the low-rank gate bottlenecks --
        # f_b/g_b consume the full d-wide vector -- and replicate. Local layout
        # [q_r|k_r|v_r|b_r|f_a|g_a] matches kda.py _KDAFusedInProj.
        return torch.cat(
            [
                _head_rows(tensor, 0, p, rank, size),
                _head_rows(tensor, p, p, rank, size),
                _head_rows(tensor, 2 * p, p, rank, size),
                _head_rows(tensor, 3 * p, h, rank, size),
                tensor[3 * p + h :],
            ],
            dim=0,
        )
    if name.endswith(".self_attn.conv1d.weight"):
        # [q|k|v] depthwise channels, same head-major split as the in_proj q|k|v rows.
        return torch.cat(
            [
                _head_rows(tensor, 0, p, rank, size),
                _head_rows(tensor, p, p, rank, size),
                _head_rows(tensor, 2 * p, p, rank, size),
            ],
            dim=0,
        )
    if name.endswith(
        (
            # Head-major output rows (column-parallel): KDA low-rank up-projections
            # and per-head gate params; MLA q_b / kv_b.
            ".self_attn.f_b_proj.weight",
            ".self_attn.g_b_proj.weight",
            ".self_attn.A_log",
            ".self_attn.dt_bias",
            ".self_attn.q_b_proj.weight",
            ".self_attn.kv_b_proj.weight",
            # MLP / shared-expert gate|up (column-parallel over intermediate).
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.shared_experts.gate_proj.weight",
            ".mlp.shared_experts.up_proj.weight",
        )
    ):
        return tensor.chunk(size, dim=0)[rank]
    if name.endswith(
        (
            # Row-parallel inputs (the block-exit all-reduce reassembles the sum).
            ".self_attn.o_proj.weight",
            ".mlp.down_proj.weight",
            ".mlp.shared_experts.down_proj.weight",
        )
    ):
        return tensor.chunk(size, dim=1)[rank]
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        per_rank = div_ceil(tensor.shape[0], size)
        return tensor[rank * per_rank : min((rank + 1) * per_rank, tensor.shape[0])]
    return tensor


_VISUAL_RENAMES = {
    "model.visual.patch_embed.proj.weight": "model.visual.patch_embed.proj_weight",
    "model.visual.patch_embed.proj.bias": "model.visual.patch_embed.proj_bias",
    # Glm5NextVisionDownsample holds the Conv2d as plain .weight/.bias attributes.
}


def iter_visual_weights(
    model_path: str, device: torch.device
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the 24-block vision tower's weights (347 bf16 tensors, 1.13 GB) with FreeToken
    state-dict names; every one is replicated. No-op unless ``FREETOKEN_LOAD_VISION=1`` --
    text-only serving never touches them, same contract as qwen4_exp."""
    from freetoken.models.config import vision_load_enabled

    if not vision_load_enabled():
        return
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading vision weights",
        disable=not get_tp_info().is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                if not raw_name.startswith("model.visual."):
                    continue
                yield _VISUAL_RENAMES.get(raw_name, raw_name), f.get_tensor(raw_name)


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    assert not include_moe_experts, (
        "GLM-5.3 stores routed experts as NVFP4 and only supports the offload backend; "
        "experts are loaded into the offload cache via load_nvfp4_expert_sources()."
    )
    assert include_non_moe
    config = parse_config(cached_load_hf_config(model_path))
    args: Glm5NextArgs = config.glm5_args
    folder = download_hf_weight(model_path)
    with open(os.path.join(folder, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    reader = _ShardReader(folder, weight_map, device)
    tp_info = get_tp_info()
    primary = tp_info.is_primary()
    attn_fp8 = config.attn_quant == "fp8_pertensor"
    mlp_fp8 = config.dense_quant == "fp8_pertensor"
    head_fp8 = config.lm_head_quant == "fp8_pertensor"
    if tp_info.size > 1 and (attn_fp8 or mlp_fp8 or head_fp8):
        raise NotImplementedError(
            "fp8 resident quant (FREETOKEN_GLM5_*_FP8) is TP=1-only; serve bf16 under TP"
        )

    def shard(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tp_info.size == 1:
            return tensor
        return _shard_dense_weight(name, tensor, rank=tp_info.rank, size=tp_info.size, args=args)

    if primary:
        from freetoken.utils import init_logger

        init_logger(__name__).info(
            f"GLM-5.3 resident quant: attn={config.attn_quant} dense={config.dense_quant} "
            f"lm_head={config.lm_head_quant} (FREETOKEN_GLM5_ATTN_FP8/FREETOKEN_GLM5_MLP_FP8; "
            "an FTW conversion records these choices implicitly -- serve with the same flags)"
        )
    try:
        for layer in tqdm(
            range(config.num_layers),
            desc="Loading GLM-5.3 dense weights",
            disable=not primary,
        ):
            src = f"{_CKPT}.layers.{layer}"
            dst = f"{_MODEL}.layers.{layer}"
            gen = (
                _iter_kda_layer(reader, layer, attn_fp8)
                if args.is_kda_layer(layer)
                else _iter_dsa_layer(reader, layer, attn_fp8)
            )
            for name, tensor in gen:
                yield name, shard(name, tensor)

            # mHC mixing tensors, fp32 on every layer (replicated).
            for hc in ("hc_attn_fn", "hc_attn_base", "hc_attn_scale",
                       "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale"):
                yield f"{dst}.{hc}", reader.get(f"{src}.{hc}").to(torch.float32)

            for norm in ("input_layernorm", "post_attention_layernorm"):
                yield f"{dst}.{norm}.weight", reader.get(f"{src}.{norm}.weight").to(
                    torch.bfloat16
                )

            if layer < config.first_k_dense_replace:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    for name, tensor in _maybe_fp8(
                        f"{dst}.mlp.{proj}", reader.get(f"{src}.mlp.{proj}.weight"), mlp_fp8
                    ):
                        yield name, shard(name, tensor)
            else:
                yield f"{dst}.mlp.gate.weight", reader.get(f"{src}.mlp.gate.weight").to(
                    torch.bfloat16
                )
                yield (
                    f"{dst}.mlp.e_score_correction_bias",
                    # fp32 like HF's router math (the module declares fp32; a bf16
                    # cast would perturb top-8 selection on fp32-bias checkpoints).
                    reader.get(f"{src}.mlp.gate.e_score_correction_bias").to(torch.float32),
                )
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    for name, tensor in _maybe_fp8(
                        f"{dst}.mlp.shared_experts.{proj}",
                        reader.get(f"{src}.mlp.shared_experts.{proj}.weight"),
                        mlp_fp8,
                    ):
                        yield name, shard(name, tensor)

        yield f"{_MODEL}.embed_tokens.weight", shard(
            f"{_MODEL}.embed_tokens.weight",
            reader.get(f"{_CKPT}.embed_tokens.weight").to(torch.bfloat16),
        )
        yield f"{_MODEL}.norm.weight", reader.get(f"{_CKPT}.norm.weight").to(torch.bfloat16)
        head = reader.get("lm_head.weight")
        if head_fp8 and not config.tie_word_embeddings:
            q, scale = _quant_fp8_per_row(head)
            yield "lm_head.weight", q
            yield "lm_head.weight_scale", scale
        else:
            yield "lm_head.weight", shard("lm_head.weight", head.to(torch.bfloat16))
    finally:
        reader.close()


__all__ = ["iter_weights", "load_nvfp4_expert_sources"]
