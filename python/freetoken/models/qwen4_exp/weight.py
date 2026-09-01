"""Qwen3.8-Flash-Next (RadixArk NVFP4) checkpoint reader.

Four separate paths, because the checkpoint's weight classes live in different places:

* :func:`iter_weights` -- every dense (non-expert) tensor, with the ``model.language_model.`` prefix stripped and fused where the model expects one buffer. See ``_FUSIONS``.
* :func:`load_ple_table` -- the 47.7 GiB FP8 n-gram table, 128 checkpoint shards concatenated into one pinned :class:`HostBank`.
* :func:`load_nvfp4_expert_sources` -- the routed NVFP4 experts, into the offload cache's source banks.
* :func:`iter_visual_weights` -- the ``model.visual.*`` ViT weights (bf16, replicated), only when vision loading is enabled (``FREETOKEN_LOAD_VISION=1``).

Dropped: ``mtp.*`` (speculative head, including its stacked ``mtp.layers.0.mlp.experts.*``). ``model.visual.*`` is dropped from the dense path and loaded by :func:`iter_visual_weights` instead.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from typing import Iterator

import safetensors
import torch
from freetoken.distributed import get_tp_info
from freetoken.models.loader import drop_page_cache, iter_weight_files
from freetoken.models.nvfp4_banks import (
    Nvfp4ExpertSourceSpec,
    load_nvfp4_expert_source_banks,
)
from freetoken.moe.host_banks import HostBank, read_range_into
from freetoken.utils import cached_load_hf_config, div_ceil, download_hf_weight
from freetoken.utils.progress import byte_bar
from tqdm import tqdm

# Routed NVFP4 experts (nvidia modelopt layout): per-expert, un-fused. Matched against the RAW
# weight_map key in nvfp4_banks. The ``model.language_model.`` anchor excludes the MTP head's
# stacked ``mtp.layers.N.mlp.experts.*`` tensors.
_EXPERT_KEY_RE = re.compile(
    r"^model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_RE = re.compile(r"\.mlp\.experts\.\d+\.")
_NVFP4_SOURCE_SPEC = Nvfp4ExpertSourceSpec(
    key_pattern=_EXPERT_KEY_RE,
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,  # every layer is MoE
    desc="Qwen3.8-Flash-Next NVFP4 experts",
)
# Per-tensor modelopt quant scales; consumed with their ``.weight`` (experts) or unused.
_SCALE_SUFFIXES = (".weight_scale", ".weight_scale_2", ".input_scale")

# The n-gram table itself: too big for the dense state dict, loaded by load_ple_table.
_PLE_TABLE_INFIX = ".ple.ple_embedding.ngram_embedding."
_PLE_SHARD_RE = re.compile(
    r"\.ple\.ple_embedding\.ngram_embedding\.shard_(?P<shard>\d+)\.weight$"
)
_PLE_SCALE_SUFFIX = ".ple.ple_embedding.ngram_embedding.weight_scale"

# Zero-centered Qwen4ExpTextRMSNorm weights, loaded RAW: GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm
# and the vendored grouped_gemma_rmsnorm all apply (1+w) at runtime in fp32, so folding the +1 into
# the bf16 weight here would double-apply it and round away small |w|. The GDN gated norm
# (linear_attn.norm) is a plain weight*x norm and is not in this set.
_ZERO_CENTERED_NORM_SUFFIXES = (
    ".hc_norm.weight",
    ".ple.norm_key.weight",
    ".ple.norm_query.weight",
    ".ple.norm_conv.weight",
    ".self_attn.q_norm.weight",
    ".self_attn.k_norm.weight",
    ".self_attn.indexer.q_layernorm.weight",
    ".self_attn.indexer.k_layernorm.weight",
)

# Fused projections: concat the checkpoint parts along dim 0 in this exact order. A nonzero pad
# rounds the merged row count up; the model splits the result back with the same sizes.
_FUSIONS: dict[str, tuple[tuple[str, ...], int]] = {
    # q carries the output gate, so its half is twice the attention width: [2*qo | kv | kv].
    ".self_attn.qkv_proj.weight": ((
        ".self_attn.q_proj.weight", ".self_attn.k_proj.weight", ".self_attn.v_proj.weight",
    ), 0),
    ".linear_attn.in_proj.weight": ((
        ".linear_attn.in_proj_qkv.weight", ".linear_attn.in_proj_z.weight",
        ".linear_attn.in_proj_b.weight", ".linear_attn.in_proj_a.weight",
    ), 0),
    ".mlp.shared_expert.gate_up_proj.weight": ((
        ".mlp.shared_expert.gate_proj.weight", ".mlp.shared_expert.up_proj.weight",
    ), 0),
    # HC mix reads the low-rank down projection and the injection logits from one GEMM; vLLM
    # pads the merged output to a multiple of 16 rows for cuBLAS (hyperconnection.py pad_size).
    # The top-level hyper_connection_mixer has no injection and so never fuses.
    ".attn_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".attn_hyper_connection.input_mix_weight_down.weight",
        ".attn_hyper_connection.block_inject_weight.weight",
    ), 16),
    ".mlp_hyper_connection.input_mix_weight_down_block_inject.weight": ((
        ".mlp_hyper_connection.input_mix_weight_down.weight",
        ".mlp_hyper_connection.block_inject_weight.weight",
    ), 16),
}


def _rename(raw_name: str) -> str | None:
    """Checkpoint key -> FreeToken state-dict key, or None to skip."""
    if raw_name.startswith(("mtp.", "model.visual.", "visual.")):
        return None
    if _PLE_TABLE_INFIX in raw_name:
        return None  # n-gram table + its scale: load_ple_table
    if _EXPERT_RE.search(raw_name):
        return None  # routed experts: offload source banks
    if raw_name.endswith(_SCALE_SUFFIXES):
        return None
    if raw_name.startswith("model.language_model."):
        return "model." + raw_name[len("model.language_model.") :]
    if raw_name.startswith("language_model."):
        return "model." + raw_name[len("language_model.") :]
    return raw_name


def _try_fuse(
    name: str, tensor: torch.Tensor, buf: dict[str, dict[int, torch.Tensor]]
) -> tuple[str, torch.Tensor] | tuple[()] | None:
    """Buffer a fusion part; return the merged ``(name, tensor)`` once all parts arrive, ``()`` while incomplete, ``None`` if ``name`` is not a fusion part."""
    for fused_suffix, (parts, pad_to) in _FUSIONS.items():
        for idx, part in enumerate(parts):
            if not name.endswith(part):
                continue
            key = name[: -len(part)] + fused_suffix
            slots = buf.setdefault(key, {})
            slots[idx] = tensor
            if len(slots) < len(parts):
                return ()
            del buf[key]
            rows = [slots[i] for i in range(len(parts))]
            pad = (-sum(t.shape[0] for t in rows)) % pad_to if pad_to else 0
            if pad:
                rows.append(torch.zeros(pad, *rows[0].shape[1:], dtype=rows[0].dtype, device=rows[0].device))
            return key, torch.cat(rows, dim=0)
    return None


# ======================================================================================
# TP sharding of the dense (non-expert) weights
# ======================================================================================
#
# Applied to the FUSED tensors (qkv_proj / in_proj / gate_up_proj), i.e. after ``_try_fuse``:
# the fusions concatenate along dim 0, and the per-segment shard below cuts the same rows the
# per-part shard would, so the two orders are equivalent. Everything not listed here
# (HC mixers, norms, PLE, the QSA indexer, the MoE router gate) is replicated: the residual
# streams R are identical on every rank because each block's output is all-reduced.


def _head_rows(t: torch.Tensor, offset: int, width: int, rank: int, size: int) -> torch.Tensor:
    """This rank's row range of a head-major block starting at ``offset`` (width % size == 0)."""
    local = width // size
    return t[offset + rank * local : offset + (rank + 1) * local]


def _shard_dense_weight(name: str, tensor: torch.Tensor, *, rank: int, size: int, config) -> torch.Tensor:
    """TP shard of one fused dense weight; identity at size 1 (never called then)."""
    if name.endswith(".self_attn.qkv_proj.weight"):
        # [2*qo | kv | kv] rows; q is 2x wide (carries the gate) and every segment is
        # head-divisible, so a per-segment dim-0 chunk stays head-aligned.
        qo = config.num_qo_heads * config.head_dim
        kv = config.num_kv_heads * config.head_dim
        segs = tensor.split([2 * qo, kv, kv], dim=0)
        return torch.cat([seg.chunk(size, dim=0)[rank] for seg in segs], dim=0)
    if name.endswith(".self_attn.o_proj.weight"):
        return tensor.chunk(size, dim=1)[rank]
    if name.endswith(".linear_attn.in_proj.weight"):
        # [qkv_conv | z | b | a], with the conv segment itself [q | k | v]; shard every
        # head-major block separately so the local layout is [q_r | k_r | v_r | z_r | b_r | a_r].
        group = config.linear_attention_group()
        key = group.num_key_heads * group.key_head_dim
        value = group.num_value_heads * group.value_head_dim
        conv = 2 * key + value
        rows = [
            _head_rows(tensor, 0, key, rank, size),
            _head_rows(tensor, key, key, rank, size),
            _head_rows(tensor, 2 * key, value, rank, size),
            _head_rows(tensor, conv, value, rank, size),  # z
            _head_rows(tensor, conv + value, group.num_value_heads, rank, size),  # b
            _head_rows(tensor, conv + value + group.num_value_heads, group.num_value_heads, rank, size),  # a
        ]
        return torch.cat(rows, dim=0)
    if name.endswith(".linear_attn.conv1d.weight"):
        # [2*key | value] depthwise rows, same head-major split as the in_proj conv segment.
        group = config.linear_attention_group()
        key = group.num_key_heads * group.key_head_dim
        value = group.num_value_heads * group.value_head_dim
        rows = [
            _head_rows(tensor, 0, key, rank, size),
            _head_rows(tensor, key, key, rank, size),
            _head_rows(tensor, 2 * key, value, rank, size),
        ]
        return torch.cat(rows, dim=0)
    if name.endswith((".linear_attn.A_log", ".linear_attn.dt_bias")):
        return tensor.chunk(size, dim=0)[rank]
    if name.endswith(".linear_attn.out_proj.weight"):
        return tensor.chunk(size, dim=1)[rank]
    if name.endswith(".mlp.shared_expert.gate_up_proj.weight"):
        half = tensor.shape[0] // 2
        return torch.cat(
            [tensor[:half].chunk(size, dim=0)[rank], tensor[half:].chunk(size, dim=0)[rank]],
            dim=0,
        )
    if name.endswith(".mlp.shared_expert.down_proj.weight"):
        return tensor.chunk(size, dim=1)[rank]
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        per_rank = div_ceil(tensor.shape[0], size)
        return tensor[rank * per_rank : min((rank + 1) * per_rank, tensor.shape[0])]
    return tensor


def iter_weights(
    model_path: str,
    device: torch.device,
    *,
    include_moe_experts: bool,
    include_non_moe: bool,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the dense (non-expert) weights, prefix-stripped and fused to the model's buffers.

    Keys keep the checkpoint's module names below the stripped prefix, so the emitted set is the
    model's state dict minus the routed experts. Nothing here is quantized: the modelopt
    ``ignore`` list covers everything except those experts, so attention, GDN, HC, PLE, the shared
    expert and lm_head are all plain bf16 (the n-gram hash constants stay int64). Fusions:
    attention q|k|v -> ``qkv_proj``, GDN ``in_proj_{qkv,z,b,a}`` -> ``in_proj``, shared-expert
    gate|up -> ``gate_up_proj``, and each per-layer HC's ``input_mix_weight_down`` |
    ``block_inject_weight`` -> a zero-padded ``input_mix_weight_down_block_inject``.

    Under TP each fused tensor is sharded for this rank (see ``_shard_dense_weight``) after
    fusion; the routed experts are sharded by expert id in ``load_nvfp4_expert_sources``.

    ``include_moe_experts`` is accepted for the loader contract but never yields anything: the
    routed experts are NVFP4 and always come from :func:`load_nvfp4_expert_sources`.
    """
    if not include_non_moe:
        return
    tp_info = get_tp_info()
    config = None
    if tp_info.size > 1:
        from .config import parse_config

        config = parse_config(cached_load_hf_config(model_path))

    def shard(name: str, tensor: torch.Tensor) -> torch.Tensor:
        if tp_info.size == 1:
            return tensor
        return _shard_dense_weight(name, tensor, rank=tp_info.rank, size=tp_info.size, config=config)

    fuse_buf: dict[str, dict[int, torch.Tensor]] = {}
    for file in tqdm(
        iter_weight_files(model_path),
        desc="Loading weights",
        disable=not tp_info.is_primary(),
    ):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for raw_name in f.keys():
                name = _rename(raw_name)
                if name is None:
                    continue
                tensor = f.get_tensor(raw_name)
                fused = _try_fuse(name, tensor, fuse_buf)
                if fused is not None:
                    if fused != ():  # () means buffered, not yet complete
                        yield fused[0], shard(*fused)
                    continue
                yield name, shard(name, tensor)

    assert not fuse_buf, f"Incomplete projection fusions: {sorted(fuse_buf)}"


# ======================================================================================
# Vision tower (model.visual.*), opt-in via FREETOKEN_LOAD_VISION=1
# ======================================================================================
#
# Separate from ``iter_weights`` on purpose: the tower is bf16, replicated on every TP rank
# (no sharding, no fusion), and only exists in the model's state dict when vision loading is
# enabled. ``_rename`` still drops these keys from the dense path unconditionally.

_VISUAL_RENAMES = {
    "model.visual.patch_embed.proj.weight": "model.visual.patch_embed.proj_weight",
    "model.visual.patch_embed.proj.bias": "model.visual.patch_embed.proj_bias",
    # nn.Embedding.weight -> the plain pos_embed tensor on Qwen4ExpVisionModel.
    "model.visual.pos_embed.weight": "model.visual.pos_embed",
}


def iter_visual_weights(
    model_path: str, device: torch.device
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield the vision tower's weights with FreeToken state-dict names (all replicated)."""
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


# ======================================================================================
# PLE n-gram table
# ======================================================================================


@dataclass(frozen=True)
class PleTable:
    """The filled n-gram table: one pinned host bank plus the checkpoint's per-tensor FP8 scale."""

    bank: HostBank
    weight_scale: torch.Tensor  # scalar, checkpoint dtype (bf16)

    @property
    def tensor(self) -> torch.Tensor:
        """``[total_rows, ngram_head_dim]`` float8_e4m3fn view of the bank."""
        return self.bank.tensor


_PLE_ST_DTYPE = "F8_E4M3"


def _safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


def _ple_table_files(folder: str) -> list[str]:
    """Shards holding a piece of the n-gram table, from the index when there is one."""
    index = os.path.join(folder, "model.safetensors.index.json")
    if not os.path.exists(index):
        return sorted(iter_weight_files(folder))
    with open(index, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    files = {shard for name, shard in weight_map.items() if _PLE_TABLE_INFIX in name}
    return sorted(os.path.join(folder, shard) for shard in files)


def load_ple_table(model_path: str, qwen4_args, *, pin: bool = True,
                   workers: int = 8, chunk: int = 8 << 20) -> PleTable:
    """Concatenate the checkpoint's ``ngram_embedding.shard_<i>`` tensors into one pinned host bank.

    The checkpoint splits the table into ``split_ngram_parts`` equal row blocks named by shard
    index and scattered over the ``model-plefp8-*`` shards in header (lexicographic) order, so the
    bank is filled shard by shard at ``shard_index * rows_per_shard``. Each read is O_DIRECT: the
    table is ~47.7 GiB and must not also sit in the page cache while the bank holds the same bytes.
    """
    folder = download_hf_weight(model_path)
    parts: dict[int, tuple[str, int, int]] = {}  # shard index -> (path, file offset, bytes)
    scale: torch.Tensor | None = None
    rows = cols = 0
    for path in _ple_table_files(folder):
        header, base = _safetensors_header(path)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key.endswith(_PLE_SCALE_SUFFIX):
                with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                    scale = f.get_tensor(key).reshape(())
                continue
            match = _PLE_SHARD_RE.search(key)
            if match is None:
                continue
            if meta["dtype"] != _PLE_ST_DTYPE:
                raise ValueError(f"PLE table shard {key} has unsupported dtype {meta['dtype']}")
            shape = meta["shape"]
            if rows and tuple(shape) != (rows, cols):
                raise ValueError(f"PLE table shard {key} is {shape}, expected {[rows, cols]}")
            rows, cols = shape
            begin, end = meta["data_offsets"]
            parts[int(match.group("shard"))] = (path, base + begin, end - begin)

    expected = int(qwen4_args.split_ngram_parts)
    if sorted(parts) != list(range(expected)):
        raise ValueError(
            f"PLE table needs shards 0..{expected - 1}, found {len(parts)}: {sorted(parts)[:8]}"
        )
    if cols != qwen4_args.ngram_head_dim:
        raise ValueError(f"PLE table row is {cols} wide, config says {qwen4_args.ngram_head_dim}")
    if scale is None:
        raise ValueError("PLE table has no weight_scale")

    bank = HostBank((expected * rows, cols), torch.float8_e4m3fn)
    shard_bytes = rows * cols
    bar = byte_bar(expected * shard_bytes, "Loading PLE table")
    try:
        buf = bank.memoryview()
        for shard in range(expected):
            path, offset, nbytes = parts[shard]
            assert nbytes == shard_bytes, f"PLE shard {shard} is {nbytes} B, expected {shard_bytes}"
            read_range_into(buf, path, file_offset=offset, nbytes=nbytes,
                            dest_offset=shard * shard_bytes, workers=workers, chunk=chunk)
            bar.update(nbytes)
    finally:
        bar.close()
    if pin and torch.cuda.is_available():
        bank.pin()
    return PleTable(bank=bank, weight_scale=scale)


# ======================================================================================
# Routed NVFP4 experts
# ======================================================================================


def _expert_shard() -> tuple[int, int] | None:
    """This rank's expert-parallel shard (experts ``id % size == rank``), None at TP=1."""
    tp_info = get_tp_info()
    return (tp_info.rank, tp_info.size) if tp_info.size > 1 else None


def load_nvfp4_expert_sources(model_path: str, config, *, layer_sink=None) -> dict:
    """Build the CPU NVFP4 expert source banks for the offload cache (gate/up fused on the output-row axis, down separate; weight_scale_2 carried as the per-row global scale)."""
    return load_nvfp4_expert_source_banks(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        layer_sink=layer_sink,
        expert_shard=_expert_shard(),
    )


def load_nvfp4_expert_sources_parallel(
    model_path: str, config, *, workers: int = 8, chunk: int = 8 << 20, layer_sink=None
) -> dict:
    """parallel: same NVFP4 source banks via the common chunked multi-threaded reader."""
    from freetoken.models.nvfp4_banks import load_nvfp4_expert_source_banks_parallel

    return load_nvfp4_expert_source_banks_parallel(
        model_path,
        config,
        _NVFP4_SOURCE_SPEC,
        drop_page_cache=drop_page_cache,
        primary=get_tp_info().is_primary(),
        workers=workers,
        chunk=chunk,
        layer_sink=layer_sink,
        expert_shard=_expert_shard(),
    )


__all__ = [
    "PleTable",
    "iter_visual_weights",
    "iter_weights",
    "load_nvfp4_expert_sources",
    "load_nvfp4_expert_sources_parallel",
    "load_ple_table",
]
