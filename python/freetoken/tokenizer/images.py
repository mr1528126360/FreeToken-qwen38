"""Multimodal (image) intake for the tokenizer worker.

Parses OpenAI-style ``image_url`` content parts (base64 data URLs and http(s)), runs the
checkpoint's image processor (the PIL backend of Qwen2VL -- the qwen4_exp preprocessor
config), and packs the result into a serializable payload that rides ``UserMsg.mm_inputs``
to the scheduler, which runs the vision tower on it.

Token layout: the chat template renders one ``<|vision_start|><|image_pad|><|vision_end|>``
placeholder per image; after encoding, each ``image_pad`` token is expanded to the image's
soft-token count (``prod(grid_thw) // merge_size**2``), matching HF's processor output.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.request
from typing import Any

import torch

_IMAGE_TOKEN_FALLBACK = "<|image_pad|>"


def extract_image_urls(messages: Any) -> list[str]:
    """Image URLs of a chat message list (post ``render_messages``), in render order.
    Returns [] for raw-string prompts and text-only conversations."""
    urls: list[str] = []
    if not isinstance(messages, list):
        return urls
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if not url:
                    raise ValueError("image_url content part is missing its url")
                urls.append(url)
    return urls


def _load_image_bytes(url: str) -> bytes:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        if not data:
            raise ValueError("malformed data URL for image")
        if ";base64" in header:
            return base64.b64decode(data)
        from urllib.parse import unquote_to_bytes

        return unquote_to_bytes(data)
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=30) as resp:
            return resp.read()
    raise ValueError(
        f"unsupported image url scheme (expected a data: or http(s) URL): {url[:64]!r}"
    )


def decode_images(urls: list[str]):
    """URL list -> PIL images (RGB). Importing PIL here keeps it off the text-only path."""
    from PIL import Image

    images = []
    for url in urls:
        try:
            images.append(Image.open(io.BytesIO(_load_image_bytes(url))).convert("RGB"))
        except Exception as exc:
            raise ValueError(f"could not decode image: {exc}") from exc
    return images


class ImageProcessor:
    """Lazy wrapper over the checkpoint's image processor + image-token metadata."""

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._processor = None
        self._unavailable = False
        self.image_token_id: int | None = None
        try:
            with open(os.path.join(model_path, "config.json"), encoding="utf-8") as fh:
                self.image_token_id = json.load(fh).get("image_token_id")
        except OSError:
            pass

    def _load(self):
        # The PIL backend needs no torchvision and matches the shipped preprocessor_config
        # (Qwen2VLImageProcessorFast declares the same resize/normalize parameters).
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
            Qwen2VLImageProcessorPil,
        )

        return Qwen2VLImageProcessorPil.from_pretrained(self._model_path)

    @property
    def available(self) -> bool:
        if self.image_token_id is None or self._unavailable:
            return False
        if self._processor is None:
            try:
                self._processor = self._load()
            except Exception:
                self._unavailable = True
                return False
        return True

    def process(self, urls: list[str]) -> dict:
        """Images -> mm payload: expanded token counts + serialized pixel tensor."""
        if not self.available:
            raise ValueError("this model does not support image inputs")
        processor = self._processor
        out = processor(images=decode_images(urls), return_tensors="pt")
        pixel_values = out["pixel_values"]  # [total_patches, C*T*P*P] float32
        grid_thw = out["image_grid_thw"]  # [num_images, 3]
        merge = int(processor.merge_size)
        token_counts = (grid_thw.prod(-1) // merge**2).tolist()
        # bf16 is the vision tower's compute dtype (HF casts pixels before patch_embed),
        # so this halves the wire bytes at zero precision cost.
        payload_bytes = (
            pixel_values.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
        )
        return {
            "pixel_values": payload_bytes,
            "dtype": "bfloat16",
            "shape": list(pixel_values.shape),
            "grid_thw": [[int(v) for v in row] for row in grid_thw.tolist()],
            "token_counts": [int(n) for n in token_counts],
        }


def expand_image_tokens(
    input_ids: torch.Tensor, image_token_id: int, token_counts: list[int]
) -> torch.Tensor:
    """Replace each single ``image_pad`` placeholder with the image's full soft-token span."""
    positions = (input_ids == image_token_id).nonzero().view(-1).tolist()
    if len(positions) != len(token_counts):
        raise ValueError(
            f"chat template emitted {len(positions)} image placeholders but the request "
            f"carries {len(token_counts)} images"
        )
    parts = []
    prev = 0
    for pos, count in zip(positions, token_counts):
        parts.append(input_ids[prev : pos + 1])
        if count > 1:
            parts.append(input_ids[pos : pos + 1].repeat(count - 1))
        prev = pos + 1
    parts.append(input_ids[prev:])
    return torch.cat(parts)


__all__ = ["ImageProcessor", "decode_images", "expand_image_tokens", "extract_image_urls"]
