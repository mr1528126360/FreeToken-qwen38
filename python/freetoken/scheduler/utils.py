from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from freetoken.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    uid: int
    input_ids: torch.Tensor
    sampling_params: SamplingParams
    chunked_req: ChunkedReq | None = None
    mm_embeds: torch.Tensor | None = None
    # Raw multimodal payload (online path); encoded into mm_embeds at prefill time.
    mm_inputs: dict | None = None

    @property
    def input_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens


def has_mm(req) -> bool:
    """Carries image content in either form (precomputed embeds or raw pixels).
    getattr-based so SimpleNamespace test doubles carrying only ``mm_embeds`` keep working."""
    return (
        getattr(req, "mm_embeds", None) is not None
        or getattr(req, "mm_inputs", None) is not None
    )


@dataclass
class ScheduleResult:
    reqs: List[PendingReq]
    output_indices: List[torch.Tensor]
