from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

AttentionFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AttentionBackend:
    attention: AttentionFn
