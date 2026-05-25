from __future__ import annotations

import torch
import torch.nn.functional as F


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    return F.softmax((q @ k.T) * scale, dim=-1) @ v
