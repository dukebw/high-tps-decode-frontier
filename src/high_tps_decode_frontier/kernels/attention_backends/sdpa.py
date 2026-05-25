from __future__ import annotations

import torch
import torch.nn.functional as F


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q[None, None, :, :],
        k[None, None, :, :],
        v[None, None, :, :],
        dropout_p=0.0,
        is_causal=False,
    )
    return out.squeeze(0).squeeze(0)
