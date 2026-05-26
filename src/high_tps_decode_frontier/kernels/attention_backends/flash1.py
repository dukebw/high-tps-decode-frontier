from __future__ import annotations

from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXTENSION_DIR = Path(__file__).parent / "cuda"


@cache
def _extension():
    return load(
        name="high_tps_decode_frontier_flash1",
        sources=[
            str(_EXTENSION_DIR / "flash1_bindings.cpp"),
            str(_EXTENSION_DIR / "flash1.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _extension().attention(q, k, v)
