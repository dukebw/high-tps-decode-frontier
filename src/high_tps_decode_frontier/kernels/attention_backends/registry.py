from __future__ import annotations

from typing import Literal

import click

from . import flash1, naive, sdpa
from .types import AttentionBackend

Backend = Literal["flash1", "naive", "sdpa"]
BACKEND_CHOICES: tuple[Backend, ...] = ("flash1", "naive", "sdpa")

BACKENDS: dict[Backend, AttentionBackend] = {
    "flash1": AttentionBackend(attention=flash1.attention),
    "naive": AttentionBackend(attention=naive.attention),
    "sdpa": AttentionBackend(attention=sdpa.attention),
}


def parse_backend(value: str) -> Backend:
    if value == "flash1":
        return "flash1"
    if value == "naive":
        return "naive"
    if value == "sdpa":
        return "sdpa"
    raise click.BadParameter(
        f"backend must be one of: {', '.join(BACKEND_CHOICES)}"
    )
