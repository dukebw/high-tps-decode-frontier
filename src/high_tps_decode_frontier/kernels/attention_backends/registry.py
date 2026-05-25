from __future__ import annotations

from typing import Literal

import click

from . import naive, sdpa
from .types import AttentionBackend

Backend = Literal["naive", "sdpa"]
BACKEND_CHOICES: tuple[Backend, ...] = ("naive", "sdpa")

BACKENDS: dict[Backend, AttentionBackend] = {
    "naive": AttentionBackend(attention=naive.attention),
    "sdpa": AttentionBackend(attention=sdpa.attention),
}


def parse_backend(value: str) -> Backend:
    if value == "naive":
        return "naive"
    if value == "sdpa":
        return "sdpa"
    raise click.BadParameter(
        f"backend must be one of: {', '.join(BACKEND_CHOICES)}"
    )
