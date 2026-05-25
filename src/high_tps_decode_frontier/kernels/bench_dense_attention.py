from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

import click
import torch
import torch.nn.functional as F

DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

ABS_TOLERANCES = {
    "bf16": 0.125,
    "fp16": 0.05,
    "fp32": 0.01,
}


def _attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    scale = q.shape[-1] ** -0.5
    return F.softmax((q @ k.T) * scale, dim=-1) @ v


def _attention_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    out = F.scaled_dot_product_attention(
        q[None, None, :, :],
        k[None, None, :, :],
        v[None, None, :, :],
        dropout_p=0.0,
        is_causal=False,
    )
    return out.squeeze(0).squeeze(0)


def _attention(
    backend: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    if backend == "naive":
        return _attention_naive(q, k, v)
    if backend == "sdpa":
        return _attention_sdpa(q, k, v)
    raise ValueError(f"unsupported backend: {backend}")


def _make_tensors(
    seq_len: int,
    prev_seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    q = torch.randn((seq_len, head_dim), dtype=dtype, device="cuda", generator=generator)
    k = torch.randn(
        (prev_seq_len, head_dim), dtype=dtype, device="cuda", generator=generator
    )
    v = torch.randn(
        (prev_seq_len, head_dim), dtype=dtype, device="cuda", generator=generator
    )
    return q, k, v


def _ensure_int_sequence(value: str | int | Sequence[int], name: str) -> list[int]:
    if isinstance(value, str) or isinstance(value, int):
        raise TypeError(f"{name} must be a sequence of integers")

    return [int(item) for item in value]


def _ensure_int(value: str | int | Sequence[int], name: str) -> int:
    if isinstance(value, Sequence) and not isinstance(value, str):
        raise TypeError(f"{name} must be an integer")

    return int(value)


def _correctness_check(
    backend: str,
    dtype_name: str,
    dtype: torch.dtype,
    seq_len: int,
    head_dim: int,
    seed: int,
) -> dict[str, float | int | bool | str]:
    check_seq_len = min(seq_len, 128)
    q, k, v = _make_tensors(check_seq_len, check_seq_len, head_dim, dtype, seed)

    with torch.inference_mode():
        ref = _attention_naive(q.double(), k.double(), v.double())
        actual = _attention(backend, q, k, v).double()

    abs_error = (actual - ref).abs()
    max_abs_error = float(abs_error.max().item())
    mean_abs_error = float(abs_error.mean().item())
    tolerance_abs = ABS_TOLERANCES[dtype_name]

    return {
        "backend": backend,
        "dtype": dtype_name,
        "seq_len": check_seq_len,
        "head_dim": head_dim,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "tolerance_abs": tolerance_abs,
        "passed": max_abs_error <= tolerance_abs,
    }


def _time_attention(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int,
    repeats: int,
) -> tuple[list[float], float, int]:
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        out: torch.Tensor | None = None
        for _ in range(warmup):
            out = _attention(backend, q, k, v)

        torch.cuda.synchronize()

        timings_ms = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = _attention(backend, q, k, v)
            end.record()
            end.synchronize()
            timings_ms.append(float(start.elapsed_time(end)))

        if out is None:
            raise RuntimeError("attention benchmark produced no output")

        checksum = float(out.float().mean().item())
        peak_memory_bytes = int(torch.cuda.max_memory_allocated())

    return timings_ms, checksum, peak_memory_bytes


def bench_dense_attention(config: Mapping[str, str | int | Sequence[int]]) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense attention benchmark")

    backend = str(config["backend"])
    dtype_name = str(config["dtype"])
    dtype = DTYPES[dtype_name]
    seq_lens = _ensure_int_sequence(config["seq_lens"], "seq_lens")
    head_dims = _ensure_int_sequence(config["head_dims"], "head_dims")
    warmup = _ensure_int(config["warmup"], "warmup")
    repeats = _ensure_int(config["repeats"], "repeats")

    results = []
    correctness = []
    for seq_len in seq_lens:
        for head_dim in head_dims:
            seed = 17 + seq_len * 1009 + head_dim * 9176
            check = _correctness_check(backend, dtype_name, dtype, seq_len, head_dim, seed)
            correctness.append(check)
            if not check["passed"]:
                raise RuntimeError(
                    "dense attention correctness failed: "
                    f"backend={backend} dtype={dtype_name} seq_len={seq_len} "
                    f"head_dim={head_dim} max_abs_error={check['max_abs_error']} "
                    f"tolerance_abs={check['tolerance_abs']}"
                )

            q, k, v = _make_tensors(seq_len, seq_len, head_dim, dtype, seed)
            timings_ms, checksum, peak_memory_bytes = _time_attention(
                backend, q, k, v, warmup, repeats
            )

            results.append(
                {
                    "backend": backend,
                    "dtype": dtype_name,
                    "seq_len": seq_len,
                    "prev_seq_len": seq_len,
                    "head_dim": head_dim,
                    "warmup": warmup,
                    "repeats": repeats,
                    "mean_ms": statistics.fmean(timings_ms),
                    "median_ms": statistics.median(timings_ms),
                    "min_ms": min(timings_ms),
                    "max_ms": max(timings_ms),
                    "timings_ms": timings_ms,
                    "checksum": checksum,
                    "peak_memory_bytes": peak_memory_bytes,
                }
            )

    return {
        "backend": backend,
        "dtype": dtype_name,
        "device": "cuda",
        "cuda_device_name": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "correctness": correctness,
        "results": results,
    }


def _parse_int_csv(value: str, option_name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise click.BadParameter(
            f"{option_name} must be a comma-separated integer list"
        ) from error

    if not parsed:
        raise click.BadParameter(f"{option_name} must include at least one integer")

    if any(item <= 0 for item in parsed):
        raise click.BadParameter(f"{option_name} values must be positive")

    return parsed


@click.command()
@click.option(
    "--backend", type=click.Choice(["naive", "sdpa"]), default="sdpa", show_default=True
)
@click.option(
    "--dtype",
    type=click.Choice(["bf16", "fp16", "fp32"]),
    default="bf16",
    show_default=True,
)
@click.option("--seq-lens", default="1024,8192", show_default=True)
@click.option("--head-dims", default="64,128", show_default=True)
@click.option("--warmup", default=10, show_default=True, type=click.IntRange(min=0))
@click.option("--repeats", default=50, show_default=True, type=click.IntRange(min=1))
@click.option("--output", type=click.Path(path_type=Path), default=None)
def main(
    backend: str,
    dtype: str,
    seq_lens: str,
    head_dims: str,
    warmup: int,
    repeats: int,
    output: Path | None,
) -> None:
    """Benchmark dense attention baselines."""

    config: dict[str, str | int | Sequence[int]] = {
        "backend": backend,
        "dtype": dtype,
        "seq_lens": _parse_int_csv(seq_lens, "seq-lens"),
        "head_dims": _parse_int_csv(head_dims, "head-dims"),
        "warmup": warmup,
        "repeats": repeats,
    }

    try:
        results = bench_dense_attention(config)
    except RuntimeError as error:
        raise click.ClickException(str(error)) from error

    rendered = json.dumps(results, indent=2, sort_keys=True)
    if output is None:
        click.echo(rendered)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n")
    click.echo(f"wrote {output}")


if __name__ == "__main__":
    main()
