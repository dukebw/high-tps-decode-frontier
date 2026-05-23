from __future__ import annotations

import json
from pathlib import Path

import click


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

    config = {
        "backend": backend,
        "dtype": dtype,
        "seq_lens": _parse_int_csv(seq_lens, "seq-lens"),
        "head_dims": _parse_int_csv(head_dims, "head-dims"),
        "warmup": warmup,
        "repeats": repeats,
    }

    # Write the config as a stand-in observable behaviour until the actual benchmark is
    # implemented.
    rendered = json.dumps(config, indent=2, sort_keys=True)
    if output is None:
        click.echo(rendered)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n")
    click.echo(f"wrote {output}")


if __name__ == "__main__":
    main()
