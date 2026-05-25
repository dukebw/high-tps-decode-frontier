from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PREVIEW_COLUMNS = (
    "backend",
    "dtype",
    "gpu",
    "seq_len",
    "head_dim",
    "median_ms",
    "min_ms",
    "correct",
    "max_abs_error",
)


def _format_preview_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _print_table(rows: list[dict[str, Any]]) -> None:
    widths = {
        column: max(
            len(column),
            *(len(_format_preview_value(row.get(column))) for row in rows),
        )
        for column in PREVIEW_COLUMNS
    }
    header = "  ".join(column.ljust(widths[column]) for column in PREVIEW_COLUMNS)
    divider = "  ".join("-" * widths[column] for column in PREVIEW_COLUMNS)
    print(header)
    print(divider)
    for row in rows:
        print(
            "  ".join(
                _format_preview_value(row.get(column)).ljust(widths[column])
                for column in PREVIEW_COLUMNS
            )
        )


def _correctness_by_key(
    payload: dict[str, Any],
) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    checks = payload.get("correctness")
    if not isinstance(checks, list):
        return {}

    indexed = {}
    for check in checks:
        if not isinstance(check, dict):
            continue
        key = (check.get("backend"), check.get("dtype"), check.get("head_dim"))
        indexed[key] = check
    return indexed


def _preview_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results")
    if not isinstance(results, list):
        return []

    checks = _correctness_by_key(payload)
    rows = []
    for result in results:
        if not isinstance(result, dict):
            continue
        key = (result.get("backend"), result.get("dtype"), result.get("head_dim"))
        check = checks.get(key, {})
        rows.append(
            {
                "backend": result.get("backend"),
                "dtype": result.get("dtype"),
                "gpu": payload.get("cuda_device_name"),
                "seq_len": result.get("seq_len"),
                "head_dim": result.get("head_dim"),
                "median_ms": result.get("median_ms"),
                "min_ms": result.get("min_ms"),
                "correct": check.get("passed"),
                "max_abs_error": check.get("max_abs_error"),
            }
        )
    return rows


def print_artifact_previews(output_dir: Path) -> None:
    previewed = False
    for path in sorted(output_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        rows = _preview_rows(payload)
        if not rows:
            continue

        if not previewed:
            print()
            print("benchmark preview")
            previewed = True

        print(f"\n{path.name}")
        _print_table(rows)
