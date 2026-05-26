from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from collections.abc import ByteString, Mapping
from pathlib import Path
from typing import Any

import modal  # type: ignore[import-not-found]

REMOTE_REPO = Path("/workspace/high-tps-decode-frontier")
REMOTE_SRC = REMOTE_REPO / "src"
DEFAULT_COMMAND = (
    "python -m high_tps_decode_frontier.kernels.bench_dense_attention "
    "--backend sdpa --dtype bf16 --seq-lens 1024 --head-dims 64 "
    "--warmup 1 --repeats 1 "
    '--output "$HTDF_ARTIFACT_DIR/dense_attention.json"'
)
SAFE_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "HOME",
    "HOSTNAME",
    "HTDF_ARTIFACT_DIR",
    "LD_LIBRARY_PATH",
    "NVIDIA_VISIBLE_DEVICES",
    "PATH",
    "PWD",
    "PYTHONPATH",
    "PYTHONUNBUFFERED",
    "SHELL",
    "USER",
    "VIRTUAL_ENV",
}


def _find_project_root() -> Path:
    for path in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir():
            return path

    return REMOTE_REPO


PROJECT_ROOT = _find_project_root()
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("click>=8.1", "ninja>=1.11", "torch>=2.8")
    .add_local_dir(PROJECT_ROOT, remote_path=str(REMOTE_REPO), copy=False)
)

app = modal.App("high-tps-decode-frontier-kernel-runs")


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _as_text(value: str | bytes | ByteString | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return bytes(value).decode("utf-8", errors="replace")


def _run_capture(args: list[str], cwd: Path = REMOTE_REPO) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
    except Exception as error:  # noqa: BLE001 - metadata capture should not fail the run.
        return {"error": repr(error)}

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _selected_env(env: Mapping[str, str]) -> str:
    lines = ["# selected non-secret environment variables"]
    omitted = 0
    for key, value in sorted(env.items()):
        if key not in SAFE_ENV_KEYS:
            omitted += 1
            continue
        lines.append(f"{key}={value}")
    lines.append(f"# omitted {omitted} environment variables")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _tar_directory(path: Path) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(path, arcname=".")
    return buffer.getvalue()


@app.function(image=image)
def _run_remote_command(
    command: str,
    gpu_request: str,
    run_id: str,
    git_sha: str,
    git_dirty: bool,
    command_timeout_seconds: int,
) -> dict[str, Any]:
    artifact_dir = Path("/tmp/high-tps-decode-frontier") / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    machine = {
        "provider": "modal",
        "gpu_request": gpu_request,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_l": _run_capture(["nvidia-smi", "-L"]),
        "nvidia_smi_topo": _run_capture(["nvidia-smi", "topo", "-m"]),
        "nvidia_smi_query_gpu": _run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "nvcc_version": _run_capture(["nvcc", "--version"]),
        "python_version": _run_capture(["python", "--version"]),
        "pip_freeze": _run_capture(["python", "-m", "pip", "freeze"]),
    }
    _write_json(artifact_dir / "machine.json", machine)

    started_at = _utc_now()
    env = os.environ.copy()
    env.update(
        {
            "HTDF_ARTIFACT_DIR": str(artifact_dir),
            "PYTHONPATH": str(REMOTE_SRC),
            "PYTHONUNBUFFERED": "1",
        }
    )
    (artifact_dir / "env.txt").write_text(_selected_env(env))

    try:
        completed = subprocess.run(
            command,
            cwd=REMOTE_REPO,
            env=env,
            shell=True,
            executable="/bin/bash",
            text=True,
            capture_output=True,
            check=False,
            timeout=command_timeout_seconds,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        timed_out = False
    except subprocess.TimeoutExpired as error:
        exit_code = 124
        stdout = _as_text(error.stdout)
        stderr = _as_text(error.stderr)
        stderr += f"\ncommand timed out after {command_timeout_seconds}s\n"
        timed_out = True

    finished_at = _utc_now()
    (artifact_dir / "stdout.txt").write_text(stdout)
    (artifact_dir / "stderr.txt").write_text(stderr)

    run = {
        "artifact_schema": "remote-kernel-run-v1",
        "command": command,
        "command_timeout_seconds": command_timeout_seconds,
        "exit_code": exit_code,
        "finished_at": finished_at,
        "git_dirty": git_dirty,
        "git_sha": git_sha,
        "gpu_request": gpu_request,
        "profile_mode": "none",
        "provider": "modal",
        "remote_repo": str(REMOTE_REPO),
        "run_id": run_id,
        "started_at": started_at,
        "timed_out": timed_out,
    }
    _write_json(artifact_dir / "run.json", run)

    return {
        "artifacts_tgz": _tar_directory(artifact_dir),
        "exit_code": exit_code,
        "run_id": run_id,
    }


def _local_git(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "gpu"


def _unpack_artifacts(bundle: bytes, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
        output_root = output_dir.resolve()
        for member in archive.getmembers():
            target = (output_dir / member.name).resolve()
            if output_root != target and output_root not in target.parents:
                raise ValueError(f"refusing to extract path outside output dir: {member.name}")

        try:
            archive.extractall(output_dir, filter="data")
        except TypeError:
            archive.extractall(output_dir)


def _print_artifact_previews(output_dir: Path) -> None:
    from high_tps_decode_frontier.benchmarks.artifact_preview import print_artifact_previews

    print_artifact_previews(output_dir)


def _print_remote_command(output_dir: Path) -> None:
    try:
        payload = json.loads((output_dir / "run.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return

    if not isinstance(payload, dict):
        return

    command = payload.get("command")
    if not isinstance(command, str) or not command:
        return

    print()
    print("remote command")
    print(command)


@app.local_entrypoint()
def main(
    cmd: str = DEFAULT_COMMAND,
    gpu: str = "H100!",
    out: str = ".remote-runs",
    timeout_seconds: int = 3600,
) -> None:
    """Run a repo command on a Modal GPU and save returned artifacts locally."""

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}-modal-{_safe_name(gpu)}"
    git_sha = _local_git(["rev-parse", "HEAD"])
    git_dirty = bool(_local_git(["status", "--porcelain"]))

    result = _run_remote_command.with_options(
        gpu=gpu,
        timeout=timeout_seconds + 300,
    ).remote(
        cmd,
        gpu,
        run_id,
        git_sha,
        git_dirty,
        timeout_seconds,
    )

    output_dir = PROJECT_ROOT / out / run_id
    _unpack_artifacts(result["artifacts_tgz"], output_dir)
    print(f"wrote {output_dir}")
    _print_remote_command(output_dir)
    _print_artifact_previews(output_dir)

    exit_code = int(result["exit_code"])
    if exit_code != 0:
        raise SystemExit(exit_code)
