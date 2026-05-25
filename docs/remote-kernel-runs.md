# Remote Kernel Runs

This repo's first remote execution path is deliberately narrow: a Modal task runs one repository command on one requested GPU type and returns a small artifact bundle. It is for exploratory kernel correctness and timing work, not for formal serving results or profiler-counter certification.

## Scope

- Backend: Modal only.
- GPU request: raw Modal GPU string, such as `H100!` or `B200`.
- Pricing model: task-style, pay for the remote function duration.
- Profiling: none in the first pass; `ncu` support needs a separate provider certification test.
- Artifacts: saved locally under `.remote-runs/` and ignored by git.
- Size expectation: keep returned artifact bundles small. Use a Modal volume or object storage later if profiles or traces become large.
- Environment capture: `env.txt` records selected non-secret runtime variables, not the full remote environment.
- Runtime shape: the Modal image contains CUDA, Python, and benchmark dependencies; the local repo is mounted at runtime and added to `PYTHONPATH` so source edits do not force an image rebuild.
- Cost note: the first run may spend extra time building and caching the dependency image before the benchmark command starts.
- App lifecycle: each `modal run` creates a short-lived ephemeral Modal app. That is expected; after the dependency image is cached, repeated runs should not rebuild the image unless the image definition or dependencies change.

## Setup

```sh
python3 -m pip install -e '.[remote]'
modal token new
```

## Smoke Run

```sh
modal run src/high_tps_decode_frontier/remote/modal_kernel_run.py \
  --gpu 'H100!'
```

The default command runs a tiny `bench-dense-attention` smoke and writes remote output into `$HTDF_ARTIFACT_DIR`.

The runner mounts the local working tree, not only committed files. `run.json` records `git_sha` and `git_dirty` so exploratory runs against local edits are distinguishable from clean committed runs.

## Run A Command

```sh
modal run src/high_tps_decode_frontier/remote/modal_kernel_run.py \
  --gpu B200 \
  --cmd 'python -m high_tps_decode_frontier.kernels.bench_dense_attention --backend sdpa --dtype bf16 --seq-lens 1024 --head-dims 64 --warmup 1 --repeats 1 --output "$HTDF_ARTIFACT_DIR/dense_attention.json"'
```

Each run writes a local bundle like:

```text
.remote-runs/20260525T120000Z-modal-H100/
  env.txt
  machine.json
  run.json
  stdout.txt
  stderr.txt
  dense_attention.json
```

After extracting artifacts, the runner delegates to the generic benchmark artifact previewer. The previewer prints a compact summary for any returned JSON file with a top-level `results` list. If the JSON also includes a top-level `correctness` list, the previewer folds in pass/fail and max-error fields where it can match rows by backend, dtype, and head dimension. The preview is only terminal output; the JSON files remain the source of record.

Use `.remote-runs/` for exploratory evidence. Promote selected results into `benchmarks/results/` only after the benchmark methodology, correctness checks, metadata, and provenance have been reviewed.
