# Benchmark Result Template

Status: draft

## Result Provenance

- Provenance: local measured / reproduced public recipe / faithfully cited / estimated limit
- Source or run ID:

## Question

What performance question does this result answer?

## Hypothesis

What should dominate, and why?

## Hardware

- GPU:
- GPU count and topology:
- Driver:
- CUDA:
- Clock or power state:
- CPU and memory:

## Software

- Runtime or harness:
- Library versions:
- Git SHAs:
- Hugging Face references, if used: repo ID, revision commit SHA, file paths
- Build flags:
- Environment notes:

## Workload

- Model:
- Phase: prefill / decode / append / mixed
- Workload source and scenario:
- Input length distribution:
- Output length distribution:
- Served concurrency sweep:
- Sweep policy: fixed / adaptive / other
- Request-rate policy:
- Prefix-cache policy:
- Prefix-cache hit target and measured hit rate, if enabled:
- Total ISL and uncached delta tokens, if prefix-cache enabled:
- Shape details:
- Dtype:

## Frontier Definition

- X-axis: tokens/s/user
- Y-axis: tokens/s/GPU
- Frontier construction method:

## Baselines And Limits

- Strongest baseline path:
- Baseline provenance:
- Speed-of-light hardware specs:
- Roofline limit:

## Method

- Command:
- Source and model revisions:
- Warmup policy:
- Repetitions:
- Variance summary:
- Excluded runs:
- Profiler:

## Correctness

- Correctness level: serving benchmark correctness / kernel benchmark correctness
- Reference:
- Tolerance or validation criteria:

## Quality Gate

- Required for serving benchmarks: yes / not applicable for kernel-only benchmark
- Predeclared eval suite:
- Gate mode: report-only / pass-fail
- General capability check:
- Target-use check:
- External reference scores:
- Sanity-check interpretation:
- Passing criteria, if pass-fail:

## Metrics

- TTFT:
- TPOT:
- Tokens/s/user:
- Tokens/s/GPU:
- Latency p50:
- Latency p95:
- Memory:
- Utilization:
- Other counters:

## Result

What happened?

## Interpretation

What evidence supports the bottleneck diagnosis?

## Frontier Impact

Did this produce a new non-dominated point or curve segment? If not, what evidence did it produce?

## Caveats

What should not be generalized from this result?

## Next Experiment

What is the smallest useful follow-up?
