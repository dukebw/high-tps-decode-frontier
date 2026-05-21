# Benchmark Result Template

Status: draft

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
- Build flags:
- Environment notes:

## Workload

- Model or proxy:
- Phase: prefill / decode / append / mixed
- Input length distribution:
- Output length distribution:
- Batch or concurrency:
- Shape details:
- Dtype:
- Baseline:

## Method

- Command:
- Warmup:
- Repetitions:
- Excluded runs:
- Profiler:

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

## Caveats

What should not be generalized from this result?

## Next Experiment

What is the smallest useful follow-up?
