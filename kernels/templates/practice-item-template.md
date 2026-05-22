# Kernel Practice Item Template

Status: draft

## Objective

What kernel skill or performance question is this item meant to teach?

## Serving Link Or Drill Scope

Which serving benchmark or benchmark-driven investigation motivates this item? If none, why is this a kernel-only drill?

## Baseline

What implementation or library path is the baseline?

## Megakernel Gate

If this is a megakernel investigation, what profiler evidence shows kernel-boundary bubbles, launch/teardown overhead, synchronization stalls, or memory-pipeline gaps?

## Target Stack

- CUDA:
- CUTLASS:
- CuTe DSL:
- GPU:

## Correctness Check

How is correctness validated, and against what tolerance?

## Benchmark Command

```sh
# command goes here
```

## Profiling Plan

- Tool:
- Counters or views:
- Expected bottleneck:

## Results

Summarize latency, throughput, bandwidth, utilization, and relevant profiler observations.

## Lessons Learned

What changed in the mental model?

## Next Step

What is the smallest useful follow-up?
