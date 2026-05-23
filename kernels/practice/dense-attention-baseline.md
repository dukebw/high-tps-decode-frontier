# Dense attention baseline

Status: draft

## Objective

This is meant to learn about modern state of the art attention kernels, in other words, FA4.
At the same time, it's a first opportunity to practice CUTLASS and CuTeDSL in
depth.
Performance question: how does actual attention kernel performance approach the
roofline limit as we gradually introduce improvements from FA1 to FA4?

## Serving Link Or Drill Scope

Which serving benchmark or benchmark-driven investigation motivates this item? If none, why is this a kernel-only drill?

This is a standalone kernel-only drill to build foundations in CUTLASS,
CuTeDSL, and modern dense attention kernel optimizations.
At the same time, we can derive shapes from recent models where dense attention
is relevant, like GLM-4.5, Kimi-K2.6 prefill, I believe?

## Baseline

FA4/CuTe DSL or cuDNN attention on B200, whichever is runnable and fastest for
the selected shapes.

Correctness and sanity ladder:
- Naive PyTorch attention
- PyTorch SDPA
- FlashAttention/cuDNN/FA4 path

## Megakernel Gate

N/A

## Target Stack

- CUDA, CUTLASS, and CuTeDSL: latest versions
- GPU: B200

## Correctness Check

Correctness reference: naive PyTorch float64 attention

## Benchmark Command

```sh
bench-dense-attention \
  --backend sdpa \
  --dtype bf16 \
  --seq-lens 1024,8192 \
  --head-dims 64,128 \
  --warmup 10 \
  --repeats 50 \
  --output /tmp/dense_attention.json
```

## Profiling Plan

- Tool: ncu
- Counters or views:
- Expected bottleneck: FA1 work partitioning, FA2 pipelining, FA3 N/A on
  Blackwell? FA4 math pipe throttle due to exp units.

## Results

Summarize latency, throughput, bandwidth, utilization, and relevant profiler observations.

## Lessons Learned

What changed in the mental model?

## Next Step

What is the smallest useful follow-up?
