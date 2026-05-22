# Blackwell Kernel Frontier

Status: active

## Question

How should datacenter Blackwell kernel depth move the Pareto frontier for high-throughput text LLM decode, measured in tokens/s/user while preserving useful latency, quality, cost, and production complexity tradeoffs?

## Scope

- Hardware target: B200/GB200-class datacenter Blackwell systems.
- First-class kernel stack: CUDA, CUTLASS, and CuTe DSL.
- Serving target: text LLM decode, especially high-concurrency and long-context workloads.
- Research style: source synthesis, bottleneck prediction, strict benchmark records, profiler-backed claims, and kernel practice items.

## Initial Hypotheses

## Evidence So Far

## First Experiments To Define

- Dense attention baseline across prefill, decode, and append shapes, with a clear comparison between naive, PyTorch/SDPA, FlashAttention-family, and Blackwell-aware paths where available.

## Open Questions
