# Hazy Research: Look Ma, No Bubbles

Source: `../../sources/html/hazyresearch-look-ma-no-bubbles-megakernel.html`

Upstream: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles

Status: summarized from imported HTML snapshot on 2026-05-22. The source was scanned for obvious credential patterns before import.

## Summary

This source describes a low-latency megakernel for Llama-3.2-1B that merges a model forward pass into a single long-running GPU kernel. The goal is to eliminate memory pipeline bubbles caused by many small kernel launches, launch/teardown overhead, straggler blocks, and coarse inter-kernel synchronization.

The implementation uses an on-GPU interpreter style: each SM executes a pre-scheduled sequence of instructions, instructions share and release shared-memory pages, and explicit counters synchronize dependencies inside the megakernel. The reported result is substantially higher memory-bandwidth utilization for low-latency batch-size-one inference than vLLM and SGLang baselines.

## Key Claims

- Low-latency single-sequence LLM inference can be strongly memory-bound, dominated by loading model weights from global memory.
- Existing engines break a forward pass into many small kernels, leaving memory pipeline bubbles between kernels.
- CUDA graphs reduce launch overhead but may still leave meaningful idle time for this workload.
- A megakernel can pipeline memory loads across instructions inside one kernel and avoid strict global kernel-boundary synchronization.
- Megakernels require explicit resource management and synchronization because CUDA no longer provides kernel-boundary dependency guarantees.
- The authors report 78% H100 memory-bandwidth utilization and more than 1.5x speedup over existing systems for the tested Llama-1B low-latency case.

## Relevance To This Repo

This source provides the canonical meaning of megakernel for this repo: a fused critical-path kernel that treats the GPU like a VM, scheduling instruction streams inside one long-running kernel to reduce launch, teardown, synchronization, and memory-pipeline bubbles.

It is especially relevant to the repo's high-tokens/s/user goal because the target regime is low TPOT and high interactivity. It also clarifies that megakernel work should be justified by a serving or kernel benchmark where many small kernels or memory pipeline bubbles are visible in profiler evidence.

## Kernel And Benchmark Implications

- A megakernel benchmark should compare against the strongest decomposed-kernel baseline, not just a naive kernel sequence.
- The benchmark should report whether the workload is memory-bound and whether global memory loads are kept continuously in flight.
- Profiling should look for kernel-boundary gaps, straggler block bubbles, launch overhead, shared-memory pressure, synchronization overhead, and activation-load latency.
- Correctness tests must cover the entire fused critical path because kernel-boundary dependency guarantees are replaced by explicit synchronization.

## Open Questions

- Which V4 Flash or V4 Flash subpath has enough short-kernel launch overhead or pipeline bubbles to justify a megakernel?
- Does the megakernel approach remain valuable at the target served concurrency, or mainly at batch-size-one/min-latency points?
- How should this repo structure megakernel code that acts like an on-GPU interpreter rather than a conventional fused operator?

## Follow-Ups

- Add a megakernel practice item only after a profiler trace shows enough critical-path kernel-boundary overhead to justify it.
- Compare this approach against CUDA graphs, PDL, torch.compile fusion, and runtime scheduling changes before claiming a megakernel win.
