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

- Decode-path kernels split into compute-bound (attention, MoE/dense GEMM) and memory-bound (normalization, RoPE, softmax, loss, elementwise) work. The memory-bound class is bounded by HBM bandwidth, so its frontier contribution is "how close to the DRAM roofline can we get," and a public CuTe DSL baseline ([QuACK](../source-notes/dao-ailab-quack-readme.md)) already reports ~90% of peak on H100 — meaning custom memory-bound kernels need to clear a high bar, not beat a naive baseline.
- CuTe DSL is a credible authoring path for both kernel classes, but speed-of-light comes with per-operator and per-input-shape tuning cost; whether that effort pays off end to end is itself an open frontier question.

## Evidence So Far

- [QuACK CuTe Kernels](../source-notes/dao-ailab-quack-readme.md): a public Apache-2.0 CuTe DSL kernel library (FlashAttention authors) on the repo's first-class stack. Demonstrates near-speed-of-light memory-bound kernels (RMSNorm, softmax, cross entropy) — 89.7% of peak HBM3 DRAM bandwidth on H100, ~50% over torch.compile at large reductions — via TV-layout coalesced load/store plus hierarchical (thread/warp/block/cluster-DSMEM) reduction, and ships Blackwell `sm100`/`sm120` GEMM + block-scaled (MX) epilogues. Caveats: published speed-of-light numbers are H100, not Blackwell; no attention kernels; it is a kernel library, not a serving runtime, so it contributes kernel-benchmark evidence, not standalone frontier points.

## First Experiments To Define

- Dense attention baseline across prefill, decode, and append shapes, with a clear comparison between naive, PyTorch/SDPA, FlashAttention-family, and Blackwell-aware paths where available.
- Memory-bound decode-kernel roofline check on Blackwell: run QuACK RMSNorm/softmax/cross-entropy (and RoPE) benchmarks on B200/B300, record measured percentage of HBM3e speed-of-light, and confirm whether cluster reduction still dominates at large reductions on Blackwell.

## Open Questions

- What fraction of Blackwell HBM speed-of-light do CuTe DSL memory-bound kernels reach, versus the published 89.7% on H100?
- Are decode-path memory-bound kernels (norm/RoPE/softmax/loss) a large enough slice of end-to-end time to matter, or are they dominated by attention and MoE GEMM?
- How do CuTe DSL block-scaled (MX/FP8) Blackwell GEMMs compare to cuBLAS, CUTLASS reference, and DeepGEMM for V4 Flash MoE and indexer shapes?
