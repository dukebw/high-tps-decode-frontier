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

- Attention remains a primary decode bottleneck, but the exact bottleneck shifts between KV-cache bandwidth, non-matmul work, metadata handling, scheduler interactions, and tensor-core utilization depending on workload shape.
- Blackwell-specific primitives such as TMA, TMEM, tcgen05, and 1SM/2SM MMA change how attention and GEMM kernels should be reasoned about; pipeline bubbles matter as much as peak FLOP utilization.
- MoE serving requires treating routing, dispatch/combine, ragged grouped GEMM, FP4/FP8 scaling, and all-to-all behavior as a single performance path rather than isolated kernels.
- Sparse or compressed attention only matters if metadata generation, block packing, memory movement, and quality risk are included in the end-to-end accounting.
- The strongest learning artifacts are before/after reports that connect a model/workload fact to a bottleneck hypothesis, benchmark, profiler evidence, intervention, and next experiment.

## Evidence So Far

- The Baseten DeepSeek V4 Flash ramp source frames model analysis as a forcing function for kernel-learning exercises, including hybrid attention, compressed KV, MoE grouped GEMM, FP4/FP8 scaling, MTP/speculative decode, and Blackwell B200 primitives.
- The sparse video attention source warns that theoretical FLOP sparsity is not wall-clock sparsity; selected work must be laid out and scheduled in a tensor-core-friendly way.

## First Experiments To Define

- Dense attention baseline across prefill, decode, and append shapes, with a clear comparison between naive, PyTorch/SDPA, FlashAttention-family, and Blackwell-aware paths where available.
- Compressed or sparse attention toy benchmark that separates metadata/index preparation from attention compute.
- MoE routing plus grouped GEMM toy benchmark with uniform and skewed expert routing.
- Blackwell CuTe/CUTLASS exercise that explains one GEMM or attention data path from GMEM to TMA to SMEM to TMEM to epilogue.

## Open Questions

- What exact B200/GB200 access will be available, and what fallback hardware should proxy early experiments?
- Which runtime should anchor the first end-to-end serving benchmark: vLLM, SGLang, TensorRT-LLM, or a smaller microbenchmark harness?
- Which model or proxy shapes best represent the target high-TPS decode frontier without over-investing in full-model infrastructure too early?
- What quality bar should be tracked when changing attention, precision, or speculative decode behavior?
