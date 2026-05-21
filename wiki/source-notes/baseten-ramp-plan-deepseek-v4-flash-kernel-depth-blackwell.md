# Baseten Ramp Plan: DeepSeek V4 Flash, Kernel Depth, Blackwell

Source: `../../sources/html/baseten-ramp-plan-deepseek-v4-flash-kernel-depth-blackwell.html`

Status: summarized from imported raw HTML on 2026-05-21. The source was scanned for obvious credentials, emails, and explicit confidential markers before import.

## Summary

This source proposes a 21-day ramp plan that uses DeepSeek-V4-Flash as a forcing function for model-performance engineering and kernel depth. The core progression is top-down: analyze the model architecture, predict likely bottlenecks, build targeted microbenchmarks and kernels, map the work to Blackwell primitives, and end with a production-style before/after performance report.

The plan emphasizes that kernel expertise should be driven by the real workload rather than by disconnected toy kernels. DeepSeek-V4-Flash is used as the workload because it stresses multiple frontier-serving paths: long context, hybrid attention, compressed KV, MoE, mixed FP4/FP8 precision, MTP/speculative decode, and Blackwell B200 execution details.

## Key Claims

- A useful ramp should produce artifacts, not just reading progress: architecture memos, microbenchmarks, Blackwell notes, profiler evidence, and a capstone report.
- Full-model infrastructure should not consume the whole ramp. If B200/GB200 access and model support are easy, run the real stack; otherwise use official configs, recipes, and proxy shapes.
- Long-context attention is the primary kernel thesis for V4 Flash, but hybrid attention adds complexity through compressed KV pools, sparse indexer metadata, prefix-cache coherence, and non-dense attention paths.
- MoE is the secondary kernel thesis. Routing, ragged expert batches, grouped GEMM, dispatch/combine, FP4 scale layout, and all-to-all behavior can dominate if they do not align with the engine.
- Blackwell performance requires understanding the pipeline, not just occupancy: GMEM to TMA to SMEM, tcgen05 MMA into TMEM, copy-out to registers, epilogue, and overlap between copy, math, and store.
- Every benchmark should record model/proxy, runtime, hardware, shape, phase, dtype, metrics, profiler, hypothesis, and result.

## Relevance To This Repo

This source is the closest match for the repository's initial shape. It supports a balanced layout with research notes, strict benchmark templates, and kernel practice conventions. It also motivates the first research question: how CUDA, CUTLASS, and CuTe DSL on datacenter Blackwell can move the tokens/s/user frontier for text LLM decode.

The source also argues for keeping kernel work tied to serving reality. A custom kernel is not automatically the right answer; a runtime configuration, library path, batching change, or serving-stack fix may be the better intervention if it produces a clearer end-to-end win.

## Kernel And Benchmark Implications

- Start with dense attention and online softmax before sparse or compressed variants.
- Treat metadata preparation and indexing as first-class costs in sparse/compressed attention experiments.
- Build MoE benchmarks that include routing, grouping, GEMM, scatter/combine, and skewed expert distributions.
- Study FP4/FP8 scale layout and epilogue conversion as kernel details, not just dtype labels.
- Use profiler evidence to distinguish HBM traffic, softmax/non-matmul work, metadata overhead, tensor-core utilization, and scheduler gaps.

## Open Questions

- Which DeepSeek-V4-Flash facts are stable enough to use as benchmark assumptions?
- Which proxy shapes best approximate the real workload if full B200/GB200 model serving is unavailable?
- Which runtime path should be compared first: vLLM, SGLang, TensorRT-LLM, FlashInfer, or microbench-only code?
- What minimum profiler evidence should be required before calling a kernel exercise successful?

## Follow-Ups

- Turn the source's benchmark metadata block into `benchmarks/templates/result-template.md`.
- Turn the source's capstone report structure into the standard performance-report format for this repo.
- Create first concept pages for Blackwell TMA, TMEM, tcgen05, online softmax, grouped GEMM, and FP4 scale layout after more primary sources are ingested.
