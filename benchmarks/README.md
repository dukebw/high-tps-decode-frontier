# Benchmarks

This directory will hold benchmark harnesses, result records, and measurement templates.

The benchmark standard is intentionally strict because this repo is about systematic performance engineering, not isolated speed anecdotes.

## Result Requirements

Every result should record hardware, software, workload, metrics, method, profiler evidence, interpretation, caveats, and next experiment.

Use `templates/result-template.md` for new records.

## Initial Benchmark Families

- Dense attention across prefill, decode, append, and mixed phases.
- Sparse or compressed attention with metadata/index cost separated from attention compute.
- MoE routing, grouping, grouped GEMM, scatter/combine, and skewed expert distributions.
- Logits and sampling kernels, especially top-k/top-p, masks, structured output constraints, and speculative decode interactions.
- End-to-end serving sweeps for TTFT, TPOT, tokens/s/user, tokens/s/GPU, and p50/p95 latency.
