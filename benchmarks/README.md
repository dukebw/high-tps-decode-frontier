# Benchmarks

This directory will hold benchmark harnesses, benchmark plans, result records, and measurement templates.

The benchmark standard is intentionally strict because this repo is about systematic performance engineering, not isolated speed anecdotes.

## Result Requirements

Every result should record hardware, software, workload, metrics, method, profiler evidence, interpretation, caveats, and next experiment.

Use `templates/result-template.md` for new records.

## Plans

- [V4 Flash baseline benchmark plan](plans/v4-flash-baseline-plan.md)

## Scripts

- `scripts/v4-flash-vllm-smoke.sh`: reproduces the pinned vLLM V4 Flash launch smoke test on `b200-aws2` or an equivalent 4-GPU Blackwell host.

## Initial Benchmark Families

- Dense attention across prefill, decode, append, and mixed phases.
- Sparse or compressed attention with metadata/index cost separated from attention compute.
- MoE routing, grouping, grouped GEMM, scatter/combine, and skewed expert distributions.
- Logits and sampling kernels, especially top-k/top-p, masks, structured output constraints, and speculative decode interactions.
- End-to-end serving sweeps for TTFT, TPOT, tokens/s/user, tokens/s/GPU, and p50/p95 latency.
