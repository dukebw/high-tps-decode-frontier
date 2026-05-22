# TileRT: Speed As The Next Scaling Law

Sources:

- `../../sources/html/tilert-speed-as-the-next-scaling-law.html`
- `../../sources/html/tilert-speed-as-the-next-scaling-law-zh.html`

Upstreams:

- https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html
- https://www.tilert.ai/blog/speed-as-the-next-scaling-law-zh.html

Status: summarized from imported English and Chinese HTML snapshots on 2026-05-22. The sources were scanned for obvious credential, private-key, and scrubbed-company markers before adding them to this public repo; no matches were found.

## Summary

TileRT argues that ultra-low-latency decode is becoming a first-class scaling axis because agents, voice, code completion, tool calls, and test-time scaling expose per-token latency rather than just aggregate throughput. The post frames GLM-5.1 production serving as a case study for TileRT's persistent execution model: a compile-time expanded, GPU-resident Engine Kernel that uses tile-level scheduling, warp/block specialization, heterogeneous GPU workers, and in-pipeline communication to reduce kernel-boundary and runtime-orchestration gaps.

## Key Claims

- An `8xH200 NVL` server provides nearly `38 TB/s` aggregate memory bandwidth.
- For GLM-5.1, the activated parameter footprint per decode token is about `42 GB`, implying a theoretical decode-throughput limit near `1000 token/s` from bandwidth alone, without MTP.
- Real systems often deliver only a few dozen token/s, suggesting an order-of-magnitude gap between hardware limits and end-to-end decode performance.
- The post attributes much of this gap to runtime and operator boundaries: kernel launch gaps, cross-kernel barriers, global-memory spills, synchronization, communication boundaries, and host/device orchestration entering the latency-critical path at small batch sizes.
- TileRT statically expands model execution into a persistent Engine Kernel at compile time, so the host launches once and the GPU runs a persistent tile pipeline during decode.
- TileRT uses tile-level tasks as the scheduling abstraction for compute, communication, and asynchronous IO.
- TileRT applies warp and block specialization inside the Engine Kernel so some worker groups focus on data movement, others on tensor compute, and others on communication overlap.
- For GLM-5.1 attention, TileRT describes heterogeneous GPU workers: `GPU0` acts as the Sparse Indexer worker for Top-K selection, sparse-index construction, and routing decisions, while `GPU1-7` act as MLA workers for RMSNorm, GEMM, Flash Sparse Attention, and AllReduce.
- TileRT pushes communication into the execution pipeline rather than treating broadcasts, reductions, and synchronization as separate externally orchestrated stages.
- The post says TileRT serves production traffic for GLM-5 and GLM-5.1, with prior work on idle-interval compression in `v0.1.1`, MTP in `v0.1.2-alpha.1`, GLM-5 support in `v0.1.3`, and an upcoming `v0.1.4` focused on ultra-long-context production optimization and heterogeneous workers.

## Relevance To This Repo

This source is directly relevant to the repo's high-TPS decode frontier because it frames speed, TPOT, and low-batch decode as end-to-end system properties rather than isolated kernel microbenchmarks. It reinforces the repo's megakernel gate: persistent kernels and GPU-resident orchestration should be motivated by profiler evidence of kernel-boundary bubbles, launch overhead, synchronization stalls, communication gaps, or memory-pipeline interruptions.

It also gives a concrete production-oriented example of heterogeneous execution across an NVL domain, which is useful when thinking about DeepSeek-style sparse indexers, Top-K routing, compressed attention, MTP, and MoE/attention stage decomposition. The source does not by itself provide enough benchmark detail to claim a comparable frontier point.

## Kernel And Benchmark Implications

- Treat TileRT as evidence that decode bottlenecks can move from individual kernels to the execution boundaries between kernels.
- For future profiler runs, explicitly inspect launch gaps, host scheduling, cross-rank synchronization, communication/compute overlap, global-memory round-trips, and per-stage idle intervals.
- Compare any proposed persistent-kernel or megakernel intervention against existing runtime paths before assuming custom kernel work will move the end-to-end frontier.
- Heterogeneous GPU-worker designs should be evaluated as topology-specific system designs, not just single-kernel optimizations.
- The GLM-5.1 bandwidth arithmetic is useful as a sanity-check style limit, but benchmark records still need full hardware, software, workload, correctness, quality, and provenance details before being used as comparable results.

## Open Questions

- What exact GLM-5.1 model variant, precision format, context distribution, batch/concurrency level, and MTP configuration underlie the post's production claims?
- What are the measured TTFT, TPOT, tokens/s/user, tokens/s/GPU, p50/p95 latency, and quality/correctness results for the GLM-5.1 service?
- How much of the reported speedup comes from persistent Engine Kernel design versus heterogeneous GPU workers, MTP handling, memory-locality improvements, or production scheduling changes?
- Does the `GPU0` Sparse Indexer worker become a bottleneck as context length, routing complexity, or concurrency increases?
- Which TileRT components are open source and reproducible enough to compare against vLLM/SGLang-style baselines on this repo's target workloads?

## Follow-Ups

- Use this source as a checklist when analyzing profiler traces from V4 Flash decode runs.
- If TileRT publishes reproducible GLM-5.1 or DeepSeek-style benchmark commands, ingest those as separate benchmark-source artifacts.
- Compare TileRT's heterogeneous-worker idea against V4 Flash CSA/HCA indexer and MoE routing stages when designing future system-level experiments.
