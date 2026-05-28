# PyTorch TokenSpeed Qwen3.5 Agentic Workloads

Source: `../../sources/html/pytorch-tokenspeed-qwen35-agentic-workloads.html`

Upstream: https://pytorch.org/blog/up-to-580tps-new-speed-record-of-qwen3-5-397b-a17b-on-gpu-for-agentic-workloads-with-tokenspeed/

Status: summarized from imported HTML snapshot on 2026-05-28. The source was scanned for obvious private-key and cloud/API-token markers after download; no private key or repo credential patterns were found. The generic credential scan did match public upstream New Relic browser-monitoring identifiers embedded in the PyTorch HTML.

## Summary

The PyTorch blog reports TokenSpeed results for `Qwen3.5-397B-A17B` on NVIDIA B200 GPUs, focusing on agentic workloads with long shared prefixes, multi-turn context growth, MTP/speculative decode, and hybrid attention state. It frames TokenSpeed as an open-source inference engine using native SPMD architecture, static compilation, prefix caching, prefill-decode disaggregation, CPU/GPU overlap, and kernel fusion to keep decode latency low.

The source is most useful to this repo as an external agentic-serving case study. Its numbers are reported by the source, not locally reproduced here, and should not be treated as comparable frontier points until benchmark commands, hardware topology, quality/correctness settings, workload generation, and tokens/s/GPU normalization are reproduced or faithfully cited in a result record.

## Key Claims

- Qwen3.5 uses a hybrid architecture with GDN/Mamba-style linear-attention layers plus periodic full-attention layers with conventional KV cache.
- TokenSpeed splits hybrid prefix-cache ownership: C++ manages the logical radix-tree, page IDs, eviction, and Mamba slot lifetime, while Python owns physical GPU tensors, stream ordering, copy-on-write, zeroing, and snapshot copies.
- Mamba prefix-cache reuse requires both cached KV page IDs and a clean recurrent-state checkpoint at the same prefix boundary.
- TokenSpeed manages separate KV-cache blocks and Mamba-state slots in the scheduler, including speculative-decoding rollback state.
- TokenSpeed's prefill-decode disaggregation transfers Mamba `conv_state` and `ssm_state` through RDMA machinery alongside KV cache by using per-layer buffer descriptors, slot offsets, a unified layer-progress counter, decode-side layer barriers, and a three-phase state-plus-token handshake.
- TokenSpeed eliminates a Mamba-state copy in speculative verification by writing each speculative step to preassigned physical rows and updating an integer `current_input_indices` table for the accepted state.
- Runtime optimizations include CUDA multi-stream overlap for shared versus routed MoE experts, dual-stream GDN input projections, fused AllReduce/residual/RMSNorm, fused Q/K RMSNorm plus partial RoPE plus gate split, fused shared-expert gate/sigmoid/mul/add, CUDA graph replay, avoiding D2H synchronization, `torch.compile` fusion for index arithmetic, and non-blocking pinned H2D transfers.
- The source says native FA4 support for Qwen3.5 `head_dim=256` is under active development after upstream community support was merged.
- The benchmark environment is described as EvalScope Benchmark, TokenSpeed latest Docker image `lightseekorg/tokenspeed-runner:latest`, and TokenSpeed commit `f797288a79deb79b9b8bcb529726a79be06278c7`.
- For the reported agentic workload, the source uses a 50K first-turn context, appends 800 tokens per subsequent turn, and runs 10-15 turns total.
- The source reports 500+ tok/s/user at batch size 1 across TP4, TP4EP4, TP8, and TP8EP8 on B200 with NVFP4, with TP8 peaking near 580 tok/s/user.
- The source reports an average KV-cache hit rate above 90% for the multi-turn agentic workload.
- For the reported 1M-context NIAH-derived run on TP8, the source reports decode throughput near 530 tok/s/user at 128K, 495 at 256K, and 445 at 1M, or about 16% degradation from 128K to 1M.

## Relevance To This Repo

This source strengthens the repo's decision to model agentic prefix-cache traffic as a first-class workload regime rather than mixing it into no-cache InferenceX frontiers. It provides a concrete public workload shape with repeated long prefixes, high cache hit rate, multi-turn growth, and long-context decode pressure.

It is also useful as a systems checklist for hybrid models whose serving bottlenecks span KV cache, recurrent state, speculative decode state, PD transfer, CPU/GPU synchronization, and launch/fusion overhead. The GDN/Mamba state-management details are not the same as DeepSeek V4 Flash's CSA/HCA stack, but the benchmark discipline is similar: cache semantics, state transfer, MTP accounting, and long-context correctness must be explicit before accepting throughput claims.

## Kernel And Benchmark Implications

- Treat reported TokenSpeed numbers as `faithfully cited` or `external reported` provenance unless reproduced locally under this repo's benchmark template.
- Any comparable record needs exact GPU count/topology, driver/CUDA versions, container digest or source SHA, TokenSpeed flags, EvalScope command, workload generator details, prefix-cache warmup policy, measured hit rate, MTP acceptance/accounting, and quality/correctness results.
- The agentic workload shape is a useful candidate comparator for this repo's current 128K/95%-cached/1K-output prefix-cache scenario, but it should be added as a separate workload variant rather than silently replacing the selected first pass.
- Mamba-state prefix-cache correctness suggests an analogous checklist for non-KV state in other hybrid-attention systems: checkpoint ownership, block alignment, copy-on-write semantics, stream-ordering guarantees, and stale-slot prevention.
- The state-transfer section is relevant to future PD-disaggregation benchmark records: bulk transfer bandwidth is not enough; layer-wise readiness, first-token handoff, and decode-side barriers affect TTFT and TPOT.
- The optimization list reinforces the repo's megakernel gate: first profile launch gaps, D2H syncs, graph replay boundaries, stream-overlap opportunities, and simple fusions before proposing larger persistent-kernel work.
- FA4 `head_dim=256` support should be tracked as a Blackwell attention-kernel baseline candidate for Qwen3.5-like models, but this source does not provide a finished TokenSpeed FA4 result.

## Open Questions

- What exact B200 topology, clocks/power state, CUDA/driver versions, and NCCL/RDMA settings were used for the reported TokenSpeed runs?
- What were TTFT, TPOT, p50/p95 latency, tokens/s/GPU, memory utilization, and MTP acceptance rates for the agentic and 1M-context runs?
- What quality or correctness suite was run for Qwen3.5 under NVFP4, MTP, prefix cache, and long-context settings?
- Are the benchmark scripts and EvalScope workload definitions pinned to immutable commits, and do they reproduce the figures from the blog?
- How much of the 580 tok/s/user result comes from MTP, prefix-cache hit rate, static compilation, CUDA graphs, kernel fusions, CPU/GPU overlap, or model-specific GDN behavior?
- Does the reported high single-user throughput remain on a non-dominated tokens/s/user versus tokens/s/GPU frontier once served concurrency is swept and throughput is normalized per GPU?

## Follow-Ups

- If TokenSpeed's Qwen3.5 benchmark scripts are public and reproducible, ingest the exact GitHub benchmark recipe as a separate source artifact.
- Compare the blog's 50K plus 800-token multi-turn agentic workload with this repo's selected 128K/95%-cached/1K-output workload before adding a second agentic-prefix-cache benchmark variant.
- Use the Mamba prefix-cache and PD-transfer invariants as a checklist when evaluating runtimes that manage non-standard recurrent or compressed attention state.
