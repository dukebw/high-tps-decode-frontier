# Sparse Attention For Video Diffusion: FA4 Follow-On Ramp

Source: `../../sources/html/sparse-attention-video-diffusion-fa4-follow-on-ramp.html`

Status: summarized from imported raw HTML on 2026-05-21. The source was scanned for obvious credentials, emails, and explicit confidential markers before import.

## Summary

This source proposes a compact 2-4 day module after FlashAttention-4 for studying sparse attention in video diffusion. Its central argument is that FA4 teaches how to make exact dense attention fast on Blackwell, while sparse video attention teaches when dense attention is the wrong operator and how difficult it is to convert sparsity into real wall-clock speed.

Although the source is about video diffusion rather than text LLM decode, it is useful for this repo because it isolates transferable kernel lessons: sparsity metadata, token layout, block lists, mask generation, contiguity, load balance, tensor-core utilization, and the difference between theoretical FLOP reduction and actual serving speed.

## Key Claims

- Video diffusion attention differs from LLM decode attention. LLM decode reads from a growing KV cache with causal structure, while video DiT diffusion repeatedly computes mostly bidirectional 3D latent attention across denoising steps.
- FLOP sparsity is not wall-clock sparsity. Sparse K/V blocks that are scattered can lose coalescing, tensor-core efficiency, occupancy, and load balance.
- Training-free methods are the best first production experiments, while trainable sparse operators and full-stack acceleration methods define more of the frontier.
- Sparse methods must be compared against strong dense and cached baselines, especially FA4 on B200/GB200 where available.
- A fair sparse-attention report must include mask/search overhead, clustering/permutation cost, metadata movement, memory usage, profiler evidence, and quality risks.

## Method Taxonomy

- Static local or tiled sparsity teaches tile windows, boundary handling, and block-sparse QK scheduling.
- Dynamic block selection teaches mask generation overhead, ragged block lists, and per-head load balance.
- Semantic permutation teaches that sparse work must often be packed into contiguous blocks before tensor cores can use it efficiently.
- Exact plus approximate blocks preserve quality by approximating non-critical work instead of dropping all skipped context.
- Trainable sparse operators move sparsity into model architecture and require finetuning, distillation, and evaluation.
- Caching and broadcasting methods can beat custom sparse kernels in production because they exploit timestep redundancy without requiring a new sparse operator.

## Relevance To Text Decode

The video-specific quality metrics and bidirectional 3D attention structure do not transfer directly to text LLM decode. The systems lessons do transfer. Any sparse or compressed decode attention experiment should account for metadata generation, block packing, index overhead, load balance, and fallback dense baselines.

This source also reinforces the repo rule that every claimed frontier improvement needs an end-to-end accounting. A sparse kernel that wins on idealized FLOPs but loses to FA4, caching, quantization, batching, or simpler runtime changes is not a Pareto improvement.

## Kernel And Benchmark Implications

- Use FA4 or the best available dense attention path as the baseline to beat.
- Include mask generation, search, clustering, permutation, and metadata costs in timing.
- Track temporary memory and profiler counters, not just latency.
- Inspect load imbalance across query blocks, L2 reuse, coalescing, tensor-core utilization, register pressure, CTA scheduling, and metadata bandwidth.
- For synthetic sparse attention tests, vary block size, density, token layout, and contiguity rather than only sparsity percentage.

## Open Questions

- Which sparse-attention lessons are most applicable to long-context text decode, where the KV cache and causal structure dominate?
- Can compressed or sparse decode attention be made tensor-core-friendly without excessive metadata overhead?
- What is the strongest dense baseline on B200 for each target shape, and when is sparse attention even a plausible competitor?
- How should quality risk be represented for text decode when attention structure or precision changes?

## Follow-Ups

- Add a concept page for "FLOP sparsity vs wall-clock sparsity" after more sparse-attention sources are ingested.
- Design a toy sparse/compressed attention benchmark that reports mask/index overhead separately from attention compute.
- Add a benchmark result template field for baseline strength, so sparse experiments cannot compare against weak dense baselines.
