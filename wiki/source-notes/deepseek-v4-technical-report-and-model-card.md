# DeepSeek V4 Technical Report And Model Card

Sources:

- `../../sources/pdf/deepseek-v4-technical-report.pdf`
- `../../sources/pdf/deepseek-v4-model-card-en.pdf`

Upstreams:

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf
- https://fe-static.deepseek.com/chat/transparency/deepseek-V4-model-card-EN.pdf

Status: summarized from imported PDF snapshots on 2026-05-22. The sources were scanned for obvious credential patterns before adding them to the public repo.

## Summary

The DeepSeek V4 technical report is the main architecture and systems source for V4. It details the V4 family, hybrid CSA/HCA attention, mHC, DeepSeekMoE changes, MTP inheritance, FP4 quantization-aware training, TileLang kernel development, fine-grained MoE communication-compute overlap, deterministic kernels, KV-cache management, on-disk KV cache storage, and model evaluation.

The shorter model-card PDF provides a public summary of model properties, distribution channels, acceptable-use framing, and high-level data statements.

## Key Claims

- V4 retains Transformer, DeepSeekMoE, and MTP foundations while adding mHC and hybrid CSA/HCA attention.
- DeepSeek-V4-Flash uses 43 Transformer layers and hidden dimension 4096.
- V4-Flash uses pure sliding-window attention in the first two layers, then interleaves CSA and HCA in subsequent layers.
- For V4-Flash CSA, compression rate `m = 4`, indexer query heads `n_I_h = 64`, indexer head dimension `c_I = 128`, and sparse-attention top-k is 512.
- For V4-Flash HCA, compression rate `m' = 128`.
- For V4-Flash CSA/HCA, query heads are 64, head dimension is 512, query compression dimension is 1024, output projection groups are 8, and each intermediate attention output dimension is 1024.
- The additional sliding-window branch uses window size 128.
- V4-Flash uses MoE layers in all Transformer blocks, with Hash routing in the first 3 MoE layers.
- Each V4-Flash MoE layer has 1 shared expert and 256 routed experts, with 6 routed experts activated per token and expert intermediate hidden dimension 2048.
- V4-Flash sets MTP depth to 1, mHC expansion factor to 4, and Sinkhorn-Knopp iterations to 20.
- The report claims V4-Flash has 284B total parameters and 13B active parameters, while the model-card PDF rounds the Flash total size to 285B.
- The report describes a single fused MoE kernel that overlaps Dispatch, Linear-1, activation/cast, Linear-2, and Combine.
- The report says the CUDA MegaMoE implementation is open-sourced as part of DeepGEMM.
- The report says DeepSeek uses TileLang for fused kernels, host codegen, SMT-assisted integer analysis, numerical precision controls, and reproducibility.
- The report describes a heterogeneous KV-cache layout with classical CSA/HCA KV cache and a state cache for SWA and uncompressed tail tokens.

## Relevance To This Repo

This source is the foundation for the `v4-flash-kernel-map.md` artifact. It converts vague architecture terms into concrete layer counts, compression ratios, top-k values, expert counts, activation counts, precision choices, and cache-management structures that can drive kernel hypotheses.

## Kernel And Benchmark Implications

- The initial kernel map should prioritize CSA/HCA decode, indexer top-k selection, compressed KV insertion, SWA state cache, grouped output projection, and MoE dispatch/GEMM/combine.
- V4-Flash MoE work should treat Hash-routed initial layers differently from routed layers if profiling shows different dispatch behavior.
- Public DeepSeek/DeepGEMM or runtime-integrated MegaMoE paths should be treated as baseline configurations when they are reproducible and stronger than non-MegaMoE configs, before attempting any new custom megakernel.
- Benchmark correctness needs to account for MTP depth, reasoning-mode prompt format, and mixed FP4/FP8 checkpoint behavior.
- Long-context serving benchmarks should explicitly track KV-cache layout, compressed-token block units, SWA state, and prefix-cache behavior.
- The report's model-evaluation tables should be used as external reference sanity checks for the first report-only V4 Flash quality gate when the eval task, context length, reasoning mode, and scoring settings are comparable.

## Open Questions

- What is the exact CSA/HCA layer ordering for V4-Flash, beyond the first two SWA layers and the interleaved statement?
- Which details are best taken from the official model `config.json` versus the paper tables and prose?
- How closely do vLLM and SGLang match DeepSeek's own inference framework cache layout and fused-kernel choices?
- Which DeepGEMM MegaMoE version is the strongest public baseline for V4-Flash on Blackwell?

## Follow-Ups

- Record pinned Hugging Face references for `config.json`, tokenizer, encoding, and inference docs from the official Hugging Face repo.
- Expand the V4 Flash kernel map with known facts, unknowns, and benchmark-driven hypotheses.
- Link MoE megakernel hypotheses to DeepGEMM MegaMoE and runtime recipe support.
