# vLLM DeepSeek V4 Support

Sources:

- `../../sources/html/vllm-deepseek-v4-blog.html`
- `../../sources/html/vllm-recipes-deepseek-v4-flash.html`
- `../../sources/github/vllm-recipes-deepseek-v4-flash.yaml`

Upstreams:

- https://vllm.ai/blog/2026-04-24-deepseek-v4
- https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4-Flash.yaml

Status: summarized from imported HTML and YAML snapshots on 2026-05-22. The sources were scanned for obvious credential patterns before adding them to the public repo; the YAML contains a documented `api_key="EMPTY"` OpenAI client placeholder, not a real credential.

## Summary

The vLLM blog announces DeepSeek V4 support and explains vLLM's implementation of the V4 long-context attention mechanism. The recipe page provides concrete deployment guidance for `deepseek-ai/DeepSeek-V4-Flash`, including hardware, runtime flags, reasoning-mode details, and prefill/decode disaggregation examples.

For this repo, vLLM is a likely strongest public reproducible baseline path for V4 Flash serving on Blackwell.

## Key Claims

- vLLM supports `deepseek-ai/DeepSeek-V4-Pro` and `deepseek-ai/DeepSeek-V4-Flash`.
- The YAML recipe sets minimum vLLM version to `0.20.0` and default strategy to `single_node_tep`.
- The highlighted V4-Flash command is runnable on 4xB200 or 4xB300.
- The highlighted V4-Flash command uses `--kv-cache-dtype fp8`, `--block-size 256`, expert parallelism, `--data-parallel-size 4`, CUDA graph compilation settings, FP4 indexer cache, `--tokenizer-mode deepseek_v4`, `--tool-call-parser deepseek_v4`, and `--reasoning-parser deepseek_v4`.
- The YAML recipe's Blackwell override adds `--attention_config.use_fp4_indexer_cache=True` and `--moe-backend deep_gemm_mega_moe`.
- vLLM describes `c4a` as roughly 1/4 KV compression with 8-token weighted sums and stride 4.
- vLLM describes `c128a` as roughly 1/128 KV compression with 128-token weighted sums and stride 128.
- vLLM uses a short sliding window of size 128 for local information.
- vLLM estimates DeepSeek V4 bf16 KV cache at 1M context as 9.62 GiB per sequence for a 61-layer stack, before further fp4/fp8 reductions.
- vLLM fixes logical blocks at 256 native token positions across compressed layers.
- vLLM treats compressor residual state like sliding-window KV state.
- vLLM collapses the five-way cache stack into three page-size buckets.
- vLLM reports kernel fusions for compressor plus RMSNorm plus RoPE plus cache insertion, inverse RoPE plus fp8 quant, and fused Q norm plus KV RoPE plus K insert.
- vLLM reports 5-6% end-to-end latency reduction at low batch sizes from multi-stream overlap on the decode path.
- vLLM planned work includes DeepGEMM MegaMoE and a paged prefill kernel.

## Relevance To This Repo

This source gives a concrete open runtime path that can be benchmarked locally or cited until local hardware is available. It also identifies implementation bottlenecks that align with the repo's kernel map: cache packing, compressor state, elementwise fusions, RoPE/inverse-RoPE, indexer overlap, and MegaMoE.

## Kernel And Benchmark Implications

- The smallest public Blackwell vLLM path for V4 Flash appears to be 4xB200 or 4xB300.
- Baseline records should capture the vLLM container tag, recipe source, flags, CUDA graph config, block size, cache dtype, parser modes, and whether disaggregated serving is enabled.
- Kernel work should not duplicate vLLM's existing fused kernels without first profiling whether they remain bottlenecks on the target hardware and workload.
- If vLLM has public DeepGEMM MegaMoE support by benchmark time, treat it as part of the vLLM baseline config sweep when it outperforms non-MegaMoE configs under the same workload and quality gate.
- The vLLM recipe is a candidate baseline, but the exact strongest baseline still depends on comparing vLLM and SGLang recipes under the same InferenceX-style workload.

## Open Questions

- Does the full `1M` context launch path need any additional vLLM flags beyond the smoke-test command before quality or performance runs?
- How does vLLM's 256-native-position block abstraction interact with InferenceX-like random prompt/output lengths and prefix-cache-disabled runs?

## Follow-Ups

- Use pinned vLLM image `docker.io/vllm/vllm-openai@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b` unless a later benchmark plan explicitly supersedes it.
- Compare the planned vLLM single-node TEP config against the SGLang config matrix before selecting the strongest baseline.
- Use vLLM's described fusions as a profiler checklist for decode traces.
