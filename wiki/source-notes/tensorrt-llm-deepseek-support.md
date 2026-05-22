# TensorRT-LLM DeepSeek Support

Sources:

- `../../sources/html/tensorrt-llm-supported-models.html`
- `../../sources/html/tensorrt-llm-deepseek-v32-blackwell.html`
- `../../sources/html/tensorrt-llm-gvr-topk-sparse-attention.html`

Upstreams:

- https://nvidia.github.io/TensorRT-LLM/models/supported-models.html
- https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.html
- https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog21_Temporal_Correlation_Meets_Sparse_Attention.html

Status: summarized from imported HTML snapshots on 2026-05-22. The sources were scanned for obvious credential patterns before adding them to the public repo.

## Summary

TensorRT-LLM is a strong NVIDIA runtime with public DeepSeek, MoE, sparse-attention, Blackwell, and benchmarking support. Its current public docs include `trtllm-serve`, `trtllm-bench`, `trtllm-eval`, DeepSeek-V3/V3.2 model support, DSA Top-K kernels, GVR Top-K, MTP, disaggregated serving, chunked prefill, KV-cache reuse, and Blackwell-oriented MoE optimization material.

For this repo, TensorRT-LLM is a credible conditional baseline candidate, but not yet a required first-run baseline for V4 Flash because the imported public sources do not show an explicit `deepseek-ai/DeepSeek-V4-Flash` model support row or V4 Flash deployment recipe.

## Key Claims

- TensorRT-LLM exposes `trtllm-serve`, `trtllm-bench`, and `trtllm-eval` for serving, benchmarking, and evaluation.
- The current supported-models page lists `DeepseekV3ForCausalLM` for DeepSeek-V3/Kimi-K2 and `DeepseekV32ForCausalLM` for DeepSeek-V3.2.
- The current supported-models page does not list DeepSeek-V4, DeepSeek-V4-Flash, or `DeepseekV4ForCausalLM`.
- TensorRT-LLM's DeepSeek-V3.2 blog describes Blackwell support for DSA, FP4/NVFP4, FP8 KV cache, MTP, disaggregated serving, chunked prefill, KV-cache reuse, Wide-EP, sparse MLA kernels, indexer Top-K, DeepGEMM MQA kernels, and kernel fusion.
- The GVR Top-K blog describes an exact DeepSeek Sparse Attention Top-K optimization integrated into TensorRT-LLM, currently for `index_topk=2048`, with future extension toward DeepSeek V4-style smaller Top-K settings such as 512 or 1024.
- The public docs describe TensorRT-LLM support for serving, benchmarking, evaluation, custom kernels, MoE, prefill/decode disaggregation, wide expert parallelism, and speculative decoding.

## Relevance To This Repo

TensorRT-LLM should be tracked because it may become the strongest public baseline if it publishes a reproducible V4 Flash path. It is also relevant to kernel hypotheses around sparse-attention Top-K, Blackwell FP4/FP8 paths, MoE, MTP, and disaggregated serving.

## Benchmark Implications

- The first V4 Flash baseline set should require vLLM and SGLang because both have explicit V4 Flash recipes in the imported sources.
- TensorRT-LLM should be added to the first baseline run only if a public, reproducible V4 Flash model support path or deployment recipe exists by benchmark time.
- If TensorRT-LLM is included later, it must run the same InferenceX-like workload, quality gate, reasoning modes, provenance labeling, and prefix-cache policy as vLLM and SGLang.
- TensorRT-LLM's DSA Top-K and GVR material should inform profiler interpretation, but should not be treated as evidence that it serves V4 Flash today.

## Open Questions

- Will TensorRT-LLM publish a DeepSeek-V4-Flash support row or recipe before the first baseline run?
- If generic DeepSeek-V3/V3.2 code can load V4 Flash, which unsupported gaps remain: CSA/HCA cache layout, tokenizer/encoding, reasoning modes, FP4/FP8 checkpoint loading, or parser behavior?
- If TensorRT-LLM can run V4 Flash only through AutoDeploy or a custom loader, is that reproducible enough to count as a baseline path?

## Follow-Ups

- Re-check TensorRT-LLM support when hardware is available or before finalizing the first benchmark plan.
- Do not make TensorRT-LLM a required first baseline until exact V4 Flash support is public and reproducible.
