# V4 Flash Kernel Map

Status: initial official-source map, not yet benchmark validated

## Source Base

- [DeepSeek V4 Technical Report And Model Card](../source-notes/deepseek-v4-technical-report-and-model-card.md) is the primary architecture source.
- [DeepSeek V4 Flash Hugging Face Model Card](../source-notes/deepseek-v4-huggingface-model-card.md) is the primary open-weight and serving-protocol source.
- [vLLM DeepSeek V4 Support](../source-notes/vllm-deepseek-v4-support.md) and [SGLang DeepSeek V4 Cookbook](../source-notes/sglang-deepseek-v4-cookbook.md) are candidate baseline runtime sources.

## Model Frame

- Target model: real `deepseek-ai/DeepSeek-V4-Flash`.
- Scale: 284B total parameters in the technical report and Hugging Face model card, sometimes rounded to 285B in runtime/model-card summaries.
- Active parameters: 13B per token.
- Context length: 1M tokens.
- Layer count: 43 Transformer layers.
- Hidden dimension: 4096.
- Precision: instruct checkpoint uses FP4 MoE expert parameters and FP8 for most other parameters.
- MTP: depth 1, inherited from DeepSeek-V3.

## Attention

### Known Facts

- V4 Flash uses pure sliding-window attention in the first two layers.
- Subsequent layers interleave CSA and HCA.
- CSA compression rate is `m = 4`.
- CSA sparse-attention top-k is 512.
- CSA indexer query heads are 64.
- CSA indexer head dimension is 128.
- HCA compression rate is `m' = 128`.
- CSA/HCA query heads are 64.
- CSA/HCA head dimension is 512.
- CSA/HCA query compression dimension is 1024.
- Grouped output projection uses 8 groups with intermediate attention output dimension 1024 per group.
- The additional sliding-window branch uses window size 128.
- CSA and HCA use shared key-value multi-query attention and grouped output projection.
- The technical report applies RoPE on the last 64 dimensions of attention queries, KV entries, and core attention outputs.
- vLLM describes a 256-native-token logical block size for compressed layers.

### Kernel Hypotheses

- CSA decode likely stresses token-level compression, indexer query/key work, top-k selection, compressed KV reads, sliding-window KV reads, and grouped output projection.
- HCA decode likely stresses compression and dense attention over heavily compressed KV entries rather than sparse top-k selection.
- The sliding-window branch should be profiled separately because it preserves local tokens and has different cache behavior from compressed CSA/HCA entries.
- RoPE and inverse-RoPE around shared K/V attention are candidates for elementwise fusion only if profiler traces show material HBM round trips or launch overhead.
- KV-cache layout and block sizing may dominate long-context memory behavior more than the arithmetic inside a single attention kernel.

### Unknowns To Resolve

- Exact CSA/HCA layer ordering after the first two SWA layers.
- Exact Hugging Face config fields corresponding to every report parameter.
- Whether the strongest vLLM and SGLang paths use the same cache layout, page size, and compressor-state model.
- Whether CSA/HCA bottlenecks appear in prefill, decode, or both under an InferenceX-like workload.

## MoE

### Known Facts

- V4 Flash uses MoE layers in all Transformer blocks.
- The first 3 MoE layers use Hash routing.
- Each MoE layer has 1 shared expert and 256 routed experts.
- Each token activates 6 routed experts.
- Expert intermediate hidden dimension is 2048.
- Expert weights use FP4 in the instruct checkpoint.
- The technical report describes a single fused MoE kernel that overlaps dispatch, GEMM, activation/cast, and combine.
- The technical report says the CUDA MegaMoE implementation is open-sourced as part of DeepGEMM.
- SGLang exposes MegaMoE W4A8 and W4A4 options on Blackwell, but not for low-latency or context-parallel recipes.
- vLLM lists DeepGEMM MegaMoE as planned work in the DeepSeek V4 support post.

### Kernel Hypotheses

- MoE dispatch/GEMM/combine is a likely first bottleneck candidate at high served concurrency.
- Hash-routed initial layers may have different routing and load-balance behavior than later routed layers.
- FP4 expert weights make dequantization, scaling, and epilogue placement part of the MoE kernel question.
- Existing DeepGEMM/SGLang MegaMoE paths are baseline configurations when they can be enabled through public recipes and outperform the runtime's other tested configs.

### Unknowns To Resolve

- Which MegaMoE or grouped-GEMM path is strongest on the selected Blackwell topology.
- Whether MoE is the bottleneck at low TPOT points, max-throughput points, or both.
- Whether W4A4 MegaMoE has acceptable quality impact for the repo's formal eval suite.

## mHC And Small Kernels

### Known Facts

- mHC expansion factor is 4.
- Sinkhorn-Knopp iterations are set to 20.
- DeepSeek reports fused kernels and recomputation strategies for mHC.

### Kernel Hypotheses

- mHC may create small elementwise or matrix operations that matter only if they show up as launch-bound decode overhead.
- mHC should not be optimized before attention, MoE, KV-cache, and runtime-boundary evidence is collected.

## Runtime Baselines

- vLLM publishes a V4 Flash command runnable on 4xB200 or 4xB300.
- SGLang maps V4 Flash to 4-GPU B200, GB200, GB300, and H200 serving paths, and to an 8-GPU H100 path.
- TensorRT-LLM is a conditional candidate: its public docs show strong DeepSeek-V3/V3.2, DSA, Blackwell, and benchmarking support, but no explicit V4 Flash support row or deployment recipe in the imported sources.
- vLLM and SGLang both expose parser and tokenizer options specific to DeepSeek V4.
- SGLang exposes low-latency, balanced, max-throughput, context-parallel, and PD-disaggregated recipes.
- vLLM exposes single-node and PD-disaggregated recipe guidance.
- MegaMoE belongs to the baseline sweep, not the intervention track, when it is public, reproducible, and stronger than non-MegaMoE runtime configs.

## Must Measure First

- End-to-end serving frontier for the strongest public vLLM recipe.
- End-to-end serving frontier for the strongest public SGLang recipe.
- Decode profiler traces at low-latency, balanced, and high-throughput frontier points.
- Attention versus MoE versus runtime-boundary contribution to TPOT.
- KV-cache memory pressure and cache-transfer behavior at long prompts.
- Quality impact for reasoning mode, MTP, FP4/FP8 path, and any MegaMoE option.
