# Wiki Index

This wiki is organized primarily around research questions, with source notes and concept pages supporting those questions.

## Research Questions

- [Blackwell kernel frontier](questions/blackwell-kernel-frontier.md): How should Blackwell kernel depth move the high-throughput decode Pareto frontier?

## Investigations

- [V4 Flash InferenceX-style frontier](investigations/v4-flash-inferencex-frontier.md): Planned serving-frontier investigation for real DeepSeek-V4-Flash on the smallest official Blackwell topology.

## Benchmark Plans

- [V4 Flash baseline benchmark plan](../benchmarks/plans/v4-flash-baseline-plan.md): Concrete first-pass run matrix for vLLM and SGLang V4 Flash baseline selection.

## Artifacts

- [V4 Flash kernel map](artifacts/v4-flash-kernel-map.md): Initial top-down architecture map for turning DeepSeek-V4-Flash facts into kernel hypotheses.

## Source Notes

- [Model-performance ramp plan: DeepSeek V4 Flash, kernel depth, Blackwell](source-notes/model-performance-ramp-plan-deepseek-v4-flash-kernel-depth-blackwell.md)
- [FlashAttention Papers 1-4](source-notes/flashattention-series.md)
- [Sparse attention for video diffusion: FA4 follow-on ramp](source-notes/sparse-attention-video-diffusion-fa4-follow-on-ramp.md)
- [SemiAnalysisAI InferenceX README](source-notes/semianalysisai-inferencex-readme.md)
- [InferenceMAX: Open Source Inference Benchmarking](source-notes/inferencemax-open-source-inference-benchmarking.md)
- [PyTorch TokenSpeed Qwen3.5 Agentic Workloads](source-notes/pytorch-tokenspeed-qwen35-agentic-workloads.md)
- [Hazy Research: Look Ma, No Bubbles](source-notes/hazyresearch-look-ma-no-bubbles-megakernel.md)
- [DeepSeek V4 Preview Release](source-notes/deepseek-v4-preview-release.md)
- [DeepSeek V4 Flash Hugging Face Model Card](source-notes/deepseek-v4-huggingface-model-card.md)
- [DeepSeek V4 Technical Report And Model Card](source-notes/deepseek-v4-technical-report-and-model-card.md)
- [vLLM DeepSeek V4 Support](source-notes/vllm-deepseek-v4-support.md)
- [SGLang DeepSeek V4 Cookbook](source-notes/sglang-deepseek-v4-cookbook.md)
- [TensorRT-LLM DeepSeek Support](source-notes/tensorrt-llm-deepseek-support.md)
- [TileRT: Speed As The Next Scaling Law](source-notes/tilert-speed-as-the-next-scaling-law.md)
- [OpenAI MRCR](source-notes/openai-mrcr.md)
- [LongBench v2](source-notes/longbench-v2.md)

## Operating Documents

- [Canonical glossary](../CONTEXT.md)
- [ADR 0001: InferenceX-style serving frontier](../docs/adr/0001-inferencex-style-serving-frontier.md)
- [Repository log](log.md)
- [LLM wiki pattern](../docs/llm-wiki-pattern.md)

## Future Index Sections

- Concept pages: decode bottlenecks, KV cache bandwidth, batching, MoE dispatch, FP4/FP8 scaling, TMA, TMEM, tcgen05, online softmax, grouped GEMM.
- Benchmark records: throughput, latency, correctness, result provenance, profiler evidence, and frontier interpretation.
- Kernel practice notes: CUDA, CUTLASS, CuTe DSL, and Blackwell pipeline exercises.
