# V4 Flash InferenceX-Style Frontier

Status: planned / blocked on target hardware access and strongest-baseline choice

## Target

- Model: real DeepSeek-V4-Flash only. Official open weights and runtime recipes are available; local benchmark setup still needs pinned Hugging Face references for config, tokenizer, encoding, and runtime versions.
- Workloads: public InferenceX random request scenarios used unchanged for no-cache comparability, plus a first-class agentic prefix-cache workload for high-ISL multi-turn coding/agent traffic.
- Hardware: smallest official Blackwell topology that can serve the model without offload or workload distortion. Current candidates include 4xB200, 4xB300, 4xGB200, and 4xGB300, depending on runtime recipe and accessible hardware.
- Frontier: continuous tokens/s/user versus tokens/s/GPU curve, with served concurrency swept to generate datapoints.

## Goal

Produce a benchmark-driven investigation that establishes the strongest public reproducible baseline for real V4 Flash serving, then uses profiling to identify whether runtime/config changes, library paths, kernel groups, custom kernels, or megakernels can create new non-dominated frontier points.

## Baseline Requirements

- Use the strongest baseline path available for the target model/workload/hardware shape.
- Treat released runtime configs, library kernels, custom kernels, or megakernels as baseline paths when they are public and reproducible enough to run or faithfully cite.
- Run both vLLM and SGLang V4 Flash recipes before declaring the strongest baseline.
- Include TensorRT-LLM in the first baseline run only if a public, reproducible V4 Flash model support path or deployment recipe exists by benchmark time.
- Treat MegaMoE as part of a runtime baseline when it can be enabled through a public recipe or documented runtime config and performs better than that runtime's other tested configs.
- Label every number with result provenance: local measured, reproduced public recipe, faithfully cited, or estimated limit.
- Use pinned live Hugging Face references for official model metadata, tokenizer, encoding, and inference files instead of copying those files into `sources/`, as long as benchmark records capture exact repo IDs, immutable revision commit SHAs, and file paths.
- Run both workload regimes before declaring the strongest baseline: no-cache public InferenceX scenarios and the first agentic prefix-cache scenario.
- First concrete benchmark plan: [V4 Flash baseline benchmark plan](../../benchmarks/plans/v4-flash-baseline-plan.md).

## Benchmark Shape

- Sweep served concurrency rather than plotting concurrency directly.
- First pass uses a fixed served-concurrency sweep only: 1, 2, 4, 8, 16, 32, 64, 128, and 256. Run the same grid across runtimes, configs, and workload regimes; adaptive refinement can be a later benchmark pass.
- First pass uses a closed-loop infinite-rate request policy: keep the target served concurrency in flight and immediately submit a replacement request when one finishes. Open-loop arrival-rate traffic is deferred to later production/SLO realism work.
- Use the public InferenceX input/output scenarios unchanged before adding any V4-Flash-specific workload variants.
- Treat prefix-cache-enabled agent/coding traffic as a separate workload regime with its own frontier, not as a modifier silently mixed into the no-cache InferenceX frontier.
- For prefix-cache-enabled runs, record total ISL, uncached delta tokens, target cached-prefix percentage, measured cache hit rate, cache warmup policy, and output length distribution.
- First agentic prefix-cache scenario: 128K total ISL, 95% cached prefix, approximately 6.4K uncached delta tokens, and 1K OSL. Add 512K and 1M total-ISL variants after the harness is stable.
- Warmup policy is workload-specific: disable warmup for the no-cache public InferenceX regime, and use explicit cache warmup for the agentic prefix-cache regime to reach the target cached-prefix percentage before timing starts.
- Run multiple repetitions for every first-pass benchmark point. Default to 3 repetitions per point unless hardware availability forces an explicit exception.
- Record TTFT, TPOT, tokens/s/user, tokens/s/GPU, p50/p95 latency, memory, and utilization for first-pass benchmark points. Collect profiler traces manually as follow-up experimentation on selected points, rather than requiring profiling for every point.
- State request-rate policy, warmup policy, prefix-cache policy, repetitions, excluded runs, and caveats.
- Compare against speed-of-light hardware specs and workload-derived roofline limits where applicable.

## Correctness

- Serving benchmark correctness must validate that the intended model, tokenizer, configuration, runtime, and request protocol are being served without feature regressions.
- Kernel benchmark correctness must validate numerical agreement with a reference implementation under stated tolerances.
- Every serving benchmark must include a predeclared formal eval suite as its serving quality gate.

## Predeclared Serving Quality Gate

The first V4 Flash serving-frontier benchmark uses two required long-context evals before performance numbers can be treated as valid:

- OpenAI MRCR at 128K, 512K, and 1M bins.
- Entire LongBench v2 `long` subset.

MRCR is the explicit 1M-context retrieval/disambiguation gate. The first quality gate uses all 100 MRCR 2-needle samples in each selected bin: 128K, 512K, and 1M. LongBench v2 long is the realistic long-context reasoning gate, and the first quality gate uses the entire long subset. Both evals must run in two V4 Flash reasoning modes: Non-think and Think High. Sampling, max-context, and output-token settings should be fixed and declared in the benchmark record.

The first baseline uses this quality gate in report-only mode. It records scores before performance results are interpreted but does not impose pass/fail thresholds until baseline distributions exist. Scores should be sanity-checked against external reference sources where settings are comparable, including the DeepSeek V4 technical report and public MRCR or LongBench v2 reference results. Any mismatch must be labeled with provenance and explained rather than silently treated as a failure.

## Kernel Extraction

Kernel benchmarks should be extracted from serving-benchmark bottlenecks, not invented independently. Candidate slices include:

- Attention and KV-cache paths.
- MoE routing, dispatch, grouped GEMM, and combine.
- Low-precision scale handling and epilogues.
- Logits, sampling, structured-output masks, and specdecode/MTP paths.
- Runtime/kernel-boundary gaps visible in traces.

## Megakernel Gate

A megakernel investigation can start only after profiler evidence shows critical-path kernel-boundary bubbles, launch/teardown overhead, synchronization stalls, or memory-pipeline gaps that simpler runtime, graph, fusion, or scheduling options cannot remove.

## Current Evidence

- [InferenceMAX: Open Source Inference Benchmarking](../source-notes/inferencemax-open-source-inference-benchmarking.md) defines the frontier methodology this investigation follows.
- [PyTorch TokenSpeed Qwen3.5 Agentic Workloads](../source-notes/pytorch-tokenspeed-qwen35-agentic-workloads.md) provides an external case study for high-hit-rate multi-turn agentic prefix-cache workloads and hybrid-state serving optimizations, but not a directly comparable V4 Flash baseline.
- [X: Fern CoCo Augmentation Thread](../source-notes/x-hi-tysam-coco-augmentation-thread.md) motivates future corrupted-context and rare-token robustness probes for agentic prefix-cache quality gates, but is not a throughput baseline or kernel source.
- [SemiAnalysisAI InferenceX README](../source-notes/semianalysisai-inferencex-readme.md) establishes InferenceX as a continuous benchmark standard and public reproducible baseline source.
- [Hazy Research: Look Ma, No Bubbles](../source-notes/hazyresearch-look-ma-no-bubbles-megakernel.md) defines the megakernel concept and the profiler evidence needed before attempting one.
- [DeepSeek V4 Preview Release](../source-notes/deepseek-v4-preview-release.md) establishes official API and open-weight availability for V4 Flash.
- [DeepSeek V4 Flash Hugging Face Model Card](../source-notes/deepseek-v4-huggingface-model-card.md) provides the open-weight model identity, checkpoint precision, encoding caveat, and local sampling recommendations.
- [DeepSeek V4 Technical Report And Model Card](../source-notes/deepseek-v4-technical-report-and-model-card.md) provides the architecture and systems facts used to prioritize kernel hypotheses.
- [vLLM DeepSeek V4 Support](../source-notes/vllm-deepseek-v4-support.md) is a candidate strongest public reproducible baseline path.
- [SGLang DeepSeek V4 Cookbook](../source-notes/sglang-deepseek-v4-cookbook.md) is a candidate strongest public reproducible baseline path.
- [TensorRT-LLM DeepSeek Support](../source-notes/tensorrt-llm-deepseek-support.md) is a conditional baseline candidate, but not a required first-run baseline without exact V4 Flash support evidence.
- [OpenAI MRCR](../source-notes/openai-mrcr.md) provides the explicit 1M-context retrieval/disambiguation quality gate.
- [LongBench v2](../source-notes/longbench-v2.md) provides the realistic long-context reasoning quality gate.
- [V4 Flash kernel map](../artifacts/v4-flash-kernel-map.md) is the current top-down architecture-map artifact.

## Open Questions

- Between vLLM and SGLang, which runtime path is the strongest public reproducible baseline for V4 Flash on the target Blackwell topology?
- Has TensorRT-LLM published a reproducible V4 Flash path by benchmark time?
- Which exact vLLM and SGLang image digests or source SHAs should be used for the first benchmark?
- When should the agentic prefix-cache workload expand from 128K total ISL to 512K and 1M variants?
- Is 3 repetitions per benchmark point sufficient once hardware cost and runtime are known?
- Which external reference scores from the DeepSeek V4 report and public benchmark leaderboards are comparable enough to list in the first result record?
