# Wiki Log

## [2026-05-21] setup | Repository skeleton

- Established `sources/`, `wiki/`, `benchmarks/`, `kernels/`, and `docs/` as the top-level research structure.
- Added `AGENTS.md` with repo-specific operating rules for sources, wiki maintenance, benchmarks, and kernel practice.
- Moved the general LLM wiki pattern document to `docs/llm-wiki-pattern.md`.

## [2026-05-21] ingest | Initial HTML ramp sources

- Imported two raw HTML source artifacts into `sources/html/` after a quick scan for obvious credentials, emails, and confidential markers.
- Created structured source notes for the DeepSeek V4 Flash model-performance ramp plan and the sparse video attention FA4 follow-on ramp.

## [2026-05-21] maintenance | Scrubbed company-specific wording

- Replaced company-specific ramp-plan wording in current `main` with neutral model-performance and production-serving language.
- Renamed the affected HTML source artifact and source note to neutral filenames.

## [2026-05-21] artifact | V4 Flash kernel map

- Added `wiki/artifacts/v4-flash-kernel-map.md` as the D1 architecture-map artifact for the top-down V4 Flash autopsy.
- Updated the ramp source to use the hyphenated artifact filename.

## [2026-05-22] ingest | InferenceX methodology sources

- Imported the `SemiAnalysisAI/InferenceX` README snapshot and InferenceMAX open-source benchmarking methodology article.
- Added structured source notes for the InferenceX benchmark standard and the tok/s/user versus tok/s/GPU Pareto frontier methodology.

## [2026-05-22] ingest | Megakernel source

- Imported Hazy Research's "Look Ma, No Bubbles!" article as the canonical source for the repo's megakernel terminology.
- Added a structured source note connecting megakernels to low-TPOT serving, profiler evidence, and correctness requirements.

## [2026-05-22] glossary | Benchmark-driven research codebase

- Added `CONTEXT.md` as the canonical glossary for repo language.
- Resolved the repo identity as a benchmark-driven research codebase with wiki narratives, benchmark code, and kernel code as linked subsystems.
- Updated README, agent instructions, benchmark templates, and kernel practice templates to reflect the InferenceX-style frontier, result provenance, correctness levels, strongest baselines, speed-of-light specs, roofline limits, and megakernel gate.

## [2026-05-22] investigation | V4 Flash InferenceX-style frontier

- Created `wiki/investigations/v4-flash-inferencex-frontier.md` as the planned investigation home for the first real V4 Flash serving-frontier benchmark.
- Marked the investigation as blocked on official model and runtime availability.

## [2026-05-22] adr | InferenceX-style serving frontier

- Added `docs/adr/0001-inferencex-style-serving-frontier.md` to record the benchmark-standard decision and its tradeoff against starting with OpenCode-derived logs or public datasets.

## [2026-05-22] ingest | Official DeepSeek V4 source set

- Imported DeepSeek's V4 release page, V4 Flash Hugging Face model card, V4 technical report PDF, and V4 model-card PDF.
- Imported public vLLM and SGLang DeepSeek V4 runtime recipe pages as candidate baseline sources.
- Added structured source notes for DeepSeek-owned model sources, vLLM support, and SGLang deployment guidance.
- Scanned the imported artifacts for obvious credential patterns before adding them to the public repo.

## [2026-05-22] artifact | V4 Flash official-source kernel map

- Expanded `wiki/artifacts/v4-flash-kernel-map.md` from a stub into an official-source architecture map with known facts, kernel hypotheses, unknowns, runtime baselines, and measurements to run first.
- Updated the V4 Flash InferenceX-style investigation now that official weights and public runtime recipes are available.
- Updated `CONTEXT.md` with runtime recipe, official-source kernel map, CSA, HCA, and refreshed V4 Flash target language.

## [2026-05-22] decision | V4 Flash serving quality gate

- Selected OpenAI MRCR at 128K, 512K, and 1M bins plus the LongBench v2 long subset as the first V4 Flash serving benchmark's predeclared quality gate.
- Selected both Non-think and Think High as required quality-gate reasoning modes.
- Selected MRCR 2-needle samples only for the first quality-gate run.
- Selected all 100 MRCR samples per selected bin for the first quality-gate run.
- Selected the entire LongBench v2 long subset for the first quality-gate run.
- Selected report-only quality gating for the first baseline, with external-reference sanity checks against the DeepSeek V4 report and public benchmark references where settings are comparable.
- Imported source artifacts for the MRCR dataset card, LongBench v2 project page, and LongBench v2 dataset card.
- Added source notes for MRCR and LongBench v2, then linked them from the V4 Flash InferenceX-style investigation.

## [2026-05-22] decision | V4 Flash baseline runtime scope

- Selected both vLLM and SGLang as required baseline candidates before declaring the strongest baseline.
- Selected MegaMoE as part of the runtime baseline sweep when it is publicly enableable and performs better than the runtime's other tested configs.
- Evaluated TensorRT-LLM as a possible third baseline and found strong DeepSeek-V3/V3.2, DSA, Blackwell, and benchmarking support, but no explicit public V4 Flash support row or deployment recipe in the imported sources.
- Recorded TensorRT-LLM as a conditional baseline candidate: include it in the first baseline run only if a public, reproducible V4 Flash path exists by benchmark time.
- Imported TensorRT-LLM source snapshots and added a structured source note. The generic README snapshot was excluded because it contained unrelated scrubbed wording; the retained sources are DeepSeek/model-support specific.

## [2026-05-22] decision | Hugging Face source pinning

- Decided to rely on live Hugging Face references for official model metadata, tokenizer, encoding, and inference files when benchmark records pin exact repo IDs, immutable revision commit SHAs, and file paths.
- Avoided importing additional Hugging Face metadata files into `sources/` unless a future decision specifically needs a local immutable snapshot.
- Deferred any Hugging Face revision auto-bump job until benchmark harness work starts.
- Added source-pinning requirements to `CONTEXT.md`, `AGENTS.md`, the benchmark result template, and the V4 Flash investigation.

## [2026-05-22] decision | First serving workload shape

- Selected the public InferenceX random request scenarios unchanged for the first V4 Flash baseline: 1024/1024 chat, 1024/8192 reasoning, and 8192/1024 summarization, with input length varied from 80% to 100% of target.
- Made agentic prefix-cache traffic a first-class workload regime rather than deferring all V4-Flash-specific variants. It should produce a separate frontier for high-ISL multi-turn coding/agent traffic with explicit cached-prefix percentage, measured hit rate, warmup policy, uncached delta tokens, and output length distribution.
- Selected the first agentic prefix-cache shape: 128K total ISL, 95% cached prefix, approximately 6.4K uncached delta tokens, and 1K OSL. Deferred 512K and 1M total-ISL variants until the harness is stable.
- Required both workload regimes before declaring the strongest baseline: no-cache public InferenceX scenarios and the first agentic prefix-cache scenario.
- Selected a fixed served-concurrency sweep for the first pass: 1, 2, 4, 8, 16, 32, 64, 128, and 256, run unchanged across runtimes, configs, and workload regimes.
- Selected closed-loop infinite-rate request policy for the first pass, with open-loop arrival-rate traffic deferred to later production/SLO realism work.
- Selected multiple repetitions for first-pass baseline points, defaulting to 3 repetitions per point unless hardware availability forces an explicit exception.
- Selected workload-specific warmup policy: no warmup for the no-cache public InferenceX regime, explicit cache warmup for the agentic prefix-cache regime before timing starts.
- Decided first-pass baseline points do not require profiler traces. Profiling will be collected manually during follow-up experimentation on selected points.
- Updated `CONTEXT.md`, the benchmark result template, the V4 Flash investigation, and the InferenceMAX source note.

## [2026-05-21] question | Blackwell kernel frontier

- Created the first research-question page focused on datacenter Blackwell kernels, CUDA, CUTLASS, CuTe DSL, and their connection to high-throughput decode metrics.
