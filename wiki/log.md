# Wiki Log

## [2026-05-31] ingest | Simon Veitner BearBlog CUDA/CuTeDSL posts

- Imported the complete post set exposed by `https://veitner.bearblog.dev/blog/` as raw HTML snapshots: 65 post pages plus the blog index under `sources/html/veitner-bearblog/`.
- Checked the blog index against the upstream sitemap before import; both exposed the same 65 post URLs.
- Added a collection source note covering the CUDA, CUTLASS/CuTe, CuTeDSL, QuACK, Hopper, Blackwell/B200, NVFP4, block-scaled GEMM, and Gated Delta Net themes.
- Reviewed all 65 imported post snapshots against the collection source note and added a post-level one-sentence summary table so every ingested post is substantively represented, not only listed in the manifest.
- Scanned the imported HTML snapshots for obvious private-key, cloud/API-token, and credential-marker patterns; the only credential-like matches were public SGLang example snippets using `openai_api_key = "EMPTY"`.

## [2026-05-28] ingest | Dao-AILab QuACK CuTe kernels

- Imported the `Dao-AILab/quack` README as a GitHub snapshot, pinned to main commit `34cfe42fe994dd961e42bbf179539b16d896aab4` and README blob `764f136b01f16d4ea469ea57db13be7550bc739e` (Apache-2.0).
- Added a structured source note covering QuACK as a CuTe DSL kernel library on the repo's first-class stack: near-speed-of-light memory-bound kernels (RMSNorm/softmax/cross entropy, 89.7% of peak HBM3 on H100), Blackwell `sm100`/`sm120` GEMM with block-scaled (MX) epilogues, and a full kernel inventory derived from the repo tree (it lists far more than the README advertises).
- Recorded the key distinctions: QuACK has no attention kernels (complementary to FlashAttention), is a kernel library rather than a serving runtime, and its published speed-of-light numbers are H100, not Blackwell.
- Updated the Blackwell kernel frontier question with QuACK as first concrete CuTe DSL evidence, a compute-bound vs memory-bound decode-kernel hypothesis, and a Blackwell roofline experiment to define.
- Scanned the imported README and the linked memory-bound speed-of-light blogpost for obvious private-key, cloud/API-token, and credential-marker patterns before commit; no secrets were found.

## [2026-05-28] synthesis | Speculative decode robustness map

- Imported primary source snapshots for the vLLM/EAGLE/TorchSpec EAGLE 3.1 blog and the z-lab DFlash README.
- Added structured source notes for EAGLE 3.1 and DFlash, preserving the distinction between DFlash as a block-diffusion speculative-decoding algorithm and DeepSeek V4 Flash as the model target.
- Added a speculative decode robustness map connecting CoCo corrupted-context recovery, TokenSpeed/MTP rollback/state handling, EAGLE 3.1 attention-drift robustness, and DFlash parallel drafting to future serving correctness and quality-gate requirements.
- Scanned the imported source snapshots for obvious private-key, cloud/API-token, and credential-marker patterns before adding them to the public repo; no secrets were found.

## [2026-05-28] ingest | CoCo augmentation X thread screenshots

- Added a user-provided screenshot transcript of Fern's X thread on Continuous Coding augmentation and the SolidGoldMagikarp stability problem.
- Added a structured source note connecting the thread to future corrupted-context, rare-token, and cached-prefix robustness probes for agentic serving quality gates.
- Recorded ingestion limitations: X exposed only the root post through public embed endpoints, so the full-thread artifact is screenshot-derived rather than an authenticated X export.

## [2026-05-28] ingest | PyTorch TokenSpeed Qwen3.5 agentic workload source

- Imported PyTorch's TokenSpeed Qwen3.5 agentic-workload article as a raw HTML snapshot.
- Added a structured source note covering TokenSpeed's hybrid prefix cache, Mamba/GDN state transfer, runtime/fusion optimizations, and reported B200 agentic/long-context benchmark claims.
- Scanned the imported HTML for obvious private-key and cloud/API-token patterns; the only credential-like matches were public upstream New Relic browser-monitoring identifiers embedded in the PyTorch page.

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

## [2026-05-22] plan | V4 Flash baseline benchmark

- Added `benchmarks/plans/v4-flash-baseline-plan.md` as the concrete first-pass run matrix for vLLM and SGLang baseline selection.
- Pinned `deepseek-ai/DeepSeek-V4-Flash` to Hugging Face revision `6976c7ff1b30a1b2cb7805021b8ba4684041f136` and listed required model, tokenizer, encoding, inference, index, and weight-shard paths to record in results.
- Imported the raw vLLM recipe YAML and updated the vLLM source note with the default `single_node_tep` strategy and Blackwell override details.
- Resolved candidate amd64 container image digests on `b200-aws2`: `docker.io/vllm/vllm-openai@sha256:4ac9b7c6dabc3ec762c0edef4e9245abe98373844da91cc53ee42e5c58280c5b` and `docker.io/lmsysorg/sglang@sha256:015f39a45844be5a7b35270c56dc4d9ebcfe9b0c21a3b4f877a4ee22e795bd7a`.
- Pulled both pinned images on `b200-aws2`, verified vLLM `0.21.0` and SGLang `0.5.12`, and recorded build commits/image tags in the benchmark plan.
- Generated the NVIDIA CDI spec on `b200-aws2`; Podman-backed containers need `--device nvidia.com/gpu=all` rather than `--gpus all` for CUDA visibility.
- Completed the pinned V4 Flash Hugging Face snapshot download on `b200-aws2`; remote logs are mirrored under `~/shared/b200-aws2/logs/v4-flash-download/20260522T1800Z/` and the HF cache was about `844G`.
- Added `benchmarks/scripts/v4-flash-vllm-smoke.sh` to reproduce the pinned vLLM launch smoke test.
- Completed a vLLM smoke run on `b200-aws2` with 4xB200, pinned vLLM `0.21.0`, pinned V4 Flash revision `6976c7ff1b30a1b2cb7805021b8ba4684041f136`, FP8 KV cache, DeepGEMM MegaMoE, and MXFP4 indexer cache; Non-think and Think High test requests both returned `323`.
- Verified the committed smoke script on `b200-aws2` and fixed its readiness loop to monitor the background `docker run` process instead of requiring the Podman container name to appear immediately in `docker ps`.
- Defined runtime config variants, workload scenarios, fixed concurrency sweep, repetition count, report-only quality matrix, serving correctness checks, and open setup items.

## [2026-05-22] ingest | TileRT GLM-5.1 decoding source

- Imported TileRT's English and Chinese "Speed as the Next Scaling Law" blog snapshots.
- Added a structured source note connecting TileRT's persistent Engine Kernel, tile-level scheduling, heterogeneous GPU workers, and GLM-5.1 production-serving claims to the repo's low-TPOT decode and megakernel investigation criteria.
- Scanned the imported HTML artifacts for obvious credential, private-key, and scrubbed-company markers before adding them to the public repo.

## [2026-05-22] ingest | FlashAttention paper series

- Imported PDF snapshots for FlashAttention 1, 2, 3, and 4.
- Added a structured source note connecting the series to the D4 dense attention baseline and D5 sparse/compressed attention toy benchmark path.
- Updated `CONTEXT.md` with D4 and D5 glossary entries from the imported model-performance ramp language.
- Scanned the imported PDFs with binary string search for obvious credential, private-key, and scrubbed-company markers before adding them to the public repo.

## [2026-05-21] question | Blackwell kernel frontier

- Created the first research-question page focused on datacenter Blackwell kernels, CUDA, CUTLASS, CuTe DSL, and their connection to high-throughput decode metrics.
