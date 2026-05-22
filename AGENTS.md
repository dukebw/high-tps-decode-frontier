# Agent Instructions

This repository is a public benchmark-driven research codebase for high-TPS (high tokens/s/user and low TPOT) text LLM serving, systematic performance engineering, and kernel authoring practice.

## Purpose

Use the repo to improve the Pareto frontier for text LLM serving: tokens/s/user on the x-axis and tokens/s/GPU on the y-axis, with served concurrency swept to produce datapoints. Keep kernel work connected to end-to-end serving constraints such as TTFT, TPOT, throughput, latency distributions, memory pressure, quality, cost, and production complexity.

## Directory Contracts

- `CONTEXT.md` contains the canonical glossary. Update it when repo language is resolved.
- `sources/` contains immutable raw inputs. Do not edit imported source artifacts in place.
- `wiki/` contains synthesized knowledge, including source notes and benchmark-driven investigation narratives.
- `wiki/investigations/` contains the narrative home for benchmark-driven investigations.
- `wiki/source-notes/` contains structured notes derived from raw sources.
- `wiki/questions/` contains active research-question pages and hypotheses.
- `benchmarks/` contains benchmark conventions, templates, and future harness code.
- `kernels/` contains kernel-practice conventions, templates, and future implementations.
- `docs/` contains background and operating-pattern documents.

## Source Workflow

1. Review a source for obvious public-safety problems before committing it to this public repo.
2. Copy raw source artifacts into `sources/` unchanged.
3. Create or update a source note under `wiki/source-notes/`.
4. Link the source note from `wiki/index.md`.
5. Add a dated entry to `wiki/log.md`.

Exception: live Hugging Face model files may be referenced instead of copied only when benchmark records include the exact repository ID, immutable revision commit SHA, and file paths. Do not use branch names such as `main` as benchmark provenance.

Source notes should include source path, review status, concise summary, key claims, relevance to Blackwell kernels or decode throughput, open questions, and follow-up actions.

## Research Workflow

Investigation pages should state the model/workload/hardware target, strongest baseline, frontier axes, workload source, current hypotheses, evidence, benchmark links, kernel links, experiments to run, and unresolved decisions. Prefer claims that can be tied to a source, formula, benchmark, profiler artifact, or explicit assumption.

Do not claim a performance win without recording the benchmark context. Separate kernel-level speedups from end-to-end serving improvements.

## Benchmark Discipline

Every benchmark result must record:

- Hardware: GPU, topology, driver, CUDA, clock/power state when known.
- Software: runtime, library versions, git SHAs, build flags, environment details.
- Workload: model or proxy, prompt/output length distribution, served concurrency sweep, phase, dtype, shapes, prefix-cache policy.
- Metrics: TTFT, TPOT, tokens/s/user, tokens/s/GPU, p50/p95 latency, memory, utilization, profiler counters where available.
- Method: commands, request-rate policy, warmup policy, repetitions, variance, excluded runs, caveats.
- Correctness: serving benchmark correctness or kernel benchmark correctness as appropriate.
- Quality: every serving benchmark must include a predeclared formal eval suite, state whether it is report-only or pass/fail, and compare to external reference scores when relevant public references exist.
- Result provenance: local measured, reproduced public recipe, faithfully cited, or estimated limit.
- Source pinning: live Hugging Face files may be referenced instead of copied only when benchmark records include the exact repository ID, immutable revision commit SHA, and file paths.
- Baselines and limits: strongest baseline, speed-of-light hardware specs, and workload-derived roofline limits where applicable.
- Interpretation: bottleneck hypothesis, profiler evidence, what changed, whether the frontier moved, and next experiment.

Use `benchmarks/templates/result-template.md` for new result records.

## Kernel Practice Discipline

The first-class kernel stack is CUDA, CUTLASS, and CuTe DSL on datacenter Blackwell. Triton, Mojo, TileLang, and other systems can be compared later, but they should not drive the initial layout.

Each kernel practice item should include objective, serving benchmark link or drill scope, strongest baseline, correctness check, benchmark command, profiling notes, lessons learned, and next step. Megakernel investigations require profiler evidence of critical-path kernel-boundary bubbles, launch/teardown overhead, synchronization stalls, or memory-pipeline gaps before implementation work starts. Use `kernels/templates/practice-item-template.md` for new items.

## Public Repo Constraints

This repo is public. Avoid committing credentials, private customer data, unpublished internal details, large generated traces, or source artifacts that are not intended for publication.

## Maintenance Rules

When adding or changing research content, update `wiki/index.md` and `wiki/log.md` in the same change. Keep the smallest correct structure; do not add framework code before there is a concrete benchmark or kernel to support it.
