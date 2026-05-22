# SemiAnalysisAI InferenceX README

Source: `../../sources/github/semianalysisai-inferencex-readme.md`

Upstream: https://github.com/SemiAnalysisAI/InferenceX

Status: summarized from imported README snapshot on 2026-05-22. The source was scanned for obvious credential patterns before import.

## Summary

The InferenceX README describes InferenceX as an open-source continuous inference benchmark and research platform for tracking real LLM inference performance across open-source frameworks, models, and accelerator platforms. It frames inference performance as a moving target driven by both hardware releases and rapid software improvement in runtimes, kernels, distributed strategies, and schedulers.

The README is important for this repo because it establishes InferenceX as an external benchmark standard for comparing inference stacks and for tracking changes in the performance Pareto frontier over time.

## Key Claims

- InferenceX continuously benchmarks popular open-source inference frameworks and models to track real performance as software stacks evolve.
- Software improvements in SGLang, vLLM, TensorRT-LLM, CUDA, ROCm, kernels, distributed inference, and scheduling can move the Pareto frontier in increments that arrive days apart.
- Fixed-point benchmarks become stale quickly when the inference software ecosystem is changing rapidly.
- The official results are in `SemiAnalysisAI/InferenceX`; other forks or repos should be treated as unofficial unless explicitly labeled.

## Relevance To This Repo

This source supports using InferenceX as a methodology reference for serving benchmarks, frontier plots, and strongest-baseline discovery. It also reinforces that this repo should separate benchmarking the current best existing stack from experimenting with new kernels or megakernels.

## Open Questions

- Which parts of the InferenceX runner design are useful for this repo, and which are too heavyweight for a personal research codebase?
- Should this repo mirror any InferenceX benchmark configuration formats or only adopt the metric conventions?
- How should this repo mark results that are comparable to InferenceX versus local exploratory results?

## Follow-Ups

- Read the InferenceX benchmark directories before designing the first serving benchmark harness.
- Link this source from any benchmark template fields that mention official versus local results.
