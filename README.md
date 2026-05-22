# High TPS Decode Frontier

This is a benchmark-driven research codebase for improving the Pareto frontier of text LLM serving: tokens/s/user on the x-axis and tokens/s/GPU on the y-axis.

The repo exists to study real model/workload/hardware targets, build serving and kernel benchmarks from that analysis, and practice writing performance-critical CUDA, CUTLASS, CuTe DSL, and megakernel-style code when profiler evidence justifies it.

## Operating Loop

1. Start from a model/workload/hardware target.
2. Establish the strongest public reproducible baseline path.
3. Sweep served concurrency to plot tokens/s/user versus tokens/s/GPU.
4. Profile the serving path and extract kernel benchmarks where they explain bottlenecks.
5. Compare against speed-of-light hardware specs and workload-derived roofline limits.
6. Test interventions only after correctness, provenance, and benchmark caveats are explicit.

## Layout

```text
CONTEXT.md    Canonical glossary for repo language.
sources/      Immutable raw inputs, including imported HTML and README source artifacts.
wiki/         Synthesized source notes, investigation narratives, indexes, and logs.
benchmarks/   Serving and kernel benchmark harnesses, templates, and result records.
kernels/      Kernels and kernel groups called from benchmarks.
docs/         Background documents, including the LLM-maintained wiki pattern.
```

## Benchmark Standard

- Use InferenceX-style continuous frontiers by default: tokens/s/user versus tokens/s/GPU, with served concurrency swept to generate datapoints.
- Use InferenceX-like random workloads for the first serving benchmark unless a benchmark-driven investigation explicitly chooses another workload source.
- Label result provenance as local measured, reproduced public recipe, faithfully cited, or estimated limit.
- Treat kernel speedups as evidence, not frontier wins, until they create a new non-dominated serving point or curve segment.
