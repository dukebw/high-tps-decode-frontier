# High TPS Decode Frontier

This is a working research notebook for improving the Pareto frontier of high-tokens/s/user vs. throughput for text LLM serving.

The repo also exists as a practice space for systematic performance engineering and kernel authoring.

## Operating Loop

1. Start from a concrete frontier serving question.
2. Collect raw sources without modifying them.
3. Write structured source notes and research-question pages.
4. Predict bottlenecks before profiling.
5. Build the smallest benchmark or kernel exercise that can test the prediction.
6. Record hardware, software, workload, metrics, profiler evidence, caveats, and next experiments.

## Layout

```text
sources/      Immutable raw inputs, including imported HTML source artifacts.
wiki/         Synthesized research questions, source notes, indexes, and logs.
benchmarks/   Benchmark conventions, templates, and future measurement harnesses.
kernels/      Kernel-practice conventions, templates, and future CUDA/CUTLASS/CuTe work.
docs/         Background documents, including the LLM-maintained wiki pattern.
```

## Current Focus

- First research question: [Blackwell kernel frontier](wiki/questions/blackwell-kernel-frontier.md).
- First kernel stack: CUDA, CUTLASS, and CuTe DSL.
- First hardware assumption: datacenter Blackwell, especially B200/GB200-class systems.
- First imported sources: two local HTML ramp documents under `sources/html/`, summarized in `wiki/source-notes/`.
