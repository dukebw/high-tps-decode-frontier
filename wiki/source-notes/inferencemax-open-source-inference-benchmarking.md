# InferenceMAX: Open Source Inference Benchmarking

Source: `../../sources/html/inferencemax-open-source-inference-benchmarking.html`

Upstream: https://inferencex.semianalysis.com/blog/inferencemax-open-source-inference-benchmarking

Status: summarized from imported HTML snapshot on 2026-05-22. The source was scanned for obvious credential patterns before import.

## Summary

This article lays out the InferenceMAX/InferenceX benchmark methodology and the core serving tradeoff between throughput and interactivity. It defines throughput as tokens/s/GPU and interactivity as tokens/s/user, then constructs a Pareto frontier from points where no other configuration is better on both axes.

The article is the best current methodology source for this repo's serving-benchmark convention. It describes model/runtime/hardware configurations, max-concurrency sweeps, input/output length scenarios, random request generation to avoid prefix-cache effects, and fairness issues such as warmup handling.

## Key Claims

- LLM serving has a fundamental tradeoff between aggregate throughput and per-user interactivity.
- The Pareto frontier is the curve of configurations where no point improves both throughput and interactivity simultaneously.
- InferenceMAX uses an inference server plus benchmark client, with requests sent at infinite rate while sweeping max concurrent requests.
- The benchmark uses random sequences to avoid prefix caching until prefix-cache workload modeling is handled explicitly.
- The methodology uses three input/output scenarios: 1024/1024 for chat, 1024/8192 for reasoning, and 8192/1024 for summarization, with input length varied from 80% to 100% of the target.
- Benchmark configuration dimensions include model, precision, GPU, open-source framework, parallelism, and max concurrency.
- Warmup policy matters for fairness; the article describes a v1 decision to disallow warmup and lengthen DeepSeek runs after the ambiguity surfaced.

## Relevance To This Repo

This source directly supports the glossary decision that the Pareto frontier has tokens/s/user on the x-axis and tokens/s/GPU on the y-axis, with served concurrency swept to generate datapoints. It also supports using serving benchmarks as the top-level unit that motivates extracted kernel benchmarks.

The source gives this repo a concrete starting standard for workload scenarios, concurrency sweeps, max-concurrency controls, random data caveats, and benchmark metadata.

## Kernel And Benchmark Implications

- Serving benchmarks should report both tokens/s/user and tokens/s/GPU, not just one metric.
- Benchmark-driven investigations should state whether they use InferenceX-like random requests, real datasets, OpenCode-derived logs, or another workload source.
- Prefix caching must be either disabled/avoided or explicitly modeled, because it changes the serving bottleneck.
- Prefix-cache-enabled agent/coding traffic should be a separate workload regime with its own frontier, not mixed into the no-cache InferenceX frontier.
- Warmup policy must be explicit before comparing runtimes or kernels. For the first V4 Flash baseline, no-cache InferenceX runs disable warmup for comparability, while prefix-cache runs use explicit cache warmup to reach the target cached-prefix percentage.
- Kernel benchmarks should be traced back to a serving-benchmark point on the Pareto curve.

## Open Questions

- When should the agentic prefix-cache workload expand from the initial 128K total-ISL shape to 512K and 1M variants?
- How should this repo represent cost, TCO, and power metrics if the initial focus is tokens/s/user versus tokens/s/GPU?

## Follow-Ups

- Add fields to the benchmark template for frontier axes, served concurrency sweep, request-rate policy, prefix-cache policy, and warmup policy.
- Use the public InferenceX scenarios unchanged for the first DeepSeek-V4-Flash serving benchmark.
