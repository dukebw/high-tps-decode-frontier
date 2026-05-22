# InferenceX-Style Serving Frontier

Status: accepted

This repo uses an InferenceX-style continuous serving frontier as its default benchmark standard: tokens/s/user on the x-axis, tokens/s/GPU on the y-axis, with served concurrency swept to generate datapoints. We chose this over OpenCode-derived logs or public datasets as the first workload source because it gives a comparable, reproducible baseline methodology with explicit prefix-cache control; realistic workload traces can be added later as separate benchmark-driven investigations.

## Consequences

- Serving benchmark records must describe frontier construction, served-concurrency sweeps, request-rate policy, prefix-cache policy, and result provenance.
- Kernel benchmarks should explain which serving-frontier bottleneck they support; isolated kernel speedups are not frontier wins until they create a new non-dominated serving point or curve segment.
