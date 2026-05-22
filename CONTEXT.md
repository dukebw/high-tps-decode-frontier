# High-TPS Decode Kernel Research

This context defines the domain language for a benchmark-driven research codebase focused on high-TPS, low-TPOT LLM serving and kernel performance engineering.

## Language

**Research codebase**:
The repository identity: a code-and-knowledge workspace for benchmark-driven kernel investigations into high-TPS, low-TPOT LLM serving. The wiki is a subsystem of the research codebase, not the whole repo.
_Avoid_: Research notebook as the repo identity

**Benchmark-driven investigation**:
The primary unit of work in this repo: a bounded study that starts from a model/workload/hardware target and uses measurement to decide whether kernel or serving changes move the Pareto frontier.
_Avoid_: Research question as the primary unit of work, source ingestion as the primary unit of work, kernel project as the primary unit of work

**Investigation home**:
The narrative page for a benchmark-driven investigation, located under `wiki/investigations/`. It links to executable serving benchmarks under `benchmarks/` and reusable or custom kernels under `kernels/`.
_Avoid_: Making benchmark code or kernel code the narrative source of truth

**Serving benchmark**:
An end-to-end measurement of an LLM serving target for a specific model, workload, hardware, and runtime. A benchmark-driven investigation starts from at least one serving benchmark unless it is explicitly scoped as a kernel-only drill.
_Avoid_: Workload test, model benchmark

**Kernel benchmark**:
A measurement of one kernel or a small group of kernels extracted from, or motivated by, a serving benchmark. A kernel benchmark includes correctness criteria and exists to explain or improve a serving benchmark bottleneck.
_Avoid_: Microbenchmark when the kernel is not tied back to serving behavior

**Kernel group**:
A critical-path slice made of multiple kernels benchmarked and correctness-tested together because their interaction explains a serving bottleneck.
_Avoid_: Loose folder of related kernels

**Kernel benchmark correctness**:
Numerical agreement between a kernel or kernel group and a reference implementation under stated tolerances, shapes, dtypes, and edge cases.
_Avoid_: Performance-only kernel benchmark

**Megakernel**:
A fused critical-path kernel that treats the GPU like a small VM: work that would normally be split across many kernel launches is scheduled inside one long-running kernel to reduce launch, teardown, synchronization, and memory-pipeline bubbles.
_Avoid_: Any large kernel, generic kernel group

**Megakernel gate**:
The evidence required before starting a megakernel investigation: profiler evidence of critical-path kernel-boundary bubbles, launch/teardown overhead, synchronization stalls, or memory-pipeline gaps after simpler runtime, graph, fusion, or scheduling options are considered.
_Avoid_: Megakernel as a default first optimization

**Serving benchmark correctness**:
Evidence that the runtime is serving the intended model, tokenizer, configuration, and request protocol without feature regressions. This is distinct from the serving quality gate, which every serving benchmark also needs.
_Avoid_: Treating throughput numbers as valid without runtime/config validation

**Serving quality gate**:
A required formal evaluation suite for every serving benchmark, distinct from serving benchmark correctness. It establishes that the benchmarked serving path still produces acceptable model behavior for the chosen workload and model.
_Avoid_: Lightweight-only smoke checks as the sole quality gate for serving benchmarks

**Predeclared eval suite**:
The formal serving quality gate selected by an investigation before performance runs begin. It should include checks for both general model capability and the investigation's target use.
_Avoid_: Choosing quality checks after seeing performance results

**Report-only quality gate**:
A serving quality gate that must be run and reported before performance results are considered interpretable, but does not yet impose pass/fail thresholds. It is appropriate for a first baseline when local reference scores do not exist yet.
_Avoid_: Treating report-only as optional

**External reference sanity check**:
A comparison between local quality-gate scores and relevant published scores, such as an official model report or public benchmark leaderboard. It is used to catch setup mistakes and contextualize baseline quality, not to claim identical conditions unless the benchmark settings match.
_Avoid_: Treating external reference scores as local measured results

**Strongest baseline**:
The best credible existing runtime, library, kernel, or configuration path for the same model/workload/hardware shape that can be run locally or faithfully cited. New hypotheses and custom kernels are compared against this baseline, not against naive implementations.
_Avoid_: Naive baseline as the primary comparator, weak baseline

**Baseline path**:
An already available and credible runtime, configuration, library kernel, custom kernel, or megakernel for the target benchmark. Available means public and reproducible enough to run or faithfully cite: a released package, upstream commit, public recipe, or official benchmark artifact.
_Avoid_: Assuming baseline means no custom kernels

**Runtime recipe**:
A public launch/configuration path for a serving runtime, such as a vLLM or SGLang command recipe. A runtime recipe can be a baseline path when it is public and reproducible enough to run or faithfully cite.
_Avoid_: Treating recipes as benchmark results without running or provenance labels

**Intervention hypothesis**:
A new or experimental change being tested by this repo to move the frontier. It can be a runtime/config change, scheduler change, library integration, custom kernel, megakernel, or kernel group.
_Avoid_: Assuming intervention means custom kernel only

**Speed of light**:
The peak hardware specification relevant to a benchmark, such as peak tensor-core throughput, HBM bandwidth, NVLink bandwidth, or other published hardware limits. It is not workload-adjusted.
_Avoid_: Roofline limit, best observed baseline

**Roofline limit**:
A workload-derived performance ceiling based on the target shape, required FLOPs, required data movement, arithmetic intensity, and relevant speed-of-light hardware specs.
_Avoid_: Speed of light

**Result provenance**:
The evidentiary label attached to a benchmark number: local measured, reproduced public recipe, faithfully cited, or estimated limit.
_Avoid_: Mixing measured, cited, and estimated numbers without labels

**Pareto frontier**:
The tradeoff curve with tokens/s/user on the x-axis and tokens/s/GPU on the y-axis. Served concurrency is swept to produce the points on the curve under fixed model, workload, hardware, runtime, and quality constraints.
_Avoid_: Using Pareto frontier without naming both axes

**Continuous frontier**:
The preferred representation of high-interactivity goals in this repo. Instead of fixed tokens/s/user bands, plot and compare the full tokens/s/user versus tokens/s/GPU curve.
_Avoid_: Reducing the 300-1000+ tokens/s/user goal to one headline threshold

**Frontier win**:
A new non-dominated point or curve segment relative to the strongest baseline under the same model, workload, hardware, runtime constraints, correctness, and quality bar.
_Avoid_: Calling isolated kernel speedups or single-axis gains a frontier win before end-to-end validation

**Served concurrency**:
The number of active requests or users served simultaneously during a serving benchmark sweep. It is a control variable used to produce Pareto-frontier datapoints, not the frontier axis itself.
_Avoid_: Batch size when discussing user-level serving sweeps

**Fixed served-concurrency sweep**:
A predeclared concurrency grid run unchanged across comparable runtimes, configs, and workloads. The first V4 Flash baseline uses this instead of adaptive refinement to keep the methodology simple and unbiased.
_Avoid_: Adding runtime-specific points during the first pass

**Closed-loop infinite-rate policy**:
A serving benchmark request policy that keeps up to the target served concurrency in flight and immediately submits a replacement request when one finishes. It is the first-pass request policy because it matches InferenceX-style frontier construction.
_Avoid_: Treating closed-loop results as fixed-arrival production SLO results

**Open-loop arrival-rate policy**:
A serving benchmark request policy that sends requests according to an external arrival schedule, regardless of whether previous requests have finished. It is useful for production/SLO realism but is not the first-pass V4 Flash baseline policy.
_Avoid_: Mixing open-loop and closed-loop points on the same unlabeled frontier

**Benchmark repetition**:
A repeated run of the same benchmark point under the same model, runtime, config, workload, concurrency, and cache policy. Repetitions estimate variance and should be reported instead of collapsing to a single unqualified number.
_Avoid_: One-off numbers for baseline selection

**InferenceX-like random workload**:
The default first serving workload style: public InferenceX random request scenarios used unchanged, with served-concurrency sweeps, chosen for comparability and explicit prefix-cache control. The initial scenarios are 1024/1024 chat, 1024/8192 reasoning, and 8192/1024 summarization, with input length varied from 80% to 100% of target.
_Avoid_: Treating it as representative of all real user traffic

**Agentic prefix-cache workload**:
A first-class serving workload regime for multi-turn agent and coding traffic where the total input sequence length can be high but most input tokens are reused from a shared prefix cache. It should report total ISL, uncached delta tokens, target cached-prefix percentage, cache hit policy, output length distribution, and whether the cache is warm before timing starts.
_Avoid_: Mixing prefix-cache-enabled results with no-cache InferenceX results on the same frontier without labeling the workload regime

**Real V4 Flash target**:
The first model target for serving benchmarks. It uses official DeepSeek-V4-Flash weights, config, tokenizer, encoding rules, and an end-to-end runtime path rather than a proxy model or proxy shape workload.
_Avoid_: V4-Flash-like proxy as the first serving benchmark target

**Pinned Hugging Face reference**:
A live Hugging Face repository file referenced by exact repository ID, immutable revision commit SHA, and file path. It is acceptable benchmark source material without copying the file into `sources/` when the benchmark record captures those identifiers.
_Avoid_: Branch names such as `main` as benchmark provenance

**Official-source kernel map**:
A top-down architecture map built from public model and runtime sources before local profiling. It lists known facts, unknowns, and kernel hypotheses, but it is not evidence that any path is a bottleneck or optimization win.
_Avoid_: Treating source-backed architecture analysis as measured performance evidence

**Compressed Sparse Attention (CSA)**:
The V4 attention path that compresses KV entries along the sequence dimension and then applies sparse selection over compressed entries, with a sliding-window branch for local tokens.
_Avoid_: Generic sparse attention

**Heavily Compressed Attention (HCA)**:
The V4 attention path that applies heavier sequence-dimension KV compression and dense attention over the compressed entries, with a sliding-window branch for local tokens.
_Avoid_: Treating it as the same kernel shape as CSA

**Smallest official Blackwell topology**:
The first hardware target for a real V4 Flash serving benchmark: the smallest official B200, GB200, or related Blackwell topology that can serve the model without offload or workload distortion. Current public runtime recipes make 4-GPU Blackwell paths candidates, but the final target depends on accessible hardware and strongest-baseline selection.
_Avoid_: Choosing rack-scale hardware before it is needed for correctness or model fit

## Example Dialogue

Developer: Should this new source become a benchmark-driven investigation?
Domain expert: Not yet. First extract the model/workload/hardware target and the strongest baseline. It becomes a benchmark-driven investigation once there is something measurable to compare.

Developer: Where do I explain why this benchmark exists?
Domain expert: In the investigation home. The benchmark and kernel directories hold executable artifacts, but the wiki page explains the target, baseline, hypotheses, results, and next steps.

Developer: Is this attention timing a serving benchmark?
Domain expert: No. It is a kernel benchmark unless it measures the end-to-end serving target. It can still be the right next step if it explains the serving benchmark bottleneck.

Developer: Router plus dispatch plus grouped GEMM plus combine all show up in the bottleneck. Is that one kernel benchmark?
Domain expert: Yes, if the interaction is the point. Treat it as a kernel group benchmark.

Developer: Does a serving benchmark need tensor-level equality checks?
Domain expert: Usually no. Tensor-level checks belong to kernel benchmark correctness. Serving benchmark correctness validates the model/config/runtime path and adds quality gates when output-changing changes are tested.

Developer: Can a pure runtime benchmark skip model quality checks?
Domain expert: No. Every serving benchmark needs a serving quality gate, even if output-changing risk is low.

Developer: Can we choose evals after the benchmark looks fast?
Domain expert: No. The eval suite is predeclared by the investigation before performance runs begin.

Developer: Is this collection of separate CUDA kernels a megakernel?
Domain expert: No. A megakernel deliberately moves scheduling and synchronization inside one long-running critical-path kernel to reduce bubbles.

Developer: Should I start a megakernel because it sounds interesting?
Domain expert: Not as a benchmark-driven investigation. First show profiler evidence that the megakernel gate is met.

Developer: Can I compare my kernel only to a naive PyTorch implementation?
Domain expert: No. A naive implementation can be useful for correctness, but the performance claim needs the strongest baseline for the target shape.

Developer: DeepSeek ships a new MoE megakernel. Is that a hypothesis because it is custom kernel work?
Domain expert: No. If it is already available and credible for the target benchmark, it is a baseline path. Novel work in this repo is an intervention hypothesis.

Developer: Is this kernel at speed of light?
Domain expert: Say whether it is close to the roofline limit. Speed of light is the hardware peak used to derive that limit.

Developer: Can this InferenceX number sit beside my local run?
Domain expert: Yes, if provenance is explicit. Label the InferenceX number as faithfully cited or reproduced from public recipe, not local measured.

Developer: Should we plot concurrency on the Pareto frontier?
Domain expert: No. Sweep served concurrency to generate points, then plot tokens/s/user against tokens/s/GPU.

Developer: Should we report only the 300 tokens/s/user point?
Domain expert: No. Plot the continuous frontier; the high-interactivity region matters, but a single threshold hides tradeoffs.

Developer: My kernel improved one profiler counter. Is that a frontier win?
Domain expert: Not yet. It is evidence for an investigation, but a frontier win requires a new non-dominated serving point or curve segment.

Developer: Should the first benchmark use OpenCode logs?
Domain expert: No. Start with an InferenceX-like random workload for comparability, then add OpenCode-derived traffic as a separate realism track.

Developer: Can we run a proxy while waiting for V4 Flash?
Domain expert: Not as the first serving benchmark. Proxy work can be a kernel-only drill or preparatory analysis, but the first serving benchmark targets real V4 Flash.

Developer: Should the first benchmark use the biggest available rack?
Domain expert: Not by default. Start with the smallest official Blackwell topology that can serve the real target without distorting it.
