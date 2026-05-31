# Simon Veitner BearBlog CUDA/CuTeDSL Posts

Source index: `../../sources/html/veitner-bearblog/blog-index.html`

Source posts: `../../sources/html/veitner-bearblog/*.html`

Upstream index: https://veitner.bearblog.dev/blog/

Status: imported as 65 raw HTML post snapshots plus the blog-index HTML snapshot on 2026-05-31. The post set was discovered from the blog index and checked against the upstream sitemap; both exposed the same 65 post URLs. The imported HTML snapshots were scanned for obvious private-key, cloud/API-token, and credential-marker patterns. The only credential-like matches were public SGLang example snippets using `openai_api_key = "EMPTY"` and passing that placeholder as `api_key`.

## Summary

This source collection captures Simon Veitner's BearBlog posts on CUDA, CUTLASS, CuTe, CuTeDSL, Mojo GPU programming, QuACK, Hopper persistent kernels, and Blackwell/B200 block-scaled tensor-core programming. The most relevant subset for this repo is the CuTeDSL and Blackwell sequence covering layout algebra, swizzling, TMA, WGMMA, persistent scheduling, warp specialization, B200 block-scaled GEMM, NVFP4 GEMV, Gated Delta Net kernels, and `tcgen05` shared-memory descriptor details such as SBO and LBO.

The collection also includes general GPU-kernel practice posts on indexing, predication, reductions, prefix sums, RMSNorm, matrix transpose, L2 persistence, CUDA streams, bit manipulation, Thrust, and Mojo. Several math or data-structure posts are less directly relevant to high-TPS decode, but they were imported because they are part of the complete blog-post set exposed by the requested index.

These are source snapshots, not benchmark records. Any performance claims in the posts should be treated as upstream reported context until a repo benchmark record captures hardware, software, workload, correctness, quality, method, provenance, and interpretation.

## Key Claims And Themes

- CuTe and CuTeDSL layout work is a recurring foundation: layout algebra, hierarchical layouts, swizzles, partitions, tensor slicing, thread-value layouts, MMA atoms, and categorical composition are presented as tools for reasoning about kernel data movement.
- Hopper-oriented CuTeDSL posts explain SGEMM, WGMMA, TMA, pipelining, consumer-producer staging, epilogues, persistent GEMM, and persistent float8 dense GEMM.
- Blackwell/B200 posts focus on `tcgen05`, block-scaled GEMM, NVFP4 GEMV, 2-CTA GEMM, warp specialization, Blackwell pipelining, persistent tile scheduling, scale-tensor construction, numeric conversions, and shared-memory descriptor offsets.
- QuACK-related posts provide practical examples for CuTeDSL kernels, including PingPong scheduling and outperforming compiled PyTorch for selected kernels.
- Gated Delta Net posts cover decoding, chunkwise recurrence, and prefill math, which are relevant analogues for non-standard recurrent or hybrid attention state in serving runtimes.
- General CUDA posts cover indexing, predication, TMA, assembly inspection, L2 cache persistence, CUDA streams, reductions, prefix sums, RMSNorm, matrix transpose, and bit-level implementation patterns.
- Mojo posts show GPU programming paths outside the repo's first-class CUDA/CUTLASS/CuTe stack, useful as comparison material but not as the initial kernel stack.

## Relevance To This Repo

The collection is directly relevant to the repo's Blackwell kernel frontier because it provides practical, source-level context for the same first-class stack: CUDA, CUTLASS/CuTe, CuTeDSL, B200, TMA, WGMMA, `tcgen05`, NVFP4, and block-scaled GEMM. It can support kernel-practice drills and explainers before local benchmark work begins.

The B200 block-scaled GEMM, NVFP4 GEMV, tile scheduling, warp-specialization, and SBO/LBO posts are especially relevant to DeepSeek-style low-TPOT serving because decode bottlenecks can involve small-batch GEMV/GEMM, quantized scale handling, persistent scheduling, and careful shared-memory layouts. The Gated Delta Net posts are relevant as a concrete non-KV recurrent-state workload, but they should not be treated as direct evidence for DeepSeek V4 Flash CSA/HCA behavior.

## Kernel And Benchmark Implications

- Use these posts as implementation background for kernel-practice items, not as proof that a frontier point moved.
- Before using any upstream timing claim, record whether it is locally measured, faithfully cited, or only background context.
- Promote individual posts to dedicated source notes if they become central to a specific benchmark, kernel-practice item, or investigation decision.
- For B200 block-scaled GEMM work, preserve distinctions between scale-tensor construction, host-side launch/setup, kernel body, descriptor construction, and scheduling policy.
- For persistent-kernel ideas, keep the repo's megakernel gate intact: require profiler evidence of launch gaps, synchronization stalls, memory-pipeline bubbles, or communication gaps before starting large persistent/megakernel interventions.
- For Gated Delta Net material, use the posts to design state-management and correctness questions, not as a substitute for the repo's V4 Flash serving quality gate.

## Post-Level Summaries

| Date | Title | Summary |
| --- | --- | --- |
| 2026-05-15 | Tile Scheduling in CuTeDSL | Explains `StaticPersistentTileScheduler` for persistent CuTeDSL kernels, including flat work indices, tile coordinates, waves, swizzling, and clustered CTA scheduling. |
| 2026-04-18 | SBO and LBO explained visually | Visually derives shared-memory descriptor SBO and LBO values for B200 `tcgen05` tensor-core operands under K-major and M/N-major layouts using swizzle atoms. |
| 2026-03-14 | Simple math to speed up GDN prefill | Derives a matrix identity that saves one inverse in chunkwise Gated Delta Net prefill and reports an upstream Torch speedup for the optimized form. |
| 2026-03-10 | Chunkwise Gated Delta Rule | Derives GPU-friendly chunkwise matrix forms for linear attention, the Delta Rule, and Gated Delta Net state/output updates. |
| 2026-02-15 | Gated Delta Net Decoding | Explains the Gated Delta Net decoding recurrence, including gates, beta factors, state updates, output computation, and the FlashInfer competition context. |
| 2026-02-07 | Grouped Blockscaled Gemm - Kernel | Analyzes the CuTeDSL persistent grouped block-scaled GEMM kernel, especially per-group tensor maps, tensor-map management, and scheduler-to-group mapping. |
| 2026-01-24 | Grouped Blockscaled Gemm - Host code | Walks through host-side setup for grouped block-scaled GEMM, including grouped pointers/scales, TMA/MMA setup, scheduler grid, and SMEM tensor-map buffers. |
| 2026-01-22 | Grouped Block scaled Gemm - Intro | Introduces grouped block-scaled GEMM as multiple block-scaled GEMMs with distinct shapes, scale factors, pointer arrays, and tensor-map setup requirements. |
| 2026-01-07 | Warp Specialisation in CuTeDSL | Shows how to split Blackwell GEMM TMA and MMA work across specialized warps with CuTeDSL pipelines, with upstream results that improve some 1-CTA cases but not the tested 2-CTA case. |
| 2026-01-04 | 2 CTA GEMM on B200 | Explains converting a 1-CTA Blackwell GEMM into a 2-CTA UMMA GEMM by changing instruction shape, cluster layout, barriers, pipeline cleanup, and launch structure. |
| 2025-12-23 | Blackwell Pipelining with CuTeDSL | Explains Blackwell CuTeDSL pipeline synchronization for overlapping TMA loads, MMA, accumulator transfer, and optional staged stores. |
| 2025-12-06 | B200 Blockscaled GEMM - The setup | Analyzes CuTeDSL block-scaled GEMM setup: layout calculation, MMA/TMA construction, scale-factor tiling, cluster layout, and epilogue tiling. |
| 2025-12-03 | Scale Tensor construction in CuTeDSL | Explains how NVFP4 scale tensors get CuTe layouts and relates scale-factor layout construction to swizzle-atom-style covering. |
| 2025-11-23 | Demystifying numeric conversions in CuTeDSL | Explains packed CuTeDSL numeric conversions through PTX `cvt` instructions and implements an FP8-to-FP16 conversion used to improve GEMV performance upstream. |
| 2025-11-16 | NVFP4 GEMV improved | Explores K-dimension parallelization strategies for NVFP4 GEMV using extra blocks, atomic adds, shared-memory reductions, and tuning tradeoffs. |
| 2025-11-13 | NVFP4 GEMV | Onboards readers to a GPU Mode/NVIDIA Blackwell NVFP4 GEMV CuTeDSL reference kernel, including tensor shapes, scale factors, layouts, conversion, accumulation, and store. |
| 2025-11-06 | Bit counting and geometric series | Connects a geometric-series proof to the `x & (x - 1)` bit trick for removing the rightmost set bit, contrasting naive loops with `popcnt`. |
| 2025-10-27 | Simple reduction in CuTeDSL | Implements an RMSNorm-style CuTeDSL reduction kernel, covering row-wise reduction, shared memory, correctness checks against PyTorch, and QuACK-inspired memory-bound design. |
| 2025-10-20 | MMA Atoms in CuTe | Explains CuTe MMA atoms through visualizations and the SM70 `8x8x4` F32/F16 atom, mapping register fragments and thread/value participation. |
| 2025-10-11 | LRU and LFU in C++ | Walks through C++ LRU and LFU cache implementations using lists/maps and eviction-policy mechanics for a slow page-access problem. |
| 2025-10-03 | Mutual Refinement and Composition | Applies Colfax's categorical CuTe layout treatment to compute layout composition via mutual refinement of nested tuple domains and codomains. |
| 2025-09-27 | Applied introduction to Categorical treatment of CuTe | Provides worked companion calculations for Colfax's categorical CuTe layout paper, encoding layouts as tuple morphisms and using visual reasoning. |
| 2025-09-23 | Layout Gymnastics | Works through CuTe layout-algebra exercises by hand, including coalesce, squeeze, sort, complement, compactness, and layout transformations. |
| 2025-09-20 | Swizzles and their usage in CuTeDSL Kernels | Shows how CuTeDSL applies swizzles through composed layouts and then uses them in shared-memory GEMM-like kernels to avoid bank conflicts. |
| 2025-09-14 | CuTe partitions | Explains CuTe inner, outer, and thread-value partitioning by implementing copy kernels for each tiling mode. |
| 2025-09-09 | Tensors Slicing in CuTe | Explains CuTe tensor slicing as an engine-plus-layout transformation, deriving offsets and sliced layouts for 2D and 3D tensors. |
| 2025-09-07 | Understanding CuTe Swizzling - The Math Behind 32B, 64B, and 128B Patterns | Derives the CUTLASS/CuTe swizzle bit-mask math behind no/32B/64B/128B swizzle patterns for shared-memory bank-conflict avoidance. |
| 2025-09-05 | GPU L2 Cache Persistence | Demonstrates NVIDIA persistent L2 cache setup with access-policy windows and repeated-data benchmarks on H100. |
| 2025-08-29 | Cuda streams | Explains CUDA streams by overlapping host-device copies and kernel execution with pinned memory, async memcpy, offsets, events, and NSYS traces. |
| 2025-08-23 | PingPong in the CuTeDSL with QuACK | Analyzes QuACK's persistent PingPong GEMM pattern where consumer warpgroups alternate MMA and epilogue work with separate load warps and explicit register-allocation tradeoffs. |
| 2025-08-17 | Bit Hacking in C | Introduces C bitwise operators and implements classic bit-hacking helpers for extracting, setting, inverting, and rotating bit fields. |
| 2025-08-13 | Intuition behind Hierarchical Layouts | Builds visual intuition for nested CuTe layouts by mapping linear indices to hierarchical coordinates and interpreting layouts as grids and cubes. |
| 2025-08-09 | Persistent Float8 Dense Gemm on Hopper | Modifies CuTeDSL dense GEMM for Hopper into a persistent batched Float8 GEMM using tile scheduling, producer/consumer warpgroups, TMA, and flexible epilogues. |
| 2025-08-04 | Epilogue in CuTeDSL H100 kernels | Walks through the H100 CuTeDSL GEMM epilogue, accumulator layouts, `stmatrix`, RMEM-to-SMEM-to-GMEM staging, and TMA store overlap. |
| 2025-07-30 | Let the compiler do the work in CuTeDSL | Explains CuTeDSL's experimental `cutlass.range(..., pipeline=...)` compiler-assisted prefetching and contrasts it with manual and naive GEMM mainloops. |
| 2025-07-25 | Persistent GEMM in CuTeDSL on Hopper | Builds a warp-specialized persistent Hopper GEMM using CuTeDSL `TileScheduler`, TMA, producer/consumer warp groups, and upstream-reported high TFLOP results. |
| 2025-07-20 | Consumer-Producer pattern on H100 in CuTeDSL | Explains Hopper `mbarrier` full/empty barriers, phase bits, and producer-consumer synchronization through QuACK RMSNorm backward reduction code. |
| 2025-07-13 | Backprob through Layernorm | Derives LayerNorm backward gradients step by step and ties the final formula to an `llm.c`-style reference implementation. |
| 2025-07-13 | Backprop through RMSNorm | Derives RMSNorm backward gradients and verifies the simplified formula against QuACK's reference implementation. |
| 2025-07-12 | Outperform compiled PyTorch code using QuACK | Gives a hands-on QuACK/CuTeDSL reduction-kernel guide, starting from RMSNorm and modifying it into LayerNorm to outperform compiled PyTorch for that case upstream. |
| 2025-07-05 | CuTeDSL on Hopper - Pipelining | Analyzes the Hopper `dense_gemm.py` CuTeDSL kernel mainloop, including TMA descriptor prefetch, staged SMEM layouts, `PipelineTmaAsync`, WGMMA, and producer/consumer state. |
| 2025-07-03 | CuTeDSL on Hopper - WGMMA and TMA intro | Introduces Hopper WGMMA and TMA setup in CuTeDSL dense GEMM, focusing on MMA atoms, TMA atoms, layouts, and non-clustered setup. |
| 2025-06-28 | Thread Value Layouts in CuTe | Provides a visual explanation of CuTe thread-value layouts, mapping linear memory values to threads through value and thread layouts. |
| 2025-06-26 | SGEMM in CuTeDSL | Gives a top-down walkthrough of the CUTLASS CuTeDSL SGEMM example, covering tensor layouts, major modes, problem setup, and kernel structure. |
| 2025-06-23 | An applied introduction to CuTeDSL | Introduces CuTeDSL through an elementwise-add kernel, showing tensor layouts, launching, compilation, and correctness checks. |
| 2025-06-21 | Calculating the fibonacci numbers on GPU | Uses Thrust scans and associative matrix multiplication to compute Fibonacci numbers on the GPU. |
| 2025-06-16 | An introduction to Thrust | Introduces Thrust as a CUDA parallel algorithms library using segmented sum as the worked example. |
| 2025-06-09 | Infinite binary strings | Maps infinite binary strings to `[0,1]`, discussing non-injective binary expansions, cylinder sets, and prefix containment. |
| 2025-06-06 | Highly efficient matrix transpose in Mojo | Implements a Hopper matrix-transpose kernel in Mojo with TMA and shared-memory tiling, reporting CUDA-like H100 bandwidth upstream. |
| 2025-06-05 | The Bijection Between Natural Numbers and Binary Strings | Defines finite binary strings and proves a concrete bijection with natural numbers by prefixing and dropping a leading `1`. |
| 2025-06-04 | Use TMA without CUDA | Walks through a Mojo implementation of Hopper TMA tile load/store using TMA descriptors, barriers, bulk tensor copy, and store fences. |
| 2025-05-29 | Use PTX instructions in Mojo | Shows how to call custom PTX intrinsics from Mojo by studying `tanh` and adding lower-precision PTX instruction support. |
| 2025-05-25 | Very fast vector sum without CUDA. | Optimizes vector reduction in Mojo on H100 through shared-memory reduction, warp/shuffle techniques, atomics, and comparisons to CUB/CUDA. |
| 2025-05-22 | Short introduction to the Mojo programming language | Introduces Mojo GPU programming with setup, vector-add code, CUDA parallels, and compilation workflow. |
| 2025-05-18 | Bridging Math and Code: CuTe Layout Algebra in CuTeDSL | Explains CuTe/CuTeDSL layout algebra concepts, including layout functions, sorted layouts, coalescing, and complementation with code examples. |
| 2025-05-14 | Load and store matrices efficently with PTX instructions | Explains PTX `ldmatrix`/matrix load-store instructions, shared-memory address conversion, inline-CUDA wrappers, and Blackwell-only low-bit shapes. |
| 2025-05-11 | How to use reasoning models with SGLang | Provides a hands-on SGLang/Qwen3 guide covering Docker setup, OpenAI-compatible inference, reasoning parsing, and structured output. |
| 2025-05-02 | Making matrix transpose really fast on Hopper GPUs | Builds a Hopper TMA matrix transpose and improves it with 128B shared-memory swizzling and explicit swizzle-index formulas. |
| 2025-04-27 | TMA introduction | Introduces Hopper TMA for 2D global/shared memory transfers using tensor maps, async barriers, TMA load/store, and swizzling. |
| 2025-04-21 | Analyze CUDA programs by looking at GPU assembly. | Uses SASS for scalar versus vectorized vector copy to explain why `LDG/STG.E.128` improves memory-bound kernels. |
| 2025-04-18 | Making RMSNorm really fast | Implements and optimizes RMSNorm for LLM-shaped matrices using reductions, shared memory, warp reductions, and vectorized load/store. |
| 2025-04-13 | Making prefix sum really fast | Optimizes blockwise prefix scan through shared memory, double buffering, warp primitives, and thread coarsening. |
| 2025-04-06 | Making vector sum really fast | Optimizes CUDA vector reduction from a two-pass shared-memory baseline to warp-aware, atomic, batched, and vectorized approaches near H100 bandwidth limits upstream. |
| 2025-03-31 | Predication in Cutlass | Shows how to add CuTe/CUTLASS predication to tiled copies so non-divisible matrix shapes avoid out-of-bounds accesses. |
| 2025-03-23 | Indexing in CUDA | Explains row-major indexing and derives shared-memory/register indexing in a tiled CUDA SGEMM kernel. |

## Imported Post Manifest

| Date | Title | Source | Upstream |
| --- | --- | --- | --- |
| 2026-05-15 | Tile Scheduling in CuTeDSL | `../../sources/html/veitner-bearblog/tile-scheduling-cute-dsl.html` | https://veitner.bearblog.dev/tile-scheduling-cute-dsl/ |
| 2026-04-18 | SBO and LBO explained visually | `../../sources/html/veitner-bearblog/sbo-and-lbo-explained-visually.html` | https://veitner.bearblog.dev/sbo-and-lbo-explained-visually/ |
| 2026-03-14 | Simple math to speed up GDN prefill | `../../sources/html/veitner-bearblog/simple-math-to-speed-up-gdn-prefill.html` | https://veitner.bearblog.dev/simple-math-to-speed-up-gdn-prefill/ |
| 2026-03-10 | Chunkwise Gated Delta Rule | `../../sources/html/veitner-bearblog/chunkwise-gated-delta-rule.html` | https://veitner.bearblog.dev/chunkwise-gated-delta-rule/ |
| 2026-02-15 | Gated Delta Net Decoding | `../../sources/html/veitner-bearblog/gated-delta-net-decoding.html` | https://veitner.bearblog.dev/gated-delta-net-decoding/ |
| 2026-02-07 | Grouped Blockscaled Gemm - Kernel | `../../sources/html/veitner-bearblog/grouped-blockscaled-gemm-kernel.html` | https://veitner.bearblog.dev/grouped-blockscaled-gemm-kernel/ |
| 2026-01-24 | Grouped Blockscaled Gemm - Host code | `../../sources/html/veitner-bearblog/grouped-blockscaled-gemm-host-code.html` | https://veitner.bearblog.dev/grouped-blockscaled-gemm-host-code/ |
| 2026-01-22 | Grouped Block scaled Gemm - Intro | `../../sources/html/veitner-bearblog/grouped-block-scaled-gemm-intro.html` | https://veitner.bearblog.dev/grouped-block-scaled-gemm-intro/ |
| 2026-01-07 | Warp Specialisation in CuTeDSL | `../../sources/html/veitner-bearblog/warp-specialisation-in-cutedsl.html` | https://veitner.bearblog.dev/warp-specialisation-in-cutedsl/ |
| 2026-01-04 | 2 CTA GEMM on B200 | `../../sources/html/veitner-bearblog/2-cta-gemm-on-b200.html` | https://veitner.bearblog.dev/2-cta-gemm-on-b200/ |
| 2025-12-23 | Blackwell Pipelining with CuTeDSL | `../../sources/html/veitner-bearblog/blackwell-pipelining-with-cutedsl.html` | https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/ |
| 2025-12-06 | B200 Blockscaled GEMM - The setup | `../../sources/html/veitner-bearblog/b200-blockscaled-gemm-the-setup.html` | https://veitner.bearblog.dev/b200-blockscaled-gemm-the-setup/ |
| 2025-12-03 | Scale Tensor construction in CuTeDSL | `../../sources/html/veitner-bearblog/scale-tensor-construction-in-cutedsl.html` | https://veitner.bearblog.dev/scale-tensor-construction-in-cutedsl/ |
| 2025-11-23 | Demystifying numeric conversions in CuTeDSL | `../../sources/html/veitner-bearblog/demystifying-numeric-conversions-in-cutedsl.html` | https://veitner.bearblog.dev/demystifying-numeric-conversions-in-cutedsl/ |
| 2025-11-16 | NVFP4 GEMV improved | `../../sources/html/veitner-bearblog/nvfp4-gemv-improved.html` | https://veitner.bearblog.dev/nvfp4-gemv-improved/ |
| 2025-11-13 | NVFP4 GEMV | `../../sources/html/veitner-bearblog/nvfp4-gemv.html` | https://veitner.bearblog.dev/nvfp4-gemv/ |
| 2025-11-06 | Bit counting and geometric series | `../../sources/html/veitner-bearblog/bit-counting-and-geometric-series.html` | https://veitner.bearblog.dev/bit-counting-and-geometric-series/ |
| 2025-10-27 | Simple reduction in CuTeDSL | `../../sources/html/veitner-bearblog/simple-reduction-in-cutedsl.html` | https://veitner.bearblog.dev/simple-reduction-in-cutedsl/ |
| 2025-10-20 | MMA Atoms in CuTe | `../../sources/html/veitner-bearblog/mma-atoms-in-cute.html` | https://veitner.bearblog.dev/mma-atoms-in-cute/ |
| 2025-10-11 | LRU and LFU in C++ | `../../sources/html/veitner-bearblog/lru-and-lfu-in-c.html` | https://veitner.bearblog.dev/lru-and-lfu-in-c/ |
| 2025-10-03 | Mutual Refinement and Composition | `../../sources/html/veitner-bearblog/mutual-refinement-and-composition.html` | https://veitner.bearblog.dev/mutual-refinement-and-composition/ |
| 2025-09-27 | Applied introduction to Categorical treatment of CuTe | `../../sources/html/veitner-bearblog/applied-introduction-to-categorical-treatment-of-cute.html` | https://veitner.bearblog.dev/applied-introduction-to-categorical-treatment-of-cute/ |
| 2025-09-23 | Layout Gymnastics | `../../sources/html/veitner-bearblog/layout-gymnastics.html` | https://veitner.bearblog.dev/layout-gymnastics/ |
| 2025-09-20 | Swizzles and their usage in CuTeDSL Kernels | `../../sources/html/veitner-bearblog/swizzles-and-their-usage-in-cutedsl-kernels.html` | https://veitner.bearblog.dev/swizzles-and-their-usage-in-cutedsl-kernels/ |
| 2025-09-14 | CuTe partitions | `../../sources/html/veitner-bearblog/cute-partitions.html` | https://veitner.bearblog.dev/cute-partitions/ |
| 2025-09-09 | Tensors Slicing in CuTe | `../../sources/html/veitner-bearblog/tensors-slicing-in-cute.html` | https://veitner.bearblog.dev/tensors-slicing-in-cute/ |
| 2025-09-07 | Understanding CuTe Swizzling - The Math Behind 32B, 64B, and 128B Patterns | `../../sources/html/veitner-bearblog/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns.html` | https://veitner.bearblog.dev/understanding-cute-swizzling-the-math-behind-32b-64b-and-128b-patterns/ |
| 2025-09-05 | GPU L2 Cache Persistence | `../../sources/html/veitner-bearblog/gpu-l2-cache-persistence.html` | https://veitner.bearblog.dev/gpu-l2-cache-persistence/ |
| 2025-08-29 | Cuda streams | `../../sources/html/veitner-bearblog/cuda-streams.html` | https://veitner.bearblog.dev/cuda-streams/ |
| 2025-08-23 | PingPong in the CuTeDSL with QuACK | `../../sources/html/veitner-bearblog/pingpong-in-the-cutedsl-with-quack.html` | https://veitner.bearblog.dev/pingpong-in-the-cutedsl-with-quack/ |
| 2025-08-17 | Bit Hacking in C | `../../sources/html/veitner-bearblog/bit-hacking-in-c.html` | https://veitner.bearblog.dev/bit-hacking-in-c/ |
| 2025-08-13 | Intuition behind Hierarchical Layouts | `../../sources/html/veitner-bearblog/intuition-behind-hierarchical-layouts.html` | https://veitner.bearblog.dev/intuition-behind-hierarchical-layouts/ |
| 2025-08-09 | Persistent Float8 Dense Gemm on Hopper | `../../sources/html/veitner-bearblog/persistent-float8-dense-gemm-on-hopper.html` | https://veitner.bearblog.dev/persistent-float8-dense-gemm-on-hopper/ |
| 2025-08-04 | Epilogue in CuTeDSL H100 kernels | `../../sources/html/veitner-bearblog/epilogue-h100-cutedsl.html` | https://veitner.bearblog.dev/epilogue-h100-cutedsl/ |
| 2025-07-30 | Let the compiler do the work in CuTeDSL | `../../sources/html/veitner-bearblog/let-the-compiler-do-the-work-in-cutedsl.html` | https://veitner.bearblog.dev/let-the-compiler-do-the-work-in-cutedsl/ |
| 2025-07-25 | Persistent GEMM in CuTeDSL on Hopper | `../../sources/html/veitner-bearblog/persistent-gemm-in-cutedsl-on-hopper.html` | https://veitner.bearblog.dev/persistent-gemm-in-cutedsl-on-hopper/ |
| 2025-07-20 | Consumer-Producer pattern on H100 in CuTeDSL | `../../sources/html/veitner-bearblog/consumer-producer-pattern-on-h100-in-cutedsl.html` | https://veitner.bearblog.dev/consumer-producer-pattern-on-h100-in-cutedsl/ |
| 2025-07-13 | Backprob through Layernorm | `../../sources/html/veitner-bearblog/backprob-through-layernorm.html` | https://veitner.bearblog.dev/backprob-through-layernorm/ |
| 2025-07-13 | Backprop through RMSNorm | `../../sources/html/veitner-bearblog/backprop-through-rmsnorm.html` | https://veitner.bearblog.dev/backprop-through-rmsnorm/ |
| 2025-07-12 | Outperform compiled PyTorch code using QuACK | `../../sources/html/veitner-bearblog/outperform-compiled-pytorch-code-using-quack.html` | https://veitner.bearblog.dev/outperform-compiled-pytorch-code-using-quack/ |
| 2025-07-05 | CuTeDSL on Hopper - Pipelining | `../../sources/html/veitner-bearblog/cutedsl-on-hopper-pipelining.html` | https://veitner.bearblog.dev/cutedsl-on-hopper-pipelining/ |
| 2025-07-03 | CuTeDSL on Hopper - WGMMA and TMA intro | `../../sources/html/veitner-bearblog/cutedsl-on-hopper-wgmma-and-tma-intro.html` | https://veitner.bearblog.dev/cutedsl-on-hopper-wgmma-and-tma-intro/ |
| 2025-06-28 | Thread Value Layouts in CuTe | `../../sources/html/veitner-bearblog/thread-value-layouts-in-cute.html` | https://veitner.bearblog.dev/thread-value-layouts-in-cute/ |
| 2025-06-26 | SGEMM in CuTeDSL | `../../sources/html/veitner-bearblog/sgemm-in-cutedsl.html` | https://veitner.bearblog.dev/sgemm-in-cutedsl/ |
| 2025-06-23 | An applied introduction to CuTeDSL | `../../sources/html/veitner-bearblog/an-applied-introduction-to-cutedsl.html` | https://veitner.bearblog.dev/an-applied-introduction-to-cutedsl/ |
| 2025-06-21 | Calculating the fibonacci numbers on GPU | `../../sources/html/veitner-bearblog/calculating-the-fibonacci-numbers-on-gpu.html` | https://veitner.bearblog.dev/calculating-the-fibonacci-numbers-on-gpu/ |
| 2025-06-16 | An introduction to Thrust | `../../sources/html/veitner-bearblog/an-introduction-to-thrust.html` | https://veitner.bearblog.dev/an-introduction-to-thrust/ |
| 2025-06-09 | Infinite binary strings | `../../sources/html/veitner-bearblog/infinite-binary-string.html` | https://veitner.bearblog.dev/infinite-binary-string/ |
| 2025-06-06 | Highly efficient matrix transpose in Mojo | `../../sources/html/veitner-bearblog/highly-efficient-matrix-transpose-in-mojo.html` | https://veitner.bearblog.dev/highly-efficient-matrix-transpose-in-mojo/ |
| 2025-06-05 | The Bijection Between Natural Numbers and Binary Strings | `../../sources/html/veitner-bearblog/the-bijection-between-natural-numbers-and-binary-strings.html` | https://veitner.bearblog.dev/the-bijection-between-natural-numbers-and-binary-strings/ |
| 2025-06-04 | Use TMA without CUDA | `../../sources/html/veitner-bearblog/use-tma-without-cuda.html` | https://veitner.bearblog.dev/use-tma-without-cuda/ |
| 2025-05-29 | Use PTX instructions in Mojo | `../../sources/html/veitner-bearblog/use-ptx-instructions-in-mojo.html` | https://veitner.bearblog.dev/use-ptx-instructions-in-mojo/ |
| 2025-05-25 | Very fast vector sum without CUDA. | `../../sources/html/veitner-bearblog/very-fast-vector-sum-without-cuda.html` | https://veitner.bearblog.dev/very-fast-vector-sum-without-cuda/ |
| 2025-05-22 | Short introduction to the Mojo programming language | `../../sources/html/veitner-bearblog/short-introduction-to-the-mojo-programming-language.html` | https://veitner.bearblog.dev/short-introduction-to-the-mojo-programming-language/ |
| 2025-05-18 | Bridging Math and Code: CuTe Layout Algebra in CuTeDSL | `../../sources/html/veitner-bearblog/bridging-math-and-code-cute-layout-algebra-in-cutedsl.html` | https://veitner.bearblog.dev/bridging-math-and-code-cute-layout-algebra-in-cutedsl/ |
| 2025-05-14 | Load and store matrices efficently with PTX instructions | `../../sources/html/veitner-bearblog/load-and-store-matrices-efficently-with-ptx-instructions.html` | https://veitner.bearblog.dev/load-and-store-matrices-efficently-with-ptx-instructions/ |
| 2025-05-11 | How to use reasoning models with SGLang | `../../sources/html/veitner-bearblog/how-to-use-reasoning-models-with-sglang.html` | https://veitner.bearblog.dev/how-to-use-reasoning-models-with-sglang/ |
| 2025-05-02 | Making matrix transpose really fast on Hopper GPUs | `../../sources/html/veitner-bearblog/making-matrix-transpose-really-fast-on-hopper-gpus.html` | https://veitner.bearblog.dev/making-matrix-transpose-really-fast-on-hopper-gpus/ |
| 2025-04-27 | TMA introduction | `../../sources/html/veitner-bearblog/tma-introduction.html` | https://veitner.bearblog.dev/tma-introduction/ |
| 2025-04-21 | Analyze CUDA programs by looking at GPU assembly. | `../../sources/html/veitner-bearblog/analyze-cuda-programs-by-looking-at-gpu-assembly.html` | https://veitner.bearblog.dev/analyze-cuda-programs-by-looking-at-gpu-assembly/ |
| 2025-04-18 | Making RMSNorm really fast | `../../sources/html/veitner-bearblog/making-rmsnorm-really-fast.html` | https://veitner.bearblog.dev/making-rmsnorm-really-fast/ |
| 2025-04-13 | Making prefix sum really fast | `../../sources/html/veitner-bearblog/making-prefix-sum-really-fast.html` | https://veitner.bearblog.dev/making-prefix-sum-really-fast/ |
| 2025-04-06 | Making vector sum really fast | `../../sources/html/veitner-bearblog/making-vector-sum-really-fast.html` | https://veitner.bearblog.dev/making-vector-sum-really-fast/ |
| 2025-03-31 | Predication in Cutlass | `../../sources/html/veitner-bearblog/predication-in-cutlass.html` | https://veitner.bearblog.dev/predication-in-cutlass/ |
| 2025-03-23 | Indexing in CUDA | `../../sources/html/veitner-bearblog/indexing-in-cuda.html` | https://veitner.bearblog.dev/indexing-in-cuda/ |

## Open Questions

- Which B200 block-scaled GEMM posts should become individual source notes for the first concrete Blackwell kernel-practice drill?
- Which upstream code repositories, if any, contain exact benchmark scripts corresponding to the reported kernel timings?
- How do the NVFP4 GEMV and grouped block-scaled GEMM examples map onto the real V4 Flash decode shapes, active expert counts, scale-tensor layouts, and served-concurrency regimes?
- Which Gated Delta Net invariants are useful for future hybrid-state serving correctness tests, and which are model-specific enough to ignore for V4 Flash?

## Follow-Ups

- Use the B200 block-scaled GEMM sequence as candidate background for a dedicated kernel-practice item once a benchmark shape is selected.
- Promote the tile-scheduling, warp-specialization, and SBO/LBO posts into focused source notes if a Blackwell persistent-GEMM or `tcgen05` descriptor investigation starts.
- If any post's images become necessary for an investigation, decide whether to mirror those image assets explicitly; this import preserves only the HTML snapshots and their upstream image URLs.
