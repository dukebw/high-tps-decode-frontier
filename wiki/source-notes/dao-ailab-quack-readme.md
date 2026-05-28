# Dao-AILab QuACK CuTe Kernels README

Source: `../../sources/github/dao-ailab-quack-readme.md`

Upstream: https://github.com/Dao-AILab/quack

Raw source: https://github.com/Dao-AILab/quack/blob/main/README.md

Snapshot metadata: `Dao-AILab/quack` main commit `34cfe42fe994dd961e42bbf179539b16d896aab4` (pushed 2026-05-28), README blob `764f136b01f16d4ea469ea57db13be7550bc739e`, Apache-2.0 license.

Related blogpost: `media/2025-07-10-membound-sol.md` in-repo, "Getting Memory-bound Kernels to Speed-of-Light" by Wentao Guo, Ted Zadouri, and Tri Dao.

Status: summarized from imported GitHub README snapshot on 2026-05-28. The README and the linked blogpost were reviewed for obvious private-key, cloud/API-token, and credential-marker patterns before commit; no secrets were found. The kernel inventory below is derived from the repository file tree at the pinned commit, not from the README alone, which lists only a subset.

## Summary

QuACK ("A Quirky Assortment of CuTe Kernels") is a public, Apache-2.0 library of GPU kernels from the FlashAttention authors, written entirely in the NVIDIA [CuTe-DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) — the Python DSL layer over CUTLASS/CuTe. It targets H100, B200/B300, and RTX 50 (GeForce Blackwell) GPUs, requires CUDA 12.9+ and Python 3.12, and ships on PyPI as `quack-kernels` with optional CUDA 13.x, matmul-heuristics, and JAX-binding extras.

QuACK is directly on the repo's declared first-class kernel stack (CUDA, CUTLASS, CuTe DSL on datacenter Blackwell). It is a **kernel library, not a serving runtime**: it provides standalone, individually tuned operators, not an end-to-end engine, scheduler, or megakernel. Its two pillars are (1) memory-bound LLM kernels (normalization, softmax, loss, elementwise/reduction) driven to near speed-of-light DRAM bandwidth, and (2) a GEMM + fused-epilogue family spanning Ampere through Blackwell, including block-scaled (microscaling/MX) matmul.

Notably, QuACK contains **no attention/FlashAttention kernels** — it is positioned as complementary to the separate FlashAttention repo. For this repo's purposes it is a candidate baseline path and reference for memory-bound decode-path kernels and Blackwell GEMM epilogues, not for the attention kernels that dominate the V4 Flash CSA/HCA hypotheses.

## Key Claims

- Kernels are authored in CuTe-DSL (Python), explicitly to get CUDA C++-level control and performance without writing CUDA C/C++.
- Supported hardware: H100 (Hopper), B200/B300 (datacenter Blackwell), and RTX 50 (GeForce/Blackwell SM120). Requirements: CUDA toolkit 12.9+ and Python 3.12.
- README-advertised kernels: RMSNorm forward+backward, Softmax forward+backward, Cross entropy forward+backward, Layernorm forward, Hopper GEMM + epilogue, Blackwell GEMM + epilogue, and Blackwell GeForce GEMM + epilogue.
- Public Python entry points include `from quack import rmsnorm, softmax, cross_entropy`; optional JAX bindings exist for some kernels (e.g. `quack.softmax_jax`).
- The linked 2025-07-10 blogpost claims the memory-bound kernels reach near hardware speed-of-light: on a softmax with batch 16K and reduction dim 131K, FP32, on one H100 (HBM3), they measure 3.01 TB/s device memory throughput, 89.7% of the 3.35 TB/s peak.
- The blogpost reports roughly 90% of peak DRAM bandwidth for reduction dims above 4K, and about 50% more throughput than `torch.compile` at reduction dim 262K (3.01 vs 1.89 TB/s for FP32 softmax).
- The blogpost claims QuACK significantly outperforms all its baselines (torch.compile PyTorch 2.7.1, Liger v0.5.10, cuDNN v9.10.1) once the reduction dim is >= 65K, attributing the edge to H100 thread-block cluster reduction over distributed shared memory (DSMEM), which avoids the register spilling that degrades single-SM kernels at very large reductions.
- The blogpost's recipe for speed-of-light memory-bound kernels is two ingredients: GMEM-coalesced (128-bit vectorized) load/store via a Thread-Value (TV) layout, plus a hardware-aware reduction that aggregates at the highest memory tier first (thread -> warp -> block -> cluster) and only forwards locally-reduced values down the hierarchy.
- The blogpost frames the cost as a productivity/performance tradeoff: speed-of-light requires per-operator and even per-input-shape tuning, but the TV-layout/load-store/reduction helpers are reusable across kernels (e.g. RMSNorm patterns transfer to softmax).

## Kernel Inventory (from repo tree at pinned commit)

The README understates the contents. The `quack/` package at commit `34cfe42` includes substantially more than the advertised list:

- GEMM family by architecture: `gemm_sm80` (Ampere), `gemm_sm90` (Hopper), `gemm_sm100` (datacenter Blackwell B200/B300), `gemm_sm120` (GeForce/RTX 50 Blackwell), plus `gemm_base`/`gemm_config`/`gemm_interface`/`gemm_default_epi` scaffolding and a `tile_scheduler`.
- Block-scaled / microscaling matmul: `blockscaled_gemm_utils`, `gemm_blockscaled_interface`, `mx_utils` — i.e. MX-format (MXFP4/MXFP8-style) GEMM support.
- Fused-epilogue and composite GEMM: `gemm_act`, `gemm_dact`, `gemm_norm_act`, `gemm_symmetric`, `gemm_sq_reduce`, `epi_composable`/`epi_ops`/`epi_utils`, plus `linear`, `mlp`, and `linear_cross_entropy` (fused LM-head-style linear + loss).
- Memory-bound LLM kernels: RMSNorm (`rmsnorm`, `rms_final_reduce`, `rmsnorm_config`), `softmax`, `cross_entropy`, layernorm/activation (`activation`), and reduction infrastructure (`reduce`, `reduction_base`).
- Other primitives: `rotary` (RoPE), `hadamard`, `topk`, a `sort/` package (bitonic sort, sorting-network generation), `rounding`, `fast_math`, `complex`.
- Infrastructure: `autotuner`, `pipeline`, `tensormap_manager`, a compile cache (`cache/`), a CuTe-DSL shim/ptxas layer (`dsl/`), `torch_library_op` for PyTorch integration, and JAX utilities (`jax_utils`, `softmax_jax`).
- `benchmarks/` ships per-kernel benchmarks (GEMM, GEMM epilogues/symmetric, RMSNorm, softmax, cross entropy, layernorm, topk, hadamard, SM120 cluster) with PyTorch/torch.compile baselines and saved speedup plots. `docs/` includes JAX bindings, DSL control flow, limitations, and SM120 NCU profiling notes. The repo also maintains its own `AGENTS.md`/`CLAUDE.md`.

## Relevance To This Repo

- QuACK is the strongest public, reproducible example so far of the repo's declared first-class kernel stack (CuTe DSL on Blackwell). It is a natural **baseline path / reference implementation** for memory-bound decode-path kernels (RMSNorm, softmax, cross entropy, RoPE) and for Blackwell GEMM + epilogue, rather than an intervention hypothesis the repo would author from scratch.
- The 2025-07-10 blogpost is a methodological match for this repo's `speed of light` and `roofline limit` language. It derives the DRAM-bandwidth roofline for memory-bound kernels (arithmetic intensity O(1) -> bytes/s bound), reports measured throughput as a percentage of peak HBM, and backs claims with Nsight Compute memory-workload diagrams and SASS register-spill evidence. It is a good template for how a kernel benchmark record in this repo should justify a near-speed-of-light claim.
- The block-scaled / MX GEMM path (`gemm_sm100`, `blockscaled_gemm_utils`, `mx_utils`) is relevant to V4 Flash's MXFP4 indexer cache, FP8 KV, and MoE/DeepGEMM matmuls noted in the V4 Flash kernel map and benchmark plan. `topk`/`sort` are relevant to MoE routing and sampling; `rotary` to RoPE in the decode path.
- QuACK is the inverse of a megakernel: standalone per-operator kernels with explicit per-shape tuning. The blogpost's own productivity/performance-frontier framing and per-input-shape tuning cost reinforce the repo's megakernel gate — fusing across kernel boundaries is a separate, evidence-gated step, and QuACK is the well-optimized per-kernel baseline a megakernel would have to beat end to end.

## Benchmark And Kernel Implications

- Treat the blogpost's throughput numbers as `faithfully cited` / `external reported`, measured on **H100 (HBM3, 3.35 TB/s peak)**, not Blackwell. Do not transfer the 89.7%-of-peak softmax result to B200/B300 without a local measurement: the kernels target Blackwell, but the published speed-of-light figures are Hopper.
- If QuACK kernels are benchmarked locally, record them under the kernel-benchmark discipline: exact GPU/topology, driver/CUDA, `quack-kernels` version or source commit, CuTe-DSL/CUTLASS version, dtype, shapes (batch and reduction dim), TV-layout/cluster-size tuning, vectorization width, the strongest baseline (torch.compile, Liger, cuDNN, cuBLAS, or a hand CUDA/CUTLASS kernel), correctness tolerance vs a reference, and the derived roofline (HBM bandwidth for memory-bound kernels, tensor-core peak for GEMM).
- Pin QuACK by PyPI version or git commit in any benchmark record; the package is actively developed (the pinned commit is a same-day SM100 GEMM epilogue-barrier hotfix), so reproducibility requires an immutable pin.
- For Blackwell GEMM comparisons, separate datacenter `sm100` (B200/B300) from GeForce `sm120` (RTX 50): they are different code paths with different cluster/profiling characteristics, and only `sm100` is in scope for the smallest official Blackwell topology target.
- Keep QuACK distinct from a serving baseline: it has no scheduler, batching, KV cache, or attention kernel, so it can only contribute to kernel-benchmark or kernel-group evidence that explains a serving bottleneck, never to a standalone tokens/s/user vs tokens/s/GPU frontier point.

## Open Questions

- What fraction of Blackwell (B200/B300) HBM3e speed-of-light do QuACK's memory-bound kernels reach, versus the published 89.7% on H100? Does cluster reduction still dominate at large reductions on Blackwell?
- How do QuACK's `gemm_sm100` block-scaled (MX) GEMMs compare to cuBLAS, CUTLASS reference kernels, and DeepGEMM for the FP8/MXFP4 shapes that show up in V4 Flash MoE and indexer paths?
- Are the QuACK RMSNorm/softmax/cross-entropy/RoPE kernels actually on the V4 Flash decode critical path enough to matter end to end, or are they dominated by attention and MoE GEMM time?
- Does the CuTe-DSL authoring path reproduce QuACK-class throughput for a custom kernel this repo would write, or does it require the same per-shape tuning effort the blogpost flags?
- What are the documented `limitations.rst` constraints (shape, dtype, alignment) that would block dropping QuACK kernels into a real serving runtime?

## Follow-Ups

- If a Blackwell GPU is available, run the in-repo `benchmarks/benchmark_rmsnorm.py`, `benchmark_softmax.py`, and `benchmark_gemm.py` under this repo's kernel-benchmark template and record measured percentage-of-roofline on B200/B300, with an explicit strongest baseline.
- When the D4 dense-attention baseline ladder is built, note that QuACK does not supply an attention kernel; the attention baselines come from FlashAttention/cuDNN/CUTLASS, while QuACK can supply the surrounding norm/RoPE/epilogue kernels.
- Cross-link QuACK as evidence in the Blackwell kernel frontier question and as a CuTe-DSL practice reference when kernel-practice items start.
- Watch for QuACK adding attention or Blackwell speed-of-light benchmark numbers; ingest an updated snapshot if the kernel set or published Blackwell results change materially.
