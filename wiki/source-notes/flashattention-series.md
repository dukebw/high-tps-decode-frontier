# FlashAttention Papers 1-4

Sources:

- `../../sources/pdf/flashattention-1-2205.14135.pdf`
- `../../sources/pdf/flashattention-2-flash2.pdf`
- `../../sources/pdf/flashattention-3-2407.08608.pdf`
- `../../sources/pdf/flashattention-4-2603.05451.pdf`

Upstreams:

- https://arxiv.org/pdf/2205.14135
- https://tridao.me/publications/flash2/flash2.pdf
- https://arxiv.org/pdf/2407.08608
- https://arxiv.org/pdf/2603.05451

Related metadata pages:

- https://arxiv.org/abs/2205.14135
- https://arxiv.org/abs/2307.08691
- https://arxiv.org/abs/2407.08608
- https://arxiv.org/abs/2603.05451

Status: summarized from imported PDF snapshots on 2026-05-22. The PDFs were scanned with binary string search for obvious credential, private-key, and scrubbed-company markers before adding them to this public repo; no matches were found.

## Summary

The FlashAttention series is the dense-attention spine for this repo's D4 and D5 kernel work. FlashAttention 1 establishes the IO-aware exact attention algorithm: tile Q/K/V through SRAM, maintain online softmax statistics, avoid materializing the N x N score/probability matrices in HBM, and recompute in backward when that is cheaper than storing intermediates. FlashAttention 2 keeps the same exact-attention semantics but improves GPU work partitioning, occupancy, and warp-level communication. FlashAttention 3 retargets the kernel to Hopper by exploiting TMA/Tensor Core asynchrony, warp specialization, interleaved matmul and softmax, and FP8 block quantization. FlashAttention 4 retargets the approach to Blackwell, where tensor-core throughput scales faster than shared-memory bandwidth and exponential units, so the bottleneck shifts toward non-matmul work, shared-memory traffic, and kernel pipeline balance.

## Paper Claims

### FlashAttention 1

- Title: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness".
- Authors: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Re.
- Core claim: exact attention should be optimized for IO between HBM and on-chip SRAM, not just FLOP count.
- Algorithm: tile Q, K, and V; compute attention blocks on chip; maintain row-wise online softmax max and normalization statistics; avoid HBM materialization of the N x N attention matrix.
- The paper states FlashAttention has O(N) additional memory beyond inputs/output and fewer HBM accesses than standard attention for typical SRAM sizes.
- It extends the same IO-aware approach to block-sparse FlashAttention, where sparse nonzero block fraction reduces the larger IO term.
- Reported results include 15% BERT-large training speedup over the MLPerf 1.1 training-speed record, 3x GPT-2 speedup, and 2.4x Long Range Arena speedup in the paper's settings.

### FlashAttention 2

- Title: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning".
- Author: Tri Dao.
- Core claim: FlashAttention 1 is memory efficient but still reaches only about 25-40% of theoretical maximum FLOP/s because of suboptimal work partitioning.
- Main changes: reduce non-matmul FLOPs, parallelize a single attention head across thread blocks to improve occupancy, and repartition work between warps to reduce shared-memory communication.
- Reported result: around 2x speedup over FlashAttention 1, reaching 50-73% of theoretical maximum FLOP/s on A100, with end-to-end GPT-style training up to 225 TFLOP/s per A100 GPU and 72% model FLOP utilization.

### FlashAttention 3

- Title: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision".
- Authors: Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
- Core claim: Hopper attention needs explicit use of asynchrony rather than just the FA2 work partitioning model.
- Main changes: overlap computation and data movement with Tensor Core and TMA asynchrony, use warp specialization, interleave block-wise matmul and softmax, and add block quantization plus incoherent processing for FP8.
- Reported result: 1.5-2.0x speedup on H100 over FA2; FP16 up to 740 TFLOP/s, about 75% utilization; FP8 close to 1.2 PFLOP/s; FP8 FA3 has 2.6x lower numerical error than the paper's baseline FP8 attention.

### FlashAttention 4

- Title: "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling".
- Authors: Ted Zadouri, Markus Hoehnerbach, Jay Shah, Timmy Liu, Vijay Thakkar, Tri Dao.
- Core claim: Blackwell changes the bottleneck mix because tensor-core throughput doubles while shared-memory bandwidth, exponential units, and other non-matmul resources scale more slowly or remain unchanged.
- Main changes: redesigned pipelines using fully asynchronous MMA operations and larger tiles, software-emulated exponential and conditional softmax rescaling to reduce non-matmul cost, and tensor memory plus 2-CTA MMA mode to reduce shared-memory traffic and backward-pass atomic adds.
- Reported result: up to 1.3x speedup over cuDNN 9.13 and 2.7x over Triton on B200 with BF16, reaching up to 1613 TFLOP/s and 71% utilization.
- Implementation claim: written entirely in CuTe DSL embedded in Python, with 20-30x faster compile times than traditional C++ template approaches while preserving full expressivity.

## Relevance To D4 And D5

D4 dense attention baseline should be built from the FlashAttention sequence rather than from naive PyTorch alone. The baseline ladder should include naive attention for correctness and intuition, PyTorch SDPA for an available library path, installed FlashAttention or cuDNN attention when available, and any FA4/CuTe DSL path available on Blackwell. The D4 artifact should measure not only runtime but also the online-softmax invariants, HBM traffic, shared-memory traffic, tensor-core utilization, and non-matmul overhead.

D5 sparse/compressed attention should treat FlashAttention 1's block-sparse extension as the conceptual bridge, but not assume sparsity wins automatically. The D5 toy benchmark should report metadata/index preparation, mask or top-k construction, packing/scatter cost, load imbalance, and dense attention over compressed entries separately from the actual QK/softmax/V compute. This directly matches the repo's CSA/HCA distinction: HCA is closer to dense attention over compressed KV entries, while CSA adds sparse selection and indexer overhead.

## Kernel And Benchmark Implications

- Dense attention performance claims need the strongest dense baseline available on the target hardware, not just a naive implementation.
- Online softmax state is part of the correctness contract: row maxima, normalization sums, and output rescaling need explicit reference checks across causal and non-causal cases.
- D4 should sweep head dimension, sequence length, phase, dtype, and masking, because each FlashAttention generation shifts bottlenecks differently.
- On Blackwell, FA4 suggests that exponentials, conditional rescaling, shared-memory traffic, tensor memory use, and 2-CTA MMA are first-class variables, not implementation details.
- For D5, sparse/compressed attention must beat a strong dense FA-style baseline after including metadata/index overhead. A lower FLOP count alone is not evidence of a frontier improvement.
- Any custom kernel result should be tied back to serving constraints: decode TPOT, prefill TTFT, total tokens/s/GPU, memory pressure, and quality/correctness constraints.

## Open Questions

- Which FlashAttention or cuDNN attention kernels are available and strongest on the repo's `b200-aws2` environment?
- Can we run a public FA4/CuTe DSL implementation on B200, or should D4 start with PyTorch SDPA and installed runtime kernels first?
- Which exact V4 Flash attention shapes should D4 approximate first: prefill, decode, SWA local windows, HCA compressed entries, or CSA sparse selected entries?
- What profiler counters should be required before claiming a D4 baseline is strong enough for D5 comparison?
- How should FP8, BF16, and any FP4 indexer/cache formats be separated between dense-attention baseline work and V4-specific compressed/sparse experiments?

## Follow-Ups

- Create a D4 dense attention baseline practice item under `kernels/` with shape sweep, references, correctness tolerances, and benchmark commands.
- Create a D5 sparse/compressed attention toy plan only after the D4 baseline ladder is runnable.
- Use FA4 as the Blackwell-specific reading checklist for TMA, TMEM, tcgen05, CuTe DSL, 2-CTA MMA, and non-matmul bottlenecks.
