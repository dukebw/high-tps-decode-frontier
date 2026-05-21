# Kernels

This directory will hold kernel-practice items and future implementations.

The first-class stack is CUDA, CUTLASS, and CuTe DSL for datacenter Blackwell. Kernel work should remain connected to high-throughput text LLM decode and should not become disconnected microbenchmark gardening.

## Practice Convention

Each practice item should include:

- Objective.
- Baseline.
- Correctness check.
- Benchmark command.
- Profiling notes.
- Lessons learned.
- Next step.

Use `templates/practice-item-template.md` for new items.

## Initial Practice Areas

- Online softmax and dense attention anatomy.
- Blackwell TMA, TMEM, tcgen05, and CuTe layout algebra.
- CUTLASS grouped GEMM for MoE-like variable-M shapes.
- FP4/FP8 block scaling and epilogue conversion.
- Sparse or compressed attention metadata, packing, and load balance.
