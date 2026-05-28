# vLLM: EAGLE 3.1 Blog

Source: `../../sources/github/vllm-eagle-3-1-blog.md`

Upstream: https://vllm.ai/blog/2026-05-26-eagle-3-1

Raw source: https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-05-26-eagle-3-1.md

Snapshot metadata: `vllm-project/vllm-project.github.io` main commit `042413bee914c2e59e9eecd0b2d7bfa5da177858`, source blob `abe000e85b484a82de15a8b419f723728199d371`.

Related sources:

- vLLM PR: https://github.com/vllm-project/vllm/pull/42764
- Attention Drift arXiv: https://arxiv.org/abs/2605.09992

Status: summarized from imported GitHub-hosted blog markdown on 2026-05-28. The imported source was reviewed for obvious private-key, cloud/API-token, and credential-marker patterns before commit; no secrets were found.

## Summary

The vLLM blog, authored by the EAGLE, vLLM, and TorchSpec teams, introduces EAGLE 3.1 as an algorithm and deployment update to EAGLE 3 speculative decoding. The core claim is that EAGLE 3.1 improves robustness and deployability by addressing attention drift, where the drafter's attention shifts away from sink or prompt tokens toward its own generated tokens as speculation depth grows.

The source attributes the drift to imbalanced fused hidden-state inputs and hidden-state magnitude growth through an unnormalized residual path. EAGLE 3.1 adds FC normalization after each target hidden state and feeds post-norm hidden states into the next decoding step. The blog says these changes improve train-time to inference-time extrapolation, long-context robustness, resilience to chat-template and system-prompt variation, and acceptance-length stability.

This is a speculative-decoding source, not a DeepSeek V4 Flash source. It does not provide a V4 Flash EAGLE 3.1 path. Its relevance to this repo is as a strong signal that speculative fast paths need their own robustness and acceptance instrumentation rather than only aggregate throughput numbers.

## Key Claims

- EAGLE 3.1 is a joint EAGLE/vLLM/TorchSpec update focused on speculative decoding robustness, efficiency, and deployability.
- Speculative decoding can degrade under different chat templates, long-context inputs, and out-of-distribution system prompts.
- The blog names this fragility `attention drift`: as speculation depth increases, the drafter shifts attention from sink tokens toward its own generated tokens.
- The reported causes are imbalanced fused target hidden states and hidden-state magnitude growth across speculation steps through an unnormalized residual path.
- EAGLE 3.1 adds FC normalization after each target hidden state and before the FC layer.
- EAGLE 3.1 feeds post-norm hidden states into the next decoding step.
- The source reports up to 2x longer acceptance length than EAGLE 3 on long-context workloads.
- TorchSpec has training support for EAGLE 3.1 and the teams released an EAGLE 3.1 draft model for Kimi K2.6: `lightseekorg/kimi-k2.6-eagle3.1-mla`.
- vLLM integrated EAGLE 3.1 as a config-driven extension of the existing `method: "eagle3"` code path, with backward compatibility for existing EAGLE 3 checkpoints.
- The source says support is merged into vLLM `main`, will be in nightly builds, and is planned for vLLM `v0.22.0`.
- The early reported benchmark is Kimi-K2.6-NVFP4, vLLM, TP=4, GB200, non-disaggregated serving, SPEED-Bench coding, with 2.03x higher per-user output throughput at concurrency 1, 1.71x at concurrency 4, and 1.66x at concurrency 16.

## Relevance To This Repo

EAGLE 3.1 directly strengthens the repo's distinction between serving correctness, serving quality gates, and performance. A speculative fast path can be lossless in principle while still creating practical serving risk if its acceptance length collapses or its draft behavior becomes unstable under long contexts, prompt-template changes, or out-of-distribution system prompts.

For the V4 Flash investigation, this source is not currently a baseline path because it does not report a DeepSeek V4 Flash draft model or launch recipe. It is still relevant to the first baseline plan because DeepSeek V4 and SGLang recipes already expose MTP/draft-token configuration, and any future external EAGLE-style draft path must be recorded as a separate runtime config with draft-model provenance, acceptance metrics, and quality-gate results.

The source also connects Fern's CoCo thread to speculative decoding. CoCo frames off-manifold context recovery as a training-time robustness problem; EAGLE 3.1 frames draft-token stability under long-context and prompt-template shifts as a speculative serving problem. Both argue against treating high-TPS decode as only a kernel or scheduler objective.

## Benchmark And Evaluation Implications

- Treat the blog's throughput and acceptance-length numbers as external reported evidence unless locally reproduced with the exact model, vLLM commit or release, hardware, workload, and commands.
- Any EAGLE 3.1 benchmark record should pin the target model, draft model, vLLM commit or release, TorchSpec/training provenance when relevant, speculative config, number of speculative tokens, acceptance length, acceptance by position, and rollback behavior.
- Compare no-spec and EAGLE 3.1 paths under the same predeclared quality gate, not only under tokens/s/user.
- Include prompt-template and long-context sensitivity in follow-up robustness probes if speculative decoding becomes part of a baseline or intervention.
- Keep EAGLE 3.1 as an algorithm/update term, not as a LLaMA 3.1 model identity.

## Open Questions

- Is there or will there be an official EAGLE 3.1 draft model for `deepseek-ai/DeepSeek-V4-Flash`?
- How do EAGLE 3.1 acceptance length and stability compare to model-native MTP on the same target model, hardware, and workload?
- What exact vLLM commit first contains the merged EAGLE 3.1 support, and which nightly or release image should be pinned for reproduction?
- Does attention drift appear in DeepSeek V4's model-native MTP heads or only in external autoregressive draft models?
- Which corrupted-context or chat-template variations are suitable for a public report-only robustness suite without overfitting to one runtime or model family?

## Follow-Ups

- If a V4 Flash EAGLE 3.1 draft model appears, ingest its model card and launch recipe before adding it to any baseline matrix.
- Add acceptance metrics and draft-model provenance to any future result template or benchmark record that enables EAGLE-style speculative decoding.
- Use this source when designing corrupted-prefix and prompt-template robustness probes for agentic prefix-cache workloads.
