# SGLang DeepSeek V4 Cookbook

Source: `../../sources/html/sglang-deepseek-v4-cookbook.html`

Upstream: https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4

Status: summarized from imported HTML snapshot on 2026-05-22. The source was scanned for obvious credential patterns before adding it to the public repo.

## Summary

The SGLang cookbook provides an interactive DeepSeek V4 deployment generator across hardware platforms, model variants, quantization modes, serving recipes, parsers, hierarchical cache settings, and MegaMoE settings. It also documents model features, installation, invocation, benchmark hooks, and tuning warnings.

For this repo, SGLang is another likely strongest public reproducible baseline path for V4 Flash, especially because the cookbook exposes Blackwell, Hopper, context-parallel, PD-disaggregated, and MegaMoE recipe choices.

## Key Claims

- Supported hardware choices include B200, B300, GB200, GB300, H200, and H100.
- Model variants include Flash at 285B and Pro at 1.6T.
- FP4 is the default quantization path, with SGLang FP8 available for H100/H200.
- Recipe choices include low-latency, balanced, max-throughput, context-parallel, and prefill/decode disaggregation.
- For V4 Flash, the command generator maps B200, GB200, GB300, and H200 to 4-GPU serving paths, while H100 uses 8 GPUs.
- The generator exposes `--reasoning-parser deepseek-v4` and `--tool-call-parser deepseekv4` options.
- HiCache can enable GPU plus CPU hierarchical KV cache and UnifiedRadixTree behavior.
- SGLang exposes MegaMoE variants W4A8 and W4A4, with MegaMoE limited to Blackwell and not supported for low-latency or context-parallel settings.
- The cookbook warns that DeepEP dispatch-buffer sizing must satisfy `max-running-requests x MTP_draft_tokens <= SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK`.
- The cookbook states that MTP low-latency uses steps 3 and draft tokens 4, balanced uses steps 1 and draft tokens 2, and max-throughput disables MTP.

## Relevance To This Repo

This source is valuable because it maps the V4 Flash target onto concrete deployment recipes and hardware topology choices. It also provides tuning dimensions that are directly relevant to a Pareto frontier sweep: low-latency versus balanced versus max-throughput, MTP settings, `--max-running-requests`, context parallelism, PD disaggregation, HiCache, and MegaMoE.

## Kernel And Benchmark Implications

- The first Blackwell serving target can treat 4xB200, 4xB300, 4xGB200, or 4xGB300 as candidate smallest official topologies, but the exact choice depends on accessible hardware and recipe verification.
- SGLang MegaMoE is an existing baseline/intervention path, not an automatic custom-kernel opportunity.
- For this repo's first baseline selection, SGLang MegaMoE should be treated as part of the baseline config sweep when it is publicly enableable and outperforms non-MegaMoE configs under the same workload and quality gate.
- Benchmark records must capture MTP settings because they change TPOT, throughput, output-token accounting, and quality behavior.
- DeepEP buffer limits and `--max-running-requests` should be swept carefully instead of treated as static defaults.
- HiCache should be disabled, fixed, or explicitly modeled because hierarchical/prefix caching changes the workload bottleneck.

## Open Questions

- Which SGLang recipe emerges strongest for the selected Blackwell topology and workload regime?
- Are the cookbook's verified recipe sets sufficient for result provenance, or does this repo need to rerun every selected point locally?
- Does MegaMoE W4A8 or W4A4 pass the report-only quality gate closely enough to serve as the strongest SGLang baseline config?

## Follow-Ups

- Compare SGLang low-latency, balanced, max-throughput, and supported MegaMoE variants against vLLM before choosing the first baseline.
- Add DeepEP dispatch-buffer and MTP settings to any SGLang benchmark metadata.
- Treat HiCache and prefix caching as separate workload regimes rather than defaulting them on.
