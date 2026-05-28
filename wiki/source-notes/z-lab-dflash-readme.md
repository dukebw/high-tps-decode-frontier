# z-lab DFlash README

Source: `../../sources/github/z-lab-dflash-readme.md`

Upstream: https://github.com/z-lab/dflash

Raw source: https://github.com/z-lab/dflash/blob/main/README.md

Snapshot metadata: `z-lab/dflash` main commit `94e4abc5e0c31b67bc1a9d30f1cc34ece28a8756`, README blob `f4e8533b4f9c978d857703a77a5b6d00e6149844`.

Related paper: https://arxiv.org/abs/2602.06036

Status: summarized from imported GitHub README snapshot on 2026-05-28. The imported source was reviewed for obvious private-key, cloud/API-token, and credential-marker patterns before commit; no secrets were found.

## Summary

The DFlash README describes DFlash as a lightweight block-diffusion model for speculative decoding. Unlike autoregressive draft models, DFlash is positioned as a parallel drafting method: the draft side generates a block of candidate tokens for target-model verification.

The source provides practical serving integration signals rather than a complete benchmark record. It lists supported draft models, says `vLLM v0.20.1+` includes core DFlash support, gives vLLM and SGLang launch examples, and includes simple evaluation commands for GSM8K, MATH500, HumanEval, MBPP, and MT-Bench. The README marks `DeepSeek-V4-Flash`, `DeepSeek-V4-Pro`, and `GLM-5.1` as `Coming soon`, so it is not yet a V4 Flash baseline path.

DFlash is unrelated to the `DeepSeek-V4-Flash` model name despite the shared word `Flash`. In this repo, use `DFlash` only for the block-diffusion speculative decoding algorithm and draft-model ecosystem.

## Key Claims

- DFlash is a lightweight block-diffusion model designed for speculative decoding.
- The claimed purpose is efficient, high-quality parallel drafting for target-model verification.
- The README links the DFlash paper, project blog, and Hugging Face model collection.
- Supported draft models listed in the README include Gemma 4, MiniMax M2.x, Kimi K2.x, Qwen3.x, GPT-OSS, Qwen3-Coder, and Llama 3.1 targets.
- `DeepSeek-V4-Flash`, `DeepSeek-V4-Pro`, and `GLM-5.1` are listed as `Coming soon`, not as available draft targets.
- The README says `vLLM v0.20.1+` includes core DFlash support for most models.
- The vLLM examples use `--speculative-config` with `method: "dflash"`, a DFlash draft-model path, and `num_speculative_tokens` around 15.
- The SGLang example uses `--speculative-algorithm DFLASH`, `--speculative-draft-model-path`, `--speculative-num-draft-tokens 16`, `--speculative-draft-attention-backend fa4`, and optional experimental schedule-overlap environment variables.
- The README says the DFlash training recipe will be open-sourced later.
- The included evaluation commands are smoke or recipe commands, not complete benchmark-result records under this repo's discipline.

## Relevance To This Repo

DFlash is relevant as a future speculative-decoding baseline or intervention candidate for high-TPS serving, especially if V4 Flash support ships publicly. It may change the draft/verify balance because a block-diffusion drafter can generate candidate tokens in parallel rather than autoregressively stepping through a draft chain.

The source also reinforces that speculative acceleration is a runtime-and-quality question, not only a kernel question. DFlash examples involve runtime support, draft attention backends, speculative-token counts, scheduler overlap, and backend-specific constraints. Any future DFlash run needs end-to-end serving instrumentation, acceptance metrics, quality gates, and fallback/rollback correctness checks.

## Benchmark And Evaluation Implications

- Do not treat DFlash as part of the current V4 Flash baseline until a public V4-compatible draft model and reproducible launch path exist.
- If added later, record DFlash separately from model-native MTP and EAGLE-style speculative decoding.
- Pin the target model, DFlash draft model, runtime version or commit, vLLM/SGLang config, draft-token count, draft attention backend, acceptance rate, acceptance by position or block position, and rejection/rollback behavior.
- Run the same serving quality gate as the no-spec baseline and record any output differences or failures.
- For agentic prefix-cache workloads, record whether DFlash interacts with cache warmup, cache hit rate, long-context prompts, tool parsers, reasoning parsers, or corrupted-prefix stressors.
- Keep reported README commands as setup evidence only; they do not include hardware, repetitions, TTFT, TPOT, tokens/s/GPU, p50/p95 latency, quality scores, or profiler evidence.

## Open Questions

- When will DFlash release public draft models and recipes for `deepseek-ai/DeepSeek-V4-Flash`?
- What acceptance rates and throughput gains does DFlash achieve on long-context, high-cache-hit agentic workloads compared with model-native MTP?
- How should DFlash block acceptance metrics be normalized against autoregressive EAGLE acceptance length and model-native MTP accepted tokens?
- Does DFlash's parallel drafting path improve or worsen robustness under corrupted prefixes, malformed tool outputs, weird tokens, and chat-template variation?
- Which runtime release first provides stable DFlash support for the models relevant to this repo?

## Follow-Ups

- Watch for a public V4 Flash DFlash model card or README update and ingest it before adding DFlash to benchmark matrices.
- If a local DFlash benchmark is run, record it as a separate runtime config with its own no-spec comparator, quality gate, and acceptance metrics.
- Use DFlash as a distinct concept in synthesis docs; do not abbreviate DeepSeek V4 Flash as DFlash.
