# DeepSeek V4 Flash Hugging Face Model Card

Source: `../../sources/huggingface/deepseek-ai-deepseek-v4-flash-readme.md`

Upstream: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash

Status: summarized from imported Hugging Face README snapshot on 2026-05-22. The source was scanned for obvious credential patterns before adding it to the public repo.

## Summary

The Hugging Face model card is the primary open-weight source for `deepseek-ai/DeepSeek-V4-Flash`. It describes the V4 model family, lists download targets, states the license, records benchmark tables, and documents local chat encoding requirements.

For this repo, it is the canonical source for open-weight availability, checkpoint precision, context length, recommended sampling defaults, and the absence of a standard Jinja chat template.

## Key Claims

- `deepseek-ai/DeepSeek-V4-Flash` is licensed under MIT and uses the Transformers `deepseek_v4` integration.
- DeepSeek-V4-Flash has 284B total parameters, 13B active parameters, and 1M context length.
- The instruct checkpoint uses FP4 plus FP8 mixed precision, with MoE expert parameters in FP4 and most other parameters in FP8.
- The V4 family combines Compressed Sparse Attention, Heavily Compressed Attention, mHC, and Muon training.
- Both V4-Pro and V4-Flash support Non-think, Think High, and Think Max reasoning modes.
- The release does not include a Jinja-format chat template; it provides a dedicated `encoding` folder with Python encode/parse scripts and tests.
- Local deployment recommends `temperature = 1.0` and `top_p = 1.0`.
- Think Max recommends a context window of at least 384K tokens.

## Relevance To This Repo

This source gives the benchmark harness concrete model identifiers and serving-protocol requirements. The missing Jinja template is especially important because an end-to-end benchmark must validate tokenizer, prompt encoding, parsing, and reasoning-mode behavior, not just model weights.

## Kernel And Benchmark Implications

- Benchmark setup should record checkpoint precision and whether the runtime serves the original FP4 plus FP8 mixed checkpoint or a converted checkpoint.
- Serving correctness should include a prompt-encoding smoke test using the official encoding path.
- Formal eval-suite prompts should declare reasoning mode and context-window budget.
- Kernel hypotheses should treat FP4 expert weights and FP8 dense/attention paths as first-class constraints.

## Open Questions

- Which exact Hugging Face commit SHA should pin the first benchmark's model metadata, tokenizer, encoding, and inference references?
- How should the benchmark distinguish Non-think, Think High, and Think Max as separate quality/performance regimes beyond the first quality gate's Non-think and Think High modes?

## Follow-Ups

- Record exact Hugging Face repo ID, revision commit SHA, and file paths for config, tokenizer, encoding, and inference documentation before writing benchmark code.
- Add serving correctness checks for the official DeepSeek V4 encoding path.
