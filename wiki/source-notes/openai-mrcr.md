# OpenAI MRCR

Source: `../../sources/huggingface/openai-mrcr-dataset-card.html`

Upstream: https://huggingface.co/datasets/openai/mrcr

Status: summarized from imported Hugging Face dataset-card snapshot on 2026-05-22. The source was scanned for obvious credential patterns before adding it to the public repo. The related OpenAI GPT-4.1 blog was reviewed during source discovery but could not be imported directly with `curl` because the site returned HTTP 403.

## Summary

OpenAI MRCR is a long-context multi-round co-reference benchmark for testing whether a model can distinguish multiple similar needles hidden inside a long conversation. The model must return a specific instance of an answer, such as the second poem about a target topic, while ignoring near-miss distractors.

For this repo, MRCR is the cleanest public benchmark source for exercising up to a 1M-token context window with an automatic scoring rule and explicit length bins.

## Key Claims

- MRCR stands for Multi-round Co-reference Resolution.
- The task hides 2, 4, or 8 identical asks inside a long synthetic conversation.
- The model must return the requested instance of the matching answer and prepend a required alphanumeric hash.
- The benchmark is hard because needles are selected from the same distribution as distractors, order among needles matters, and longer contexts increase difficulty.
- The measured metric uses Python `difflib.SequenceMatcher` ratio, with score set to 0 if the required hash is missing.
- There are 100 samples per bin.
- Length bins include `[4096, 8192]`, `(8192, 16384]`, `(16384, 32768]`, `(32768, 65536]`, `(65536, 131072]`, `(131072, 262144]`, `(262144, 524288]`, and `(524288, 1048576]`.
- The dataset has 2,400 rows and is MIT licensed.

## Relevance To This Repo

MRCR should be the first required serving quality gate for the V4 Flash serving-frontier investigation. It directly exercises 1M-context retrieval/disambiguation and aligns with DeepSeek V4's own reported long-context evaluation category, while staying reproducible enough for local or faithfully cited runs.

## Benchmark Implications

- Run all 100 MRCR 2-needle samples in each selected bin: 128K, 512K, and 1M.
- Report per-bin scores rather than only an aggregate, because context-length degradation is the point.
- Run both Non-think and Think High, with sampling parameters, max context, and output token cap fixed before performance runs.
- Treat missing-hash failures as quality failures, not just formatting noise, because the official metric scores them as 0.
- Use the first V4 Flash MRCR run as report-only, then compare it to DeepSeek V4 report values or public MRCR references when settings are comparable.

## Open Questions

No open questions for the first MRCR quality-gate shape. Future runs can add 4-needle and 8-needle samples after the first frontier is established.

## Follow-Ups

- Add MRCR 2-needle 128K, 512K, and 1M to the benchmark harness as required quality-gate bins.
