# LongBench v2

Sources:

- `../../sources/html/longbench-v2-project-page.html`
- `../../sources/huggingface/longbench-v2-dataset-card.html`

Upstreams:

- https://longbench2.github.io/
- https://huggingface.co/datasets/THUDM/LongBench-v2

Status: summarized from imported project-page and Hugging Face dataset-card snapshots on 2026-05-22. The sources were scanned for obvious credential patterns before adding them to the public repo.

## Summary

LongBench v2 is a realistic long-context benchmark designed for deep understanding and reasoning over long inputs. It uses multiple-choice questions for reliable scoring and covers single-document QA, multi-document QA, long in-context learning, long-dialogue history understanding, code repository understanding, and long structured-data understanding.

For this repo, the long subset is the realistic complement to MRCR. MRCR checks explicit 1M retrieval/disambiguation, while LongBench v2 long checks whether the serving path preserves meaningful long-context reasoning behavior.

## Key Claims

- LongBench v2 contains 503 challenging multiple-choice questions.
- Contexts range from 8K to 2M words, with most examples under 128K.
- The benchmark defines length categories: short, medium, and long.
- The project page defines long as 128K to 2M words.
- The benchmark covers realistic scenarios across six major categories.
- Human experts achieve 53.7% accuracy under a 15-minute time constraint.
- The data format includes domain, sub-domain, difficulty, length category, question, four answer choices, answer label, and long context.
- The Hugging Face dataset card points to the GitHub repo for automated evaluation.

## Relevance To This Repo

LongBench v2 long should be the required realistic long-context quality gate for the V4 Flash serving-frontier investigation. It is less directly tied to exactly 1M tokens than MRCR, but it is more application-like and avoids relying only on synthetic needle retrieval.

## Benchmark Implications

- Run the entire `length = long` subset for the first V4 Flash quality gate to focus cost on >128K contexts without sampling ambiguity.
- Report aggregate accuracy plus category-level breakdown if sample counts are large enough.
- Run both Non-think and Think High; LongBench v2 leaderboards distinguish settings with and without CoT, so the benchmark record should report the two modes separately.
- Use multiple-choice scoring as the primary metric to avoid model-graded quality gates for the first benchmark.
- Use the first V4 Flash LongBench v2 long run as report-only, then compare it to public leaderboard or DeepSeek V4 report values when settings are comparable.

## Open Questions

No open questions for the first LongBench v2 quality-gate shape. Future runs can add pass/fail regression thresholds once local baseline distributions exist.

## Follow-Ups

- Add the entire LongBench v2 long subset to the benchmark harness as the required realistic long-context quality gate.
