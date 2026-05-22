# DeepSeek V4 Preview Release

Source: `../../sources/html/deepseek-api-news-deepseek-v4-preview-release.html`

Upstream: https://api-docs.deepseek.com/news/news260424

Status: summarized from imported HTML snapshot on 2026-05-22. The source was scanned for obvious credential patterns before adding it to the public repo.

## Summary

This DeepSeek API news post announces the DeepSeek-V4 Preview release, including open weights and API availability for DeepSeek-V4-Pro and DeepSeek-V4-Flash. It frames V4 as a cost-effective 1M-context model family with structural changes for long-context efficiency.

For this repo, the key value is that the source establishes official model availability, API model names, open-weight links, and a high-level architecture claim from DeepSeek itself.

## Key Claims

- DeepSeek-V4 Preview is officially live and open-sourced.
- DeepSeek-V4-Pro has 1.6T total parameters and 49B active parameters.
- DeepSeek-V4-Flash has 284B total parameters and 13B active parameters.
- Both models support 1M context.
- The release highlights token-wise compression plus DeepSeek Sparse Attention as the novel attention path.
- The release links the technical report and the Hugging Face DeepSeek V4 collection.
- API users can switch to `deepseek-v4-pro` or `deepseek-v4-flash` while keeping the same base URL.
- `deepseek-chat` and `deepseek-reasoner` are scheduled for retirement after 2026-07-24 and currently route to V4 Flash non-thinking/thinking modes.

## Relevance To This Repo

This source resolves the earlier question of whether V4 Flash is public enough to target directly: DeepSeek publishes a V4 Flash API endpoint and open-weight repository. It also anchors the first model target to official DeepSeek terminology rather than third-party summaries.

## Kernel And Benchmark Implications

- The first serving benchmark can target real `deepseek-ai/DeepSeek-V4-Flash` instead of a proxy model.
- The API model names are useful for quality comparisons against hosted DeepSeek behavior, but local benchmark claims still need local serving metadata.
- The attention headline points the kernel map toward compressed attention, sparse selection, and KV-cache layout before generic attention tuning.

## Open Questions

- Should hosted DeepSeek API outputs be used only as quality smoke tests, or also as a behavioral reference for local runtimes?
- How should this repo handle the 284B versus 285B total-parameter wording used across official and runtime docs?

## Follow-Ups

- Use the linked technical report and Hugging Face model card as the source of detailed architecture facts.
- Update the planned investigation so official model availability is no longer listed as a blocker.
