# Wiki Log

## [2026-05-21] setup | Repository skeleton

- Established `sources/`, `wiki/`, `benchmarks/`, `kernels/`, and `docs/` as the top-level research structure.
- Added `AGENTS.md` with repo-specific operating rules for sources, wiki maintenance, benchmarks, and kernel practice.
- Moved the general LLM wiki pattern document to `docs/llm-wiki-pattern.md`.

## [2026-05-21] ingest | Initial HTML ramp sources

- Imported two raw HTML source artifacts into `sources/html/` after a quick scan for obvious credentials, emails, and confidential markers.
- Created structured source notes for the DeepSeek V4 Flash model-performance ramp plan and the sparse video attention FA4 follow-on ramp.

## [2026-05-21] maintenance | Scrubbed company-specific wording

- Replaced company-specific ramp-plan wording in current `main` with neutral model-performance and production-serving language.
- Renamed the affected HTML source artifact and source note to neutral filenames.

## [2026-05-21] artifact | V4 Flash kernel map

- Added `wiki/artifacts/v4-flash-kernel-map.md` as the D1 architecture-map artifact for the top-down V4 Flash autopsy.
- Updated the ramp source to use the hyphenated artifact filename.

## [2026-05-21] question | Blackwell kernel frontier

- Created the first research-question page focused on datacenter Blackwell kernels, CUDA, CUTLASS, CuTe DSL, and their connection to high-throughput decode metrics.
