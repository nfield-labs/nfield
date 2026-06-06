# The seven-stage pipeline

Every extraction runs the same stages (S0–S6). Each is a pure step over a shared
`PipelineState`; only S0, S4, and S5 ever touch the provider.

| Stage | Name | API calls | What it does |
|-------|------|-----------|--------------|
| S0 | Resource calibration | 1 (once per engine) | Measure `chars_per_token`; read context window / output ceiling. |
| S1 | Schema analysis | 0 | Flatten schema to dot-notation fields; per-field token cost (τ), dependency graph, difficulty. |
| S2A | Structural grouping | 0 | Group fields by their parent path in the schema tree. |
| S2.5 | Document pre-pass | 0 | Chunk the document, build a BM25 index, score chunks per group (`D_cost`). |
| S2C | Capacity packing | 0 | Pack groups into leaves that fit context + output budgets; order by dependencies. |
| S3 | Excerpt finalisation | 0 | Per leaf: collect, dedup, trim, and order the matched document spans. |
| S4 | Extraction | K (one per leaf) | Build the prompt, call the model, parse the SFEP response. |
| S5 | Validation & retry | R (retry only) | Validate each field; surgically re-extract only the failures. |
| S6 | Assembly | 0 | Reassemble flat pairs into nested JSON; compute quality and status. |

## Why decomposition

A single call cannot reliably emit a very wide schema. Capacity packing computes how
many calls are actually needed from the model's real numbers — `K_min` is the lower
bound — and splits the schema only as much as the budget requires. On a small schema
that fits one call, there is exactly one leaf; on a 1000-field schema it is many.

## Calibration is measured, not guessed

`chars_per_token` is a property of the model's tokenizer (the Normalized Sequence
Length, chars/token), so it is measured once at first use and reused for the life of
the engine — never hardcoded and never re-measured per document.
