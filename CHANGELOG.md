# Changelog

Notable changes to nfield. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-07-10

### Added

**More providers**
- Native Google Gemini (`google/...`) and Anthropic Claude (`anthropic/...`), alongside
  the existing Groq and OpenAI-compatible ones.
- One OpenRouter key reaches many vendors through `openrouter/...`, plus ready-made
  presets for DeepSeek, Together, Fireworks, Mistral, xAI, Perplexity, Cerebras, and
  Ollama - each a prefix and its own key. Providers install as extras, e.g.
  `pip install "nfield[google]"`.

**Grounding, provenance, and a cost receipt**
- Grounding. Turn on `ground_values` to label each value by how well the source supports
  it and get a `hallucination_rate`. It never drops a value, only labels it, because a
  correct answer is often not word-for-word.
- Provenance. Turn on `provenance` and `result.provenance` gives each value's
  `[start, end)` character span in the source document.
- `save_html(result, document, path)` renders a provenance run as one self-contained HTML
  page: every located value highlighted in the source, field names on hover, and a table
  of exact spans. Stdlib only.
- Token usage and cost on every result. `metadata.tokens_prompt` and `tokens_completion`
  report what the run really spent, from the provider's own counts; set
  `pricing=(input, output)` (USD per million tokens) and `metadata.cost` does the billing
  math. Cache hits add nothing, so a warm rerun reports zero.

**Response caching**
- Set `cache=True` for an in-process cache, or pass a `DiskCache` to keep responses
  between runs; any object with `get`/`set` works as a custom backend. Keyed on the exact
  request, so a hit is always the text the model would have returned. `--cache-dir` adds
  the same on the command line, and `.stats()` reports hits, misses, entries, and size.

**Reliability**
- `fallback_model` escalates fields that still fail after retries to a stronger model,
  once, so only the stragglers pay for it. It also takes a list, tried cheapest-first,
  with each model seeing only what the previous one left.
- Results now carry the reason a run came up short: `metadata.error` holds a
  representative provider failure, and a call-failure count is kept separate from fields
  that are genuinely absent from the document.
- Schema preflight rejects a provably impossible schema (`minimum > maximum`, empty
  `enum`, and the like) with the field and a fix hint before any model is called.
- `closed_book` fills a schema from the model's own knowledge with no document at all, and
  `self_consistency` keeps a value only when two samples agree.

**Command line**
- Every `ExtractionConfig` setting is now a flag, a `batch` command runs a whole directory
  through one reused engine, and `--format` picks JSON, JSON Lines, or CSV output.

### Improved

The biggest body of work in this release is extraction quality on hard, wide, real
documents - especially long and nested arrays, unions, and messy source text.

- **Long and unbounded arrays** are now read to completion across the document instead of
  stopping at the first window: output-truncated arrays are detected and repaired, every
  entry is reached in reading order, and near-duplicate rows are merged away.
- **Nested and scalar arrays** parse and come back in their real shape - arrays of arrays
  stay nested, and a scalar array becomes a clean list instead of being flattened.
- **Unions (`anyOf`)** resolve per document: an array-or-object field keeps the branch the
  document actually filled, object branches are combined, all-null and per-element cases
  are handled, and `additionalProperties` folds back into its object.
- **Segment and geography coverage.** Fields that repeat per segment or per region (think
  a filing's disaggregated tables) are swept exhaustively rather than sampled once.
- **Messy text.** Numbers parse across locale grouping styles, JSON array values are
  hardened and repaired when a model returns them slightly malformed, an empty extracted
  value counts as absent for every type, dates match by calendar value rather than exact
  string, and repeated boilerplate is stripped from list items.
- **Budgeting and retrieval.** Output tokens are budgeted for nested and unbounded arrays,
  excerpt coverage reaches every region of the document, excerpt size is capped against a
  tokenizer that undercounts, and a boundary pass reads the document's head and tail where
  hard-to-find fields hide.
- **Recovery** keeps the fuller value when a re-extraction finds nothing better, and drops
  values that turn out to be document furniture rather than real content.
- **Provider request timeouts** scale with the number of tokens a call actually books, so
  a large output is not cut off early.

### Fixed
- Correct the Groq default context window.
- Record the provider call-failure reason during assembly so a failed run explains itself.

### Documentation and benchmarks
- A full documentation site: a configuration guide, CLI, errors, and examples pages, an
  API reference, and a runnable notebook suite.
- A benchmark suite behind the `bench` extra - ExtractBench, FinTagging, field-count
  scaling, and head-to-head figures - for reproducing the numbers in the README.

## [0.1.0] - 2026-06-27

First public release.

### Added
- Extract hundreds of structured fields from a document. nfield splits a wide schema
  into groups that fit the model, retrieves the relevant part of the document for each,
  validates every value against the text, retries the ones that fail, and reassembles
  nested JSON.
- Schemas as a JSON Schema dict, a Pydantic model, or a dataclass.
- Sync and async API: `nfield`, `nfield_async`, `NField`, `AsyncNField`, plus
  `extract_batch` for running many documents through one reused engine.
- Groq and OpenAI-compatible providers. The OpenAI provider's `base_url` reaches any
  compatible endpoint, hosted (OpenAI, Together, Fireworks, DeepSeek) or local
  (Ollama, vLLM, LM Studio), so documents can stay on your machine.
- Reasoning-model support: `ExtractionConfig(reasoning_model=True)` turns off a model's
  thinking pass per call so it does not eat the answer's output budget.
- Command-line interface (`nfield extract`, `nfield inspect`) under the `[cli]` extra.
- Tabular export to pandas or CSV under the `[export]` extra.
- Fully typed (`py.typed`, `mypy --strict`) with no required dependencies; a provider SDK
  is pulled in only when you use it.
- Documentation site and runnable examples.

[Unreleased]: https://github.com/nfield-labs/nfield/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/nfield-labs/nfield/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/nfield-labs/nfield/releases/tag/v0.1.0
