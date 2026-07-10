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
  Ollama - each a prefix and its own key, no extra code.

**Grounded, and now with a receipt**
- Grounding and provenance. Turn on `ground_values` to score how well the document
  supports each value and get a `hallucination_rate`; turn on `provenance` to get the
  exact `[start, end)` character span each value came from. Grounding never drops a
  value, only labels it.
- Token usage and cost on every result. `metadata.tokens_prompt` and `tokens_completion`
  report what the run really spent, from the provider's own counts; set
  `pricing=(input, output)` (USD per million tokens) and `metadata.cost` does the billing
  math. Cache hits add nothing, so a warm rerun reports zero.
- `save_html(result, document, path)` renders a provenance run as one self-contained HTML
  page: every located value highlighted in the source, field names on hover, and a table
  of exact spans. Stdlib only.

**Cheaper repeat runs**
- Response caching. Set `cache=True` for an in-process cache, or pass a `DiskCache` to
  keep responses between runs; any object with `get`/`set` works as a custom backend.
  Keyed on the exact request, so a hit is always the text the model would have returned.
  `--cache-dir` adds the same on the command line.
- `.stats()` on both caches: hits, misses, entries, and on-disk size.

**Reliability and reach**
- `fallback_model` escalates fields that still fail after retries to a stronger model,
  once, so only the stragglers pay for the bigger model. It also takes a list, tried in
  order - cheapest first - with each model seeing only what the previous one left.
- Schema preflight: an impossible schema (`minimum > maximum`, empty `enum`, and the
  like) raises `SchemaError` with the field and a fix hint before any model is called.
- `closed_book` fills a schema from the model's own knowledge with no document at all,
  and `self_consistency` keeps a value only when two samples agree.
- A `batch` command in the CLI runs a whole directory through one reused engine.

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
