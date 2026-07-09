# Command-line interface

Install the CLI extra and the `nfield` command is on your path:

```bash
pip install "nfield[cli]"
pip install "nfield[cli,export]"     # add CSV output
```

There are three commands: [`inspect`](#nfield-inspect) analyses a schema offline,
[`extract`](#nfield-extract) runs one document, and [`batch`](#nfield-batch) runs many. The API
key works exactly as it does in the library ([Configuration](configuration.md#setting-your-api-key)):

```bash
export GROQ_API_KEY="gsk_..."
```

## `nfield inspect`

Analyse a schema **offline**: field count, type breakdown, and a minimum-call (`K_min`)
estimate. Makes no API calls, so it is free and instant.

```console
$ nfield inspect schema.json
Total leaf fields : 5
K_min estimate    : 1  (M_O=8192, English NSL)
Field types:
  boolean    1
  number     1
  string     3
Paths:
  vendor
  invoice_number
  total
  currency
  paid
```

`K_min` depends on the model's output ceiling; pass `--max-output-tokens` for your model to get
a faithful estimate (it defaults to a conservative 8K).

## `nfield extract`

Extract one document. JSON goes to stdout (or `--output`); a run summary and any warnings go to
stderr, so piping stdout stays clean.

```console
$ nfield extract invoice.txt -s schema.json -m groq/llama-3.1-8b-instant
{
  "vendor": "Acme Corporation",
  "invoice_number": "4471",
  "total": 1284.5,
  "currency": "USD",
  "paid": true
}
```

With `--show-metadata`, a compact run summary is written to **stderr**:

```console
$ nfield extract invoice.txt -s schema.json -m groq/llama-3.1-8b-instant --show-metadata
# stdout: the JSON above
# stderr:
status=success quality=1.000 K=1/1 (gap 0.00)
fields: 5/5 extracted, 0 missing, 0 call-failed, 0 retry rounds
```

### Core options

| Flag | Meaning |
|------|---------|
| `document` | Path to the source document (positional). In closed-book mode, pass an empty file. |
| `-s`, `--schema` | Path to a JSON Schema file (required). |
| `-m`, `--model` | `provider/model-name`. Falls back to `$NFIELD_MODEL` when omitted. |
| `-o`, `--output` | Write here instead of stdout. |
| `-f`, `--format` | Output format: `json` (data), `jsonl` (full result envelope), or `csv`. |
| `--show-metadata` | Print a run summary (status, quality, K, grounding) to stderr. |
| `--instructions` | Extra steering for the model, prepended to the prompt. |

### Connection options

| Flag | Meaning |
|------|---------|
| `--api-key` | Provider API key. Prefer the provider env var; pass only for vault use. |
| `--base-url` | Override the provider API base URL (proxy / gateway / self-hosted). |
| `--context-window` | The model's real context window in tokens. |
| `--max-output-tokens` | The model's real output ceiling in tokens. |

### Config flags

Every [`ExtractionConfig`](configuration.md#extractionconfig) setting is a flag, grouped into
panels in `nfield extract --help`. Boolean settings are negatable (`--ground-values` /
`--no-ground-values`); unset flags inherit the library default.

| Panel | Example flags |
|-------|---------------|
| Grounding and provenance | `--ground-values`, `--grounding-min-score`, `--provenance` |
| Reliability and recovery | `--max-api-retries`, `--fallback-model`, `--validate-schema/--no-validate-schema` |
| Extraction tuning | `--max-fields-per-call`, `--max-concurrent-calls`, `--z-target`, `--confidence TIER=SCORE` |
| Closed-book and knowledge | `--closed-book`, `--knowledge-fallback`, `--self-consistency` |

Run `nfield extract --help` for the complete, panelled list.

### Output formats

```bash
nfield extract doc.txt -s schema.json -m groq/llama-3.1-8b-instant -f json    # data only (default)
nfield extract doc.txt -s schema.json -m groq/llama-3.1-8b-instant -f jsonl   # full result envelope
nfield extract doc.txt -s schema.json -m groq/llama-3.1-8b-instant -f csv     # one CSV row (needs [export])
```

`jsonl` emits the complete result (data, metadata, status), which round-trips with
`nfield.load_results`.

## `nfield batch`

Extract every document in a directory (or an explicit list of files) with **one reused,
calibrated engine**, streaming the results to JSON Lines in input order.

```console
$ nfield batch ./docs -s schema.json -m groq/llama-3.1-8b-instant -o out.jsonl --show-metadata
# stderr:
Wrote 2 results to out.jsonl
[inv1.txt] status=success quality=1.000 K=1/1 (gap 0.00)
[inv1.txt] fields: 5/5 extracted, 0 missing, 0 call-failed, 0 retry rounds
[inv2.txt] status=success quality=1.000 K=1/1 (gap 0.00)
[inv2.txt] fields: 5/5 extracted, 0 missing, 0 call-failed, 0 retry rounds
```

| Flag | Meaning |
|------|---------|
| `inputs` | Document files and/or directories to scan (positional, one or more). |
| `-s`, `--schema` | Path to a JSON Schema file (required). |
| `-m`, `--model` | `provider/model-name`. Falls back to `$NFIELD_MODEL`. |
| `-o`, `--output` | Write results here instead of stdout. |
| `-f`, `--format` | `jsonl` (default, streams best), `json` (array), or `csv`. |
| `--pattern` | Glob applied inside directory inputs (default `*.txt`). |
| `--max-concurrent` | Documents extracted in parallel. |
| `--show-metadata` | Print a per-document run summary to stderr. |

Batch also accepts `--api-key`, `--base-url`, `--context-window`, `--max-output-tokens`,
`--instructions`, and the common opt-ins (`--ground-values`, `--provenance`, `--reasoning-model`,
`--knowledge-fallback`, `--fallback-model`, `--max-retry-rounds`). For per-field fine tuning,
extract documents individually with `extract`.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | The run completed; results were written. |
| `1` | A bad input (missing file, invalid JSON schema), or an API/call failure left the result incomplete (`fields_call_failed > 0`). |

A non-zero exit on call failure lets a script tell an incomplete run apart from a document that
genuinely held nothing to extract. Output is still written before exiting, so no data is lost.

## See also

- [Configuration](configuration.md) - the settings behind every flag.
- [Examples](examples.md) - end-to-end recipes.
