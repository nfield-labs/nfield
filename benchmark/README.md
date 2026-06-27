# nfield-bench

A field-count scaling benchmark: how well does grounded extraction hold up as the schema
grows from a handful of fields to thousands?

Each run reports two numbers, always together:

- **Coverage** - the share of schema fields that get filled instead of left empty.
- **Value accuracy** - of the filled fields, the share whose value is correct (a type-aware
  match against a gold key).

The benchmark lives in the repo so it is reproducible with the code it measures. It is not
part of the published wheel.

## How nfield holds up as the schema grows

On `groq/llama-3.3-70b-versatile` (2026-06-21), one run per row, each at the model's real
context and output limits:

| Fields | Document | Coverage | Value accuracy | Calls (minimum) |
|-------:|----------|---------:|---------------:|----------------:|
| 304 | clinical-trial record | 93% | 89% | 10 (9) |
| 335 | country factbook | 100% | 91% | 10 (10) |
| 1045 | multi-country factbook | 100% | 79% | 29 (29) |

Coverage stays high as the field count climbs past a thousand, and the number of model calls
tracks the computed minimum rather than exploding. These are point estimates for one model on
one date; value accuracy depends on the model you choose, so the benchmark characterises the
method, not a ranking.

## Scaling to thousands of fields

nfield is built for wide schemas, so the benchmark also runs synthetic schemas far past what
a single call can hold. Filling every field and keeping the call count near its minimum is the
system's job (value accuracy is the model's), so the scale runs report coverage and calls:

| Fields | Coverage | Calls (minimum) |
|-------:|---------:|----------------:|
| 2,523 | 100% | 61 (61) |
| 4,000 | 100% | 95 (94) |
| 5,641 | 100% | 126 (124) |

The schema is split exactly as much as the budget requires, so the call count stays within a
couple of the computed minimum with no call storm, even at 5,641 fields.

## How it stays honest

- **One model, named and dated.** Model behaviour drifts, so every number is tied to a model
  version and a date. No averaging across models, no "best" claim.
- **Coverage and value accuracy are always shown together**, so "filled but wrong" cannot
  hide behind a high coverage number.
- **A failed call scores zero**, not dropped: a refusal or truncation on a hard schema is
  part of what the benchmark measures.
- **Reproducible.** Every result directory ships a `MANIFEST.json` (model, date, seed, prompt
  hash, library versions) and the raw per-record outputs.

## Running it

The scorer and offline paths need no API key. The live sweeps cost real calls and are run by
hand:

```bash
# Extract one fixture and score it (needs GROQ_API_KEY):
uv run python -m benchmark.runner run --method nfield --fixture clinicaltrial --seeds 1

# Re-score an existing run offline (no API):
uv run python -m benchmark.runner score --method nfield --fixture clinicaltrial

# Aggregate a result directory into a table:
uv run python -m benchmark.report results/<model>_<date>
```

## Results

Results live under `results/<model>_<date>/` with raw outputs, scored aggregates,
`summary.csv`, and a `MANIFEST.json`, all committed so a run can be reproduced and checked.
The documents under `datasets/real/` are public-domain sources, and no credentials are stored
in the repo (the key is read from a gitignored `.env`).
