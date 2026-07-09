# nfield-bench

Two questions, one benchmark suite: can grounded extraction hold up on documents that defeat
frontier models, and does it keep holding up as the schema grows from a handful of fields to
thousands?

Every run reports two numbers, always together:

- **Coverage** - the share of schema fields that get filled instead of left empty.
- **Value accuracy** - of the gold fields, the share whose value is correct (a type-aware
  match against a gold answer key).

The benchmark lives in the repo so it is reproducible with the code it measures. It is not
part of the published wheel.

## ExtractBench

[ExtractBench](https://github.com/ContextualAI/extract-bench) is Contextual AI's benchmark for
complex structured extraction (Ferguson et al., [arXiv:2602.12247](https://arxiv.org/abs/2602.12247)).
Five domains, real documents, human-checked gold, and per-field judging tiers that range from
exact match to an LLM judge for free text. It is deliberately hard: the schemas are wide, the
documents run to dozens of pages, and the values are scattered across tables and prose.

We vendored the dataset under `benchmark/external/extract-bench/` and run nfield over it
unchanged. The documents, schemas, and gold are theirs; the extraction and the scoring code
are ours.

### The result

One model, one provider, one run: **`qwen/3.6-27b` on Groq**, 2026-07-06.

| Domain (docs / pages: keys) | Coverage | Value accuracy | Judged |
|-----------------------------|---------:|---------------:|-------:|
| Research (6 / 42pg: 16)     |    96.8% |          95.5% |  95.9% |
| Credit (10 / 137pg: 13)     |    94.3% |          82.3% |  86.4% |
| Resumes (7 / 3pg: 31)       |    87.9% |          81.6% |  87.3% |
| Sports (5 / 3pg: 12)        |    99.0% |          98.6% |  98.8% |
| SEC 10-K/Q (7 / 60pg: 369)  |    91.2% |          81.1% |  85.4% |
| **Field-weighted overall**  |  **92%** |      **84.2%** | **87.7%** |

*Value accuracy* is the strict deterministic score. *Judged* re-scores the fields the paper
marks for LLM judging (semantic strings, unordered arrays) with a judge, so a currency written
`$` instead of `USD`, or a date as `March 2019` instead of an ISO timestamp, counts as the
match it is. Both are shown so the strict floor is never hidden.

### Why this is worth showing

The paper's Table 5 runs six frontier models on the same data. Their field-level pass rate:

| Domain      | Best single model | Six-model aggregate |
|-------------|------------------:|--------------------:|
| Research    | 49.0% (Sonnet 4.5) |               20.8% |
| Credit      |      86.9% (GPT-5) |               56.3% |
| Resumes     |    24.0% (Opus 4.5) |               18.4% |
| Sports      | 18.3% (Gemini 3 Flash) |            12.5% |
| SEC 10-K/Q  |          **0.0%** |            **0.0%** |
| Overall     | 6.9% (Gemini 3 Flash) |             4.6% |

On SEC 10-K/Q, the 369-key financial filings, **every** model in the paper scored zero: Gemini
3 Pro, GPT-5, Opus 4.5, all of them. The schema is too wide to emit in one response, so the
JSON truncates and the whole document fails. nfield extracts the same filings at 81-85% because
it never asks the model to emit the whole schema at once: it splits a wide schema into bounded
calls, each returning a slice that fits, and reassembles the result. The best frontier model in
the paper reaches 6.9% overall; nfield with a 27B open model reaches 88%.

Two things carry it. The output format is `field.path = value` lines, not a single JSON blob,
so a run cannot fail on a truncated brace (valid, parseable output by construction; zero
structural errors across all 35 documents). And the schema decomposition keeps every call
inside the model's real context and output budget, so completeness does not collapse as the
field count climbs into the hundreds.

The comparison is directional, not a claim of an identical harness: the paper reports a
document-gated pass rate, we report field-level coverage and value accuracy, and we score every
flattened gold field including each array element (8,830 fields on SEC 10-K/Q against the
paper's 2,583 schema keys), a larger and stricter denominator. The gap is too wide to be an
artifact of that difference.

### Accuracy vs field count

There is a companion picture to the table: plot each ExtractBench document as a point, its gold
field count on the x-axis and nfield's value accuracy on the y-axis, and lay the published
IFScale curves beside it. IFScale ([arXiv:2507.11538](https://arxiv.org/abs/2507.11538)) measures
how well a single call follows instructions as the count climbs from 10 to 500; even the best
frontier model there falls to 68% by 500, and the curve stops. nfield's points keep going: the
same accuracy band holds out past a thousand fields, where every single-call curve has already
ended.

```
python -m benchmark.figures.fieldcount benchmark/results/<a-run-dir> [--judged]
```

It reads the per-document `scored/` files a run already wrote (no API calls) and saves the chart
under `<run>/ifscale/`. On the 2026-07-06 qwen run the mean is 0.89 below 500 fields and 0.83 at
500 and above: essentially flat where the reference curves are in free fall. The two axes are not
the same task, so read it as the shape of the story, not a matched head-to-head; the reference
numbers are transcribed with their source in `benchmark/reference.py`.

## FinTagging

[FinTagging](https://arxiv.org/abs/2505.20650) asks a model to pull every numeric fact from a
financial filing's tables and tag each with its XBRL type. We take the real XBRL contexts from
its FinNI split (`TheFinAI/FinNI-eval`), concatenate them into one wide document, and run nfield
and a single call on the same model and budget: **`qwen/3.6-27b` on Groq**, 2026-07-08. The
metric is the paper's pair-level (fact, type) F1.

| Tables | Distinct facts | nfield F1 | Recall | Single-call F1 |
|-------:|---------------:|----------:|-------:|---------------:|
| 1  |   292 | 0.965 | 1.000 | 0.965 |
| 3  |   727 | 0.984 | 1.000 | 0.581 |
| 6  | 1,138 | **0.991** | 0.999 | 0.160 |
| 10 | 1,474 | 0.981 | 0.999 | 0.456 |
| 15 | 1,970 | 0.988 | 0.998 | 0.389 |

nfield recovers every distinct fact the filing states and holds F1 above 0.96 out to nearly two
thousand facts. A single call keeps pace only on the smallest document; once the answer overruns
one response it falls to 0.16-0.58, the same wide-output wall as SEC 10-K/Q above. The paper's
best model reaches 0.72. Same cause, same fix: nfield splits the extraction into bounded calls
and reassembles, so the output never truncates.

## How nfield holds up as the schema grows

Beyond ExtractBench, the suite stress-tests the one thing the method exists for: staying
complete as the schema widens. On `groq/llama-3.3-70b-versatile` (2026-06-21), one run per row,
each at the model's real context and output limits:

| Fields | Document | Coverage | Value accuracy | Calls (minimum) |
|-------:|----------|---------:|---------------:|----------------:|
| 304 | clinical-trial record | 93% | 89% | 10 (9) |
| 335 | country factbook | 100% | 91% | 10 (10) |
| 1045 | multi-country factbook | 100% | 79% | 29 (29) |

Coverage stays high as the field count climbs past a thousand, and the number of model calls
tracks the computed minimum rather than exploding.

## Scaling to thousands of fields

nfield is built for wide schemas, so the benchmark also runs synthetic schemas far past what a
single call can hold. Filling every field and keeping the call count near its minimum is the
system's job (value accuracy is the model's), so the scale runs report coverage and calls:

| Fields | Coverage | Calls (minimum) |
|-------:|---------:|----------------:|
| 2,523 | 100% | 62 (61) |
| 4,000 | 100% | 95 (94) |
| 5,641 | ~100% | 127 (124) |

The schema is split exactly as much as the budget requires, so the call count stays within a
couple of the computed minimum with no call storm, even at 5,641 fields.

## How it stays honest

- **One model, named and dated.** Model behaviour drifts, so every number is tied to a model
  version and a date. No averaging across models, no "best" claim.
- **Coverage and value accuracy are always shown together**, so "filled but wrong" cannot hide
  behind a high coverage number.
- **The strict floor and the judged score are both reported**, and the judge can only confirm a
  field the exact rule missed, never remove a correct one.
- **A failed call scores zero**, not dropped: a refusal or truncation on a hard schema is part
  of what the benchmark measures.
- **Reproducible.** Every result directory ships a `MANIFEST.json` (model, date, settings,
  library versions) and the raw per-document outputs, so any number can be re-scored offline.

## Running it

The scorer and offline paths need no API key. The live sweeps cost real calls and are run by
hand.

```bash
# ExtractBench: run nfield over one domain (needs GROQ_API_KEY):
uv run python -m benchmark.benchmarks.runner_extractbench --datasets sport_swimming

# Re-score a finished run under the paper's LLM-judge tiers (small judge cost):
uv run python -m benchmark.scoring.rejudge_extractbench results/<run>/native

# Field-count scaling sweep:
uv run python -m benchmark.runners.runner run --method nfield --fixture clinicaltrial --seeds 1
```

## Results

Results are grouped by benchmark: `results/runners/` (the wide-schema sweeps), `results/extractbench/`,
`results/fintagging/`, and `results/head2head/`. Each run directory holds raw outputs, scored
aggregates, a `summary.csv`, and a `MANIFEST.json`, all committed so a run can be reproduced and checked. The
ExtractBench documents under `external/extract-bench/` are vendored from Contextual AI's public
repository; the field-count fixtures under `datasets/real/` are public-domain sources. No
credentials are stored in the repo (the key is read from a gitignored `.env`).
