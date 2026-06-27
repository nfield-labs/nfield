# nfield-bench

A field-count scaling benchmark for grounded structured extraction.

The x-axis is **N**, the number of schema fields. The y-axis is **grounded
field-level Value Accuracy**. Every published number takes exactly one form:

> *On model M, on date D, at N fields, nfield extracts X% of fields correctly
> where library Y extracts Z%.*

Never "nfield is the best." Never an averaged-over-models number. One model, one
date, one curve.

This folder lives in-repo so the benchmark is reproducible with the code it
measures. It is **not** part of the published `nfield` wheel.

---

## What it measures

| Metric | Role | Source |
|---|---|---|
| **Coverage @ N** | **primary** - fields filled vs left NULL | gold-based recall (`score.py`) |
| **Value Accuracy (VA)** | secondary - are the filled values correct | gold-diff scorer (`score.py`) |
| Per-field-type VA | mandatory breakdown | `score.py` (boolean / enum / int / number / short-string / long-string) |
| JSON Pass Rate | reported, never headline | structural validity (≈100% for nfield by construction) |
| Error decomposition | diagnostic | omission / accuracy / hallucination / structural |
| call-failed | its own category | `Metadata.fields_call_failed` - never counted as a model omission |
| Optimality Gap | efficiency | `(K - K_min) / K_min` |
| Latency | efficiency | `elapsed_seconds` |

The pipeline already emits coverage, K, K_min, and latency. The **only** new
measurement the benchmark adds is the gold-diff scorer that turns a returned
value into a *correct-or-not* judgement (`score.py`).

### Matching rules (type-aware)

| Field type | Rule |
|---|---|
| boolean / enum / integer | exact match (light coercion: `"true"`→bool, `"1,000"`→int) |
| number | numeric match within a relative tolerance (default ≈ exact) |
| short string (id / name / code) | normalised exact - case, whitespace, and diacritics folded |
| long string / free text | bounded edit distance (normalised Levenshtein ≤ 10%) |
| array / list | **ordered positional** match (`item_0` vs `item_0`); a correct set in a different order counts as per-element accuracy errors - a disclosed reorder penalty, following SOB |

## Honest-claims charter

Every number published from this benchmark must satisfy **all** of:

1. **Single model, named with version** (e.g. `groq/llama-3.3-70b-versatile`) -
   no averaging across models.
2. **Dated** - model behaviour drifts; a number without a date is meaningless.
3. **Reproducible** - a `MANIFEST.json` accompanies every result directory with
   model id, date, temperature, seed, prompt hash, library versions, and a cost
   ledger; raw per-record outputs are committed.
4. **Failures count as 0%, not dropped** - a baseline's 400 / timeout / refusal
   is scored as a miss in the denominator. A separately-labelled "valid-only"
   number may also be reported and must be marked a biased sample.
5. **Coverage is the primary metric (fields filled vs left NULL - the recall the
   decomposition/retrieval architecture drives); Value Accuracy is reported
   alongside it as the secondary check that the filled values are also correct.
   Both are always shown together - coverage is never quoted without VA, so
   "filled but wrong" can never hide.** JSON-pass is reported but never a headline.
6. **No cross-substrate comparison.** nfield is orchestration-layer (any API).
   Decoding-layer libraries (Outlines, XGrammar, …) mask logits and need local
   weights; they cannot run on the same hosted model and must never be
   cross-compared on a shared chart without a loud, explicit label.
7. **Unmeasured = labelled ILLUSTRATIVE / PROJECTED.**
8. **Per-field-type breakdown shown** - no hiding long-string / list weakness
   behind a scalar-inflated average.

Claim template:

> *On {model+version} ({date}, temp={t}, {runs} runs, ±{sigma}), nfield achieves
> {X}% field-level Value Accuracy at N={k} fields vs {baseline}'s {Z}%
> (Δ={X−Z}pp). We do not claim this ordering generalises to other models or
> schema families.*

## Baseline fairness

Same base model, same prompt, same token budget for every method. A baseline's
API error / 400 / timeout scores 0% (it stays in the denominator) - refusing a
hard schema is exactly the capability gap being measured. Tune nothing for nfield
that you do not equally tune for the baselines.

**Retry is mechanism-native, capped at one round for every method.** Each method
gets exactly one corrective attempt in its own idiom: nfield one recovery round
(`max_retry_rounds=1`, below its own default of 2), Instructor one re-validation
(`max_retries=1`), and the raw / native single-call methods the one response the
provider returns (a parse failure on that response is the method's result, not a
retried-away artifact). No method gets more corrective budget than another.

Two substrate tracks that never share a chart:

- **Track A - orchestration-layer**, run on the *same* hosted model as nfield
  (raw prompt, native JSON mode, Instructor, LangChain, …). The fair comparison.
- **Track B - decoding-layer**, run on *local weights* (Outlines, XGrammar,
  lm-format-enforcer, …). A separately-labelled appendix, never cross-compared
  with Track A numbers.

## Layout

```
benchmark/
  score.py          gold-diff scorer: type-aware VA + error decomposition (no API)
  budget.py         native vs constrained token-budget profiles
  models.py         per-model native limits (context / output ceilings)
  datasets.py       registry: fixture name -> (schema, document, gold, instructions)
  report.py         aggregate raw/scored -> summary.csv + tables (+ optional plots)
  runner.py         main sweep: method x document fixture -> raw JSON + MANIFEST
  runner2.py        scale sweep: nfield on the large (2.5k-5.6k field) fixtures
  runner3.py        comparison: nfield vs other extraction libraries
  runner4.py        closed-book sweep: every method fills a schema with no document
  adapters/         nfield and the baselines, behind one uniform interface
  datasets/         real/ document fixtures + closed_book/ knowledge fixtures
  results/
    <model>_<date>/
      <native|constrained>/
        raw/<method>_<fixture>.json    indented JSON array (one entry per seed)
        scored/<method>_<fixture>.json coverage_mean (primary) + value_accuracy_mean
      summary.csv
      MANIFEST.json
```

## How to run

The scorer and the synthetic/offline paths need no API and run in CI. The real
single-model sweeps cost real API calls and are **manual, budgeted, dated**
runs - never auto-run the expensive sweep.

```bash
# Run one method over one fixture and score it (needs GROQ_API_KEY, costs calls):
uv run python -m benchmark.runner run --method nfield --fixture clinicaltrial --seeds 1

# Main sweep: the method x document-fixture matrix into one dated result dir:
uv run python -m benchmark.runner sweep \
  --methods nfield,raw_prompt,native_json,instructor,langchain \
  --fixtures clinicaltrial,factbook_us,factbook_multi --seeds 1

# Re-score an already-produced extraction against a gold key (no API):
uv run python -m benchmark.runner score --method nfield --fixture clinicaltrial

# Aggregate a result directory into a table + plot:
uv run python -m benchmark.report results/<model>_<date> --plot
```

Three focused sweeps live alongside the main one, each writing the same dated
result layout:

```bash
uv run python -m benchmark.runner2   # scale: nfield on the 2.5k-5.6k field fixtures
uv run python -m benchmark.runner3   # comparison vs other libraries
uv run python -m benchmark.runner4   # closed-book: fill a schema with no document
```

The baseline and comparison adapters (Instructor, LangChain, ContextGem,
LangExtract, LangStruct) need the `bench` dependency group: `uv sync --group bench`.

## Results are committed (and scrubbed)

Per-record raw outputs, scored aggregates, `summary.csv`, `MANIFEST.json`, and a
per-run README are committed under `results/` - committing the processed results
plus the per-record model responses is the reproducibility norm for open
benchmarks. The one nuance we apply: the `error` field is a short diagnostic, not
a raw SDK dump - it is whitespace-collapsed, provider-org-id-redacted, and length
-bounded before it is written, so no secret or multi-KB event stream is committed
(the full traceback stays on the console at run time). Documents under
`datasets/real/` are public-domain sources; no credentials live in the repo (the
key is read from a gitignored `.env`).

## Status

Committed single-model runs on `groq/llama-3.3-70b-versatile`, one canonical run
per question (see each run's `MANIFEST.json`; all are **1 run per cell**, so treat
the numbers as point estimates, not variance-banded curves):

- **Main sweep** (`…_2026-06-21_13-12-00`) - nfield vs four baselines (raw prompt,
  native JSON, Instructor, LangChain) on N = 304 / 335 / 1045 plus coverage-only
  War & Peace. At **N=1045 the single-call baselines collapse** (truncation /
  context-exceeded → 0.0, or ≤0.55 VA) while **nfield holds ≈0.80 VA at ~1.0
  coverage** with near-minimal calls (K=29=K_min on the native budget). At small N
  a single native-JSON call is competitive on VA.
- **Scale** (`groq-nk-…_2026-06-21_11-16-55`) - nfield on the large fixtures:
  **100% VA at 2,523 and 4,000 fields, 98% at 5,641 fields**, each at near-optimal
  K (no call storm).
- **Competitors** (`…_2026-06-22_02-26-37`, `…_02-40-00`) - nfield vs ContextGem /
  LangExtract / LangStruct; nfield is the only method scoring 1.0 on both
  chemical_element and smartphone_spec.
- **Closed-book** (`…closed-book_2026-06-24_13-17-26`) - every method fills a
  schema from model knowledge (no document) across N = 59 / 205 / 600 / 1002;
  reported with answer-rate and reliability, an honest result where nfield is
  competitive rather than dominant.
