# FinTagging FinNI wide head-to-head

One sweep, both methods, one model (groq/qwen/qwen3.6-27b), identical 24k output
budget. Data: real FinNI-eval contexts (TheFinAI, arXiv:2505.20650). The
multi-table concatenation is ours; the tables, gold, and pair-level metric are
the benchmark's.

## Distinct-pair F1 (the fair metric for a de-duplicating extractor)

| Tables | Distinct gold | NField F1 | NField P | NField R | Baseline F1 |
|---:|---:|---:|---:|---:|---:|
| 1  | 292  | 0.965 | 0.933 | 1.000 | 0.965 |
| 3  | 727  | 0.984 | 0.968 | 1.000 | 0.581 |
| 6  | 1138 | 0.991 | 0.983 | 0.999 | 0.160 |
| 10 | 1474 | 0.981 | 0.963 | 0.999 | 0.456 |
| 15 | 1970 | 0.988 | 0.980 | 0.998 | 0.389 |

NField recovers every distinct fact the filing states (recall approx. 1.000 at
every size) and holds F1 0.965-0.991 as the answer grows to 2,582 facts; a single
call collapses once the answer overruns one response.

## Paper multiset metric

| Tables | Gold facts | NField F1 | NField R | Calls | Baseline F1 | Baseline R |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 322  | 0.968 | 1.000 | 2 | 0.965 | 0.994 |
| 3  | 768  | 0.852 | 0.997 | 2 | 0.581 | 0.421 |
| 6  | 1326 | 0.950 | 0.973 | 2 | 0.160 | 0.087 |
| 10 | 1936 | 0.886 | 0.841 | 7 | 0.456 | 0.298 |
| 15 | 2582 | 0.875 | 0.820 | 9 | 0.389 | 0.243 |

The multiset metric counts every repeated occurrence of a value (2,582 instances
vs 1,970 distinct at 15 tables). NField de-duplicates recurring rows across its
overlapping windows, so it emits a recurring figure fewer times than the multiset
rewards; the distinct-pair metric above is the honest measure of what it recovers.

`summary.csv` carries both metrics per size.
