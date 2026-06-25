# Benchmark datasets

Each fixture is a self-contained directory of committed files. A gold answer key
corresponds to the exact committed document snapshot — that pairing is what makes
a Value Accuracy number reproducible, so every document is committed.

```
real/<name>/
  schema.json     target JSON Schema                        (committed)
  gold.json       flat gold answer key                      (committed; absent => coverage-only)
  document.txt    source document                           (committed)
```

## `real/` — document-extraction fixtures

The model reads the document and fills the schema from it.

| Fixture | Gold fields (N) | Scorable | Notes |
|---|---|---|---|
| `chemical_element` | 20 | yes | Periodic-table element profile |
| `smartphone_spec` | 111 | yes | Smartphone specification sheet |
| `clinicaltrial` | 304 | yes | ClinicalTrials.gov record, typed/nested |
| `factbook_us` | 335 | yes | CIA World Factbook, single country |
| `factbook_multi` | 1045 | yes | Factbook, multi-country (the N≈1000 anchor) |
| `clinical_registry` | 2523 | yes | Multi-record clinical registry (scale) |
| `cre_rent_roll` | 4000 | yes | Commercial-real-estate rent roll (scale) |
| `financial_consolidation` | 5641 | yes | Multi-entity financial consolidation (scale) |
| `war_and_peace` | — | no (coverage-only) | 3.3 MB public-domain Gutenberg text |

The large fixtures (2.5k–5.6k fields) are the scale track. `war_and_peace` has no
gold key, so it measures coverage only, not Value Accuracy.

## `closed_book/` — knowledge fixtures (no document)

Each method fills the schema from the model's own knowledge; `document.txt` is
empty. The subject lives in each gold path (e.g. `Carbon.symbol`), so one
instruction per domain selects it.

| Fixture | Gold fields (N) | Domain |
|---|---|---|
| `elements_59` | 59 | Periodic table |
| `countries_205` | 205 | Country reference facts |
| `pokemon_600` | 600 | Pokedex stats |
| `airports_1002` | 1002 | Airports by IATA code |

`build_fixtures.py` regenerates these folders from four public reference dumps
(URLs documented in the script). The raw dumps are gitignored; the generated
fixture folders are committed.

## Registry

The map from a fixture name to its loader and domain instructions lives one level
up in `benchmark/datasets.py` (`get` for `real/`, `get_closed_book` for
`closed_book/`).
