# Oracle v3 Migration Guide

## What Changed in v0.3

FormatShield v0.3 replaces the sklearn `CalibratedClassifierCV` oracle with a closed-form
information-theoretic routing score Φ(prompt, schema). No model artifacts, no benchmark CSV
files, and no training runs are required.

## Removed APIs

The following methods now raise `DeprecationWarning` + `NotImplementedError`:

| Method | Replacement |
|---|---|
| `ThresholdOracle.from_benchmark_data(csv_path)` | No replacement — training not required |
| `ThresholdOracle.save(path)` | No replacement — no artifact to save |
| `ThresholdOracle.load(path)` | No replacement — no artifact to load |
| `OracleX.from_benchmark_data(...)` | No replacement |
| `OracleX.update_online(...)` | No replacement |

## Migration Steps

**Before (v0.2):**
```python
# Required: run benchmark, train, save pkl
oracle = ThresholdOracle.from_benchmark_data("benchmark_results/summary.csv", save=True)
```

**After (v0.3):**
```python
# No setup required — routing is active immediately
from formatshield import FormatShield
shield = FormatShield(backend=my_backend)  # Φ routing active out of the box
```

## Why No Training?

The Φ formula derives routing decisions purely from the structure of the prompt and schema:

- **λ̃₂** — algebraic connectivity of the schema dependency graph (spectral graph theory)
- **τ** — schema constraint tightness (information entropy of the type tree)
- **ΔK** — Normalized Compression Distance between prompt and schema (Kolmogorov complexity proxy)

These three components capture the essential signals that determine whether TTF is beneficial,
without requiring any labeled training data.

## Dependency Changes

`scikit-learn` and `joblib` are no longer required and have been removed from the default
dependencies. If you had them pinned in your own `requirements.txt` for FormatShield's oracle,
they can be removed.
