# Head-to-head: 10-Q, 1407 scored gold values, same model, same document

Model `groq/llama-3.3-70b-versatile`, non-thinking, so no method gains on reasoning tokens.
Every method scored with the one ExtractBench scorer. A failed run stays a miss.

| Method | Value accuracy | Judged | Coverage | K | Fail |
|---|---:|---:|---:|---:|---|
| nfield | 0.530 | 0.618 | 0.645 | 31 | - |
| native_json | 0.001 | 0.001 | 0.002 | 1 | - |
| langchain | 0.001 | 0.001 | 0.002 | 1 | - |
| langstruct | 0.001 | 0.001 | 0.007 | 1 | - |
| instructor | 0.000 | 0.000 | 0.000 | 0 | schema_validation_failed |
