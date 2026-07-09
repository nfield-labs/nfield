# Head-to-head: 10-Q, 1328 scored gold values, same model, same document

Model `groq/llama-3.3-70b-versatile`, non-thinking, so no method gains on reasoning tokens.
Every method scored with the one ExtractBench scorer. A failed run stays a miss.

| Method | Value accuracy | Judged | Coverage | K | Fail |
|---|---:|---:|---:|---:|---|
| nfield | 0.544 | 0.639 | 0.680 | 25 | - |
| instructor | 0.123 | 0.221 | 0.176 | 1 | - |
| langstruct | 0.015 | 0.081 | 0.021 | 1 | - |
| native_json | 0.002 | 0.002 | 0.002 | 1 | - |
| langchain | 0.002 | 0.002 | 0.002 | 1 | - |
