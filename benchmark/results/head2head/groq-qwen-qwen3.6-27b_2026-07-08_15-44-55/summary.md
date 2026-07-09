# Head-to-head: 10-Q, 1328 scored gold values, same model, same document

Model `groq/qwen/qwen3.6-27b`, a reasoning model: nfield strips its reasoning trace, the competitor libraries see the raw completion (json-mode baselines avoid it via forced JSON).
Every method scored with the one ExtractBench scorer. A failed run stays a miss.

| Method | Value accuracy | Judged | Coverage | K | Fail |
|---|---:|---:|---:|---:|---|
| nfield | 0.824 | 0.843 | 0.961 | 25 | - |
| instructor | 0.367 | 0.472 | 0.404 | 1 | - |
| native_json | 0.221 | 0.307 | 0.233 | 1 | - |
| langchain | 0.221 | 0.303 | 0.233 | 1 | - |
| langstruct | 0.014 | 0.089 | 0.021 | 1 | - |
