# Head-to-head: 10-Q, 1407 scored gold values, same model, same document

Model `groq/qwen/qwen3.6-27b`, a reasoning model: nfield strips its reasoning trace, the competitor libraries see the raw completion (json-mode baselines avoid it via forced JSON).
Every method scored with the one ExtractBench scorer. A failed run stays a miss.

| Method | Value accuracy | Judged | Coverage | K | Fail |
|---|---:|---:|---:|---:|---|
| nfield | 0.744 | 0.763 | 0.845 | 41 | - |
| native_json | 0.201 | 0.222 | 0.233 | 1 | - |
| langchain | 0.201 | 0.225 | 0.233 | 1 | - |
| instructor | 0.189 | 0.235 | 0.193 | 1 | - |
| langstruct | 0.001 | 0.001 | 0.002 | 1 | - |
