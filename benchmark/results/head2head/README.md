# nfield vs single-call extraction libraries on a wide 10-Q

Four runs here compare nfield against four other libraries on one real 10-Q filing
from ExtractBench: native JSON mode, instructor, LangChain, and LangStruct. Every
method gets the same document, the same schema, the same output budget, and is
graded by the same scorer. The only thing that changes is how each one asks the
model. nfield decomposes the schema into bounded sub-calls; the others send the
whole schema in a single call.

## Results

Value accuracy per gold field, same model within each group.

| Method       | llama-3.3-70b | qwen3.6-27b |
|--------------|--------------:|------------:|
| nfield       | 0.54          | 0.82        |
| instructor   | 0.12          | 0.37        |
| native_json  | 0.00          | 0.22        |
| langchain    | 0.00          | 0.22        |
| langstruct   | 0.02          | 0.01        |

nfield wins on both models. A single call returns the top-level shape of the filing
but almost none of its ~1,300 line-item values; nfield recovers the bulk of them and
adds nothing the document does not state (zero hallucinated fields).

## Layout

- `combined_nke_llama_qwen/summary_grouped.png`: the chart for the table above (Nike
  10-Q, 1,328 gold fields). This is the headline run.
- `combined_llama_qwen/summary_grouped.png`: an earlier run on a Cisco 10-Q (1,407
  gold fields), same shape of result.
- `groq-<model>_<stamp>/`: one folder per model+document, with `summary.csv`,
  `summary.md`, `summary.png`, `MANIFEST.json`, and the per-method raw and scored
  output for anyone who wants to check a number by hand.

## Notes

- A method that errored or refused the schema is scored as a miss on every gold
  field, never dropped from the count.
- qwen3.6-27b is a reasoning model. nfield strips the reasoning trace; the JSON-mode
  baselines avoid it because a forced JSON response leaves no room for one, so their
  scores reflect extraction, not reasoning. llama-3.3-70b is non-reasoning, so that
  question does not arise there.
- Numbers are per-field accuracy under the shared scorer, not a whole-document pass
  rate. Reproduce a run with `uv run python -m benchmark.benchmarks.head2head --judge`.
