# How the pipeline works

Every extraction runs the same sequence of steps over a shared state object. The trick is
that almost all of them are plain local computation. Only the extraction and retry steps call
the model, so the LLM is asked to do one thing: read the relevant text and fill in values.

| Step | What it does | Calls the model? |
|------|--------------|:----------------:|
| Calibrate | Read the model's context and output limits; estimate characters per token. | no |
| Analyze schema | Flatten the schema to dot-notation paths and estimate each field's token cost. | no |
| Group | Group the fields by where they sit in the schema. | no |
| Retrieve | Chunk the document and score each chunk against each group with lexical retrieval (BMX, a BM25 variant). | no |
| Pack | Split the groups into calls that fit the context and output budget. | no |
| Excerpt | For each call, gather, dedup, and trim the document spans it needs. | no |
| Extract | Build the prompt, call the model, and parse the `key = value` reply. | **yes** |
| Validate and retry | Check every value against its schema and re-extract only the fields that failed. | **yes** (only to fix failures) |
| Assemble | Rebuild the flat pairs into nested JSON and score the result. | no |

The code numbers these stages S0 through S6 (the planning work, grouping through packing, is
all stage S2).

## Why split the work

A single call cannot reliably fill a very wide schema. The packing step works out how many
calls are actually needed from the model's real limits: it computes a lower bound (`K_min`)
and splits the schema only as much as the budget requires. A small schema that fits one call
stays one call; a thousand-field schema becomes many, and the number of calls tracks that
minimum instead of ballooning.

## Calibration never calls the model

The first step is local. It reads the context window and output limit straight from the
provider, and gets characters-per-token from a fast estimate based on the document's language
(or an exact value you pass in `ExtractionConfig`). Nothing is measured against a live API, so
calibration costs no tokens and gives the same answer every time.
