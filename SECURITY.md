# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a security problem.

Report it privately through GitHub's
[security advisory form](https://github.com/nfield-labs/nfield/security/advisories/new). Include what you found, how to reproduce it, and the impact
you expect. We aim to acknowledge a report within a few days and will keep you updated while
we work on a fix.

## Supported versions

nfield is pre-1.0. Security fixes land on the latest release; please upgrade before
reporting an issue you hit on an older version.

## What nfield does with your data

nfield is middleware between your document and an LLM provider. Two things are worth knowing:

- **Your document text is sent to the provider you choose.** For sensitive documents, point
  nfield at a local model instead of a hosted API by setting a `base_url` on the OpenAI
  provider (Ollama, vLLM, or LM Studio), so the text never leaves your machine.
- **API keys are read from the environment**, never from the document or the schema. nfield
  reads each provider's standard variable (`GROQ_API_KEY`, `OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, and so on), so you set the key the same way you would for that
  provider's own SDK. You can also pass the key directly in code for a secret vault; it is
  never logged. Keep keys out of source control either way.

## Handling untrusted input

A document is untrusted text that an LLM reads, so the usual LLM cautions apply:

- Treat extracted values as model output, not ground truth. nfield validates every value
  against the document and reports per-field confidence, but a determined prompt-injection
  attempt in the source text can still influence a result. Check anything that drives an
  action downstream.
- A schema can come from an untrusted source too. nfield validates the schema before use and
  raises `SchemaError` on malformed input rather than failing deep in the pipeline.
