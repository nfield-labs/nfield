# Reference — CLI

FormatShield ships a command-line interface for quick testing, schema-driven generation, and benchmarking without writing Python code.

---

## Installation

The CLI is included with the base install:

```bash
pip install formatshield
```

Verify installation:

```bash
formatshield --version
```

---

## `formatshield generate`

```bash
formatshield generate PROMPT [OPTIONS]
```

Generate structured output from a prompt.

### Arguments

| Argument | Description |
|---|---|
| `PROMPT` | The prompt string. Can be quoted text or `-` to read from stdin |

### Options

| Option | Default | Description |
|---|---|---|
| `--model`, `-m` | `groq/llama-3.1-70b-versatile` | Model identifier in `"provider/model"` format |
| `--schema`, `-s` | None | JSON Schema as a string or `@path/to/schema.json` to load from file |
| `--debug`, `-d` | False | Print routing trace to stderr |
| `--thinking`, `-t` | False | Include thinking text in output |
| `--output`, `-o` | `json` | Output format: `json`, `pretty`, `raw` |
| `--latency-budget` | None | Maximum latency budget in milliseconds |
| `--backend` | None | Override backend (overrides model prefix inference) |
| `--base-url` | None | Override backend base URL |
| `--api-key` | None | Override API key (prefer env vars for security) |

### Examples

```bash
# Simple generation, no schema
formatshield generate "What is the capital of France?"

# With a schema
formatshield generate "Analyze sentiment: I love this product!" \
  --model groq/llama-3.1-70b-versatile \
  --schema '{"type": "object", "properties": {"label": {"type": "string"}, "score": {"type": "number"}}, "required": ["label", "score"]}'

# Load schema from file
formatshield generate "Extract entities from: Apple hired Tim Cook in 2011." \
  --schema @schemas/ner_schema.json \
  --debug

# Read prompt from stdin
echo "What is the boiling point of water?" | formatshield generate - \
  --model ollama/llama3.1

# With latency budget (suppress TTF if too slow)
formatshield generate "Complex reasoning task..." \
  --latency-budget 2000 \
  --debug

# Pretty-printed output
formatshield generate "Plan a trip to Paris" \
  --schema @travel_schema.json \
  --output pretty
```

### Output Formats

=== "json (default)"

    One-line JSON of the parsed result:

    ```json
    {"label": "positive", "score": 0.94}
    ```

=== "pretty"

    Indented JSON with routing metadata header:

    ```
    Route: ttf (complexity=0.723, confidence=0.70)
    Latency: 1234ms

    {
      "label": "positive",
      "score": 0.94
    }
    ```

=== "raw"

    Raw LLM output string only:

    ```
    {"label": "positive", "score": 0.94}
    ```

---

## `formatshield benchmark`

```bash
formatshield benchmark [OPTIONS]
```

Run the built-in benchmark suite.

### Options

| Option | Default | Description |
|---|---|---|
| `--tasks` | `gsm_symbolic,medical_ner,template_fill` | Comma-separated list of task names |
| `--backends` | `dryrun` | Comma-separated list of backends |
| `--models` | `groq=groq/llama-3.1-70b-versatile` | Backend→model mapping as `key=value,...` |
| `--output-dir` | `benchmark_results` | Output directory |
| `--quick` | False | Use reduced problem sets for fast testing |
| `--artifacts` | False | Generate paper-ready artifacts after the run |

### Examples

```bash
# Quick dry-run benchmark (no API key needed)
formatshield benchmark --quick

# Full benchmark on Groq
formatshield benchmark \
  --tasks gsm_symbolic,medical_ner \
  --backends groq \
  --models groq=groq/llama-3.1-70b-versatile \
  --output-dir my_results \
  --artifacts

# Multi-backend benchmark
formatshield benchmark \
  --backends groq,ollama \
  --models groq=groq/llama-3.1-70b-versatile,ollama=ollama/llama3.1 \
  --quick
```

### Output

```
Running benchmark: 3 tasks × 1 backend = 3 combinations
  [1/3] gsm_symbolic/dryrun ... 20 problems (quick=True: 5)
  [2/3] medical_ner/dryrun  ... 5 problems
  [3/3] template_fill/dryrun ... 5 problems

Results:
  gsm_symbolic  direct=0.40  ttf=0.60  delta=+0.200  overhead=28%
  medical_ner   direct=0.60  ttf=0.70  delta=+0.100  overhead=22%
  template_fill direct=0.80  ttf=0.80  delta=+0.000  overhead=18%

Summary written to: benchmark_results/summary.csv
```

---

## `formatshield score`

```bash
formatshield score PROMPT [OPTIONS]
```

Compute the complexity score for a prompt without running generation. Useful for understanding why a request was (or would be) routed to TTF.

### Options

| Option | Description |
|---|---|
| `--schema`, `-s` | JSON Schema string or `@path/to/schema.json` |
| `--model`, `-m` | Model identifier for instruction-tune score lookup |

### Example

```bash
formatshield score "Solve step by step: 3x + 7 = 22" \
  --schema '{"type": "object", "properties": {"x": {"type": "number"}, "steps": {"type": "array"}}}' \
  --model groq/llama-3.1-70b-versatile
```

Output:

```
Complexity Analysis
  token_entropy:           0.782
  schema_depth:            2
  required_reasoning_ops:  2
  instruction_tune_score:  0.500
  prompt_length_bucket:    1
  schema_constraint_count: 1

Composite score: 0.612
Routing (groq): direct  [threshold=0.650]
```

---

## `formatshield version`

```bash
formatshield version
```

Print version information:

```
formatshield 0.0.1
Python 3.11.9
Platform: linux
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `OLLAMA_HOST` | Ollama server URL (default: `http://localhost:11434`) |
| `VLLM_BASE_URL` | vLLM server URL (default: `http://localhost:8000/v1`) |
| `FORMATSHIELD_LOG_LEVEL` | Override log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `FORMATSHIELD_DEBUG` | Set to `1` to enable debug mode globally |

---

## Config File

You can place a `formatshield.yml` in the current directory to set defaults:

```yaml
model: groq/llama-3.1-70b-versatile
debug: false
ttf_fallback: true
latency_budget_ms: null
log_level: WARNING
```

CLI options override config file values. Config file values override built-in defaults.
