# Tutorial 07 — Observability

FormatShield ships structured logging and a metrics collector so you can monitor routing decisions, latency, and fallback rates in production.

---

## What's Instrumented

Every `generate()` call automatically records:

- **Routing decision** — which strategy (`ttf` or `direct`) was chosen, and for which backend
- **Latency** — total wall-clock time in milliseconds
- **Fallback events** — when TTF validation fails and falls back to direct

These are recorded in-memory via `MetricsCollector` and also emitted as structured log lines via `StructuredLogger`.

---

## 1. In-Memory Metrics

By default, every `FormatShield` instance maintains an internal `MetricsCollector`. Access it via `shield._metrics`:

```python
import asyncio
import formatshield as fs
from pydantic import BaseModel

class Answer(BaseModel):
    value: str
    confidence: float

async def main():
    shield = fs.FormatShield(model="groq/llama-3.1-70b-versatile")

    # Run several generations
    prompts = [
        "What is the capital of France?",
        "Solve step by step: if 3x + 7 = 22, what is x?",
        "List three benefits of async programming.",
        "Compare bubble sort and merge sort. Which is faster for large inputs?",
    ]
    for p in prompts:
        await shield.generate(p, schema=Answer)

    # Inspect collected metrics
    m = shield._metrics
    print(f"Total calls:     {m.total_calls}")
    print(f"TTF routes:      {m.ttf_count}")
    print(f"Direct routes:   {m.direct_count}")
    print(f"Fallbacks:       {m.fallback_count}")
    print(f"Avg latency:     {m.mean_latency_ms:.1f}ms")
    print(f"p95 latency:     {m.p95_latency_ms:.1f}ms")

asyncio.run(main())
```

---

## 2. Passing a Shared MetricsCollector

For applications with multiple `FormatShield` instances (e.g. different models), pass a shared `MetricsCollector` to aggregate metrics across all instances:

```python
from formatshield.observability.metrics import MetricsCollector
import formatshield as fs

# Single shared collector
metrics = MetricsCollector()

fast_shield = fs.FormatShield(
    model="groq/llama-3.1-8b-instant",
    metrics=metrics,
)
smart_shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    metrics=metrics,
)

# ... make calls on both shields ...

# All metrics are aggregated in one place
print(metrics.total_calls)
print(metrics.mean_latency_ms)
```

---

## 3. Structured Logging

FormatShield emits JSON-structured log lines via Python's standard `logging` module. Configure the log level at instantiation:

```python
shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    log_level="INFO",   # "DEBUG", "INFO", "WARNING", "ERROR"
)
```

Each generation call emits a log line like:

```json
{
  "event": "generation",
  "model": "groq/llama-3.1-70b-versatile",
  "backend": "groq",
  "route": "ttf",
  "latency_ms": 1234.5,
  "schema_valid": true,
  "fallback": false,
  "timestamp": "2026-04-12T14:30:22.123Z"
}
```

### Configuring Python Logging

```python
import logging
import json

# Pretty-print structured logs to stdout
class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.getLogger("formatshield").addHandler(handler)
logging.getLogger("formatshield").setLevel(logging.INFO)
```

---

## 4. Exporting Metrics to Prometheus

You can export `MetricsCollector` data as Prometheus metrics using the `prometheus_client` library:

```python
from prometheus_client import Counter, Histogram, start_http_server
from formatshield.observability.metrics import MetricsCollector
import formatshield as fs

# Prometheus metrics
ROUTING_COUNTER = Counter(
    "formatshield_routing_total",
    "Number of routing decisions",
    ["strategy", "backend"],
)
LATENCY_HISTOGRAM = Histogram(
    "formatshield_latency_ms",
    "Generation latency in milliseconds",
    ["backend"],
    buckets=[100, 250, 500, 1000, 2000, 5000, 10000],
)
FALLBACK_COUNTER = Counter(
    "formatshield_fallbacks_total",
    "Number of TTF fallback events",
)

class PrometheusMetricsCollector(MetricsCollector):
    def record_routing(self, strategy: str, backend: str) -> None:
        super().record_routing(strategy, backend)
        ROUTING_COUNTER.labels(strategy=strategy, backend=backend).inc()

    def record_latency(self, latency_ms: float, backend: str) -> None:
        super().record_latency(latency_ms, backend)
        LATENCY_HISTOGRAM.labels(backend=backend).observe(latency_ms)

    def record_fallback(self) -> None:
        super().record_fallback()
        FALLBACK_COUNTER.inc()

# Start Prometheus server on port 8080
start_http_server(8080)

shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    metrics=PrometheusMetricsCollector(),
)
```

---

## 5. Exporting Metrics to OpenTelemetry

For OpenTelemetry-based observability (DataDog, Honeycomb, Grafana Tempo):

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
import formatshield as fs
from pydantic import BaseModel

# Set up OTLP tracer
provider = TracerProvider()
provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"))
)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("formatshield")

class MySchema(BaseModel):
    answer: str

shield = fs.FormatShield(model="groq/llama-3.1-70b-versatile")

async def traced_generate(prompt: str):
    with tracer.start_as_current_span("formatshield.generate") as span:
        result = await shield.generate(prompt, schema=MySchema)
        span.set_attribute("fs.strategy", result.routing.strategy)
        span.set_attribute("fs.complexity_score", result.complexity_score)
        span.set_attribute("fs.latency_ms", result.latency_ms)
        span.set_attribute("fs.schema_valid", result.schema_valid)
        span.set_attribute("fs.backend", result.backend)
        return result
```

---

## 6. Health Check Endpoint

For services running FormatShield, expose a health check:

```python
from fastapi import FastAPI
import formatshield as fs
from formatshield.observability.metrics import MetricsCollector

app = FastAPI()
metrics = MetricsCollector()
shield = fs.FormatShield(model="groq/llama-3.1-70b-versatile", metrics=metrics)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "metrics": {
            "total_calls": metrics.total_calls,
            "ttf_rate": metrics.ttf_count / max(metrics.total_calls, 1),
            "fallback_rate": metrics.fallback_count / max(metrics.total_calls, 1),
            "mean_latency_ms": metrics.mean_latency_ms,
            "p95_latency_ms": metrics.p95_latency_ms,
        },
    }
```

---

## 7. Metrics Reference

`MetricsCollector` exposes these properties:

| Property | Type | Description |
|---|---|---|
| `total_calls` | `int` | Total generation calls recorded |
| `ttf_count` | `int` | Calls routed to TTF |
| `direct_count` | `int` | Calls routed to direct |
| `fallback_count` | `int` | TTF fallback events |
| `mean_latency_ms` | `float` | Mean latency across all calls |
| `p95_latency_ms` | `float` | 95th percentile latency |
| `p99_latency_ms` | `float` | 99th percentile latency |

Methods:

| Method | Description |
|---|---|
| `record_routing(strategy, backend)` | Record a routing decision |
| `record_latency(latency_ms, backend)` | Record a latency measurement |
| `record_fallback()` | Record a fallback event |
| `reset()` | Clear all accumulated metrics |

---

## Next Steps

- [Tutorial 08: Contributing](08-contributing.md) — how to contribute to FormatShield
- [Reference: Core](../reference/core.md) — full `FormatShield` API reference
