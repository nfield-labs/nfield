"""
FormatShield test configuration and shared fixtures.

Provides MockBackend (implements the Backend protocol without real API keys),
schema fixtures, prompt fixtures, and environment helpers used across all
unit and integration test modules.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest

from formatshield.scorer.features import StreamEvent

# ---------------------------------------------------------------------------
# Mock backend implementations
# ---------------------------------------------------------------------------


class MockBackend:
    """
    Deterministic test backend that implements the Backend protocol without
    requiring any real API keys or network access.

    All responses are computed from the inputs so tests remain reproducible.
    """

    name: str = "mock"

    @property
    def supports_kv_cache_reuse(self) -> bool:
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        return 0.20

    async def generate(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
    ) -> str:
        """
        Return a deterministic response string based on the supplied arguments.

        Rules:
        - constraints == "json"  → always valid JSON object
        - schema has "properties"  → build a mock object from property names
        - no constraints         → thinking-style response with <think> prefix
        """
        if constraints == "json":
            return json.dumps({"result": "mock_answer", "confidence": 0.95})

        if schema and isinstance(schema.get("properties"), dict):
            mock_obj: dict = {}
            for prop_name, prop_schema in schema["properties"].items():
                prop_type = (
                    prop_schema.get("type", "string") if isinstance(prop_schema, dict) else "string"
                )
                if prop_type == "integer":
                    mock_obj[prop_name] = 42
                elif prop_type == "number":
                    mock_obj[prop_name] = 3.14
                elif prop_type == "boolean":
                    mock_obj[prop_name] = True
                elif prop_type == "array":
                    mock_obj[prop_name] = []
                elif prop_type == "object":
                    mock_obj[prop_name] = {}
                else:
                    mock_obj[prop_name] = f"mock_{prop_name}"
            return json.dumps(mock_obj)

        # Default: thinking-style response
        thinking = (
            "<think>Step 1: analyze the prompt carefully. "
            "Step 2: conclude with the best answer.</think>"
        )
        return f"{thinking}\nThe answer to your question is: mock_response."

    async def stream(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Yield exactly 3 output StreamEvents then a complete event."""
        return self._stream_impl(prompt, schema, constraints)

    async def _stream_impl(
        self,
        prompt: str,
        schema: dict | None,
        constraints: str | None,
    ) -> AsyncIterator[StreamEvent]:
        tokens = ["mock_", "stream_", "response"]
        for i, token in enumerate(tokens):
            yield StreamEvent(
                type="output",
                token=token,
                backend=self.name,
                latency_ms=float((i + 1) * 10),
            )
        yield StreamEvent(
            type="complete",
            content="mock_stream_response",
            json={"result": "mock_answer", "confidence": 0.95},
            backend=self.name,
            latency_ms=40.0,
        )


class MockBackendWithKVCache(MockBackend):
    """MockBackend variant that advertises KV-cache prefix reuse support."""

    name: str = "mock_kv"

    @property
    def supports_kv_cache_reuse(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Pytest fixtures — backends
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend() -> MockBackend:
    """Return a fresh MockBackend instance for each test."""
    return MockBackend()


@pytest.fixture
def mock_backend_kv() -> MockBackendWithKVCache:
    """Return a fresh MockBackendWithKVCache instance for each test."""
    return MockBackendWithKVCache()


# ---------------------------------------------------------------------------
# Pytest fixtures — schemas
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_schema() -> dict:
    """
    A flat, depth-1 JSON schema with three string properties.

    Depth: 1  (object → properties only, no nesting)
    """
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "age": {"type": "integer"},
        },
        "required": ["name", "email"],
    }


@pytest.fixture
def complex_schema() -> dict:
    """
    A deeply nested, depth-3 JSON schema representing an order with nested
    address and line-item sub-objects.

    Depth: 3  (order → address → city / order → items → product → sku)
    """
    return {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "pattern": "^ORD-[0-9]+$"},
            "status": {
                "type": "string",
                "enum": ["pending", "processing", "shipped", "delivered"],
            },
            "shipping_address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                    "country": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "minLength": 2, "maxLength": 2},
                            "name": {"type": "string"},
                        },
                        "required": ["code", "name"],
                    },
                },
                "required": ["street", "city", "country"],
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "quantity": {"type": "integer", "minimum": 1},
                        "unit_price": {"type": "number", "minimum": 0.0},
                    },
                    "required": ["product_id", "quantity", "unit_price"],
                },
                "minItems": 1,
            },
            "total": {"type": "number", "minimum": 0.0},
        },
        "required": ["order_id", "status", "shipping_address", "line_items", "total"],
    }


# ---------------------------------------------------------------------------
# Pytest fixtures — prompts
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_prompt() -> str:
    """A short, simple arithmetic question."""
    return "What is 2 + 2?"


@pytest.fixture
def complex_prompt() -> str:
    """
    A long, reasoning-heavy math problem that explicitly demands step-by-step
    analysis.  Exceeds 200 words to exercise the long-prompt bucket.
    """
    return (
        "A train departs from Station A at 08:00 and travels towards Station B "
        "at a constant speed of 90 km/h. A second train departs from Station B "
        "at 09:00 and travels towards Station A at a constant speed of 110 km/h. "
        "The distance between the two stations is 450 km. "
        "\n\n"
        "Step 1: Calculate the distance covered by the first train before the "
        "second train departs. Because the first train travels for one hour at "
        "90 km/h, it covers 90 km. Therefore the remaining distance between the "
        "two trains when the second departs is 450 - 90 = 360 km. "
        "\n\n"
        "Step 2: Analyze the combined closing speed of both trains. Since both "
        "trains are moving towards each other, their speeds add: "
        "90 + 110 = 200 km/h. "
        "\n\n"
        "Step 3: Calculate the time until they meet. Solve the equation: "
        "time = distance / combined_speed = 360 / 200 = 1.8 hours = 1 hour "
        "48 minutes. "
        "\n\n"
        "Step 4: Evaluate the meeting time. The second train departs at 09:00 "
        "and they meet 1 hour 48 minutes later, so the meeting time is 10:48. "
        "\n\n"
        "Step 5: Derive the exact meeting point. The first train has been "
        "travelling for 2 hours 48 minutes total (90 × 2.8 = 252 km from A). "
        "Verify: second train covers 110 × 1.8 = 198 km from B. "
        "252 + 198 = 450. Correct. "
        "\n\n"
        "Given this analysis, please reason through the problem carefully and "
        "explain each step, then provide your final structured answer as JSON "
        "with the fields: meeting_time (HH:MM string), distance_from_a_km "
        "(number), and distance_from_b_km (number). Prove your workings before "
        "concluding. Compare your result against the expected value of 252 km "
        "from Station A. Calculate with full precision."
    )


# ---------------------------------------------------------------------------
# Pytest fixtures — environment
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_groq_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Patch the environment so that GROQ_API_KEY is set to a fake test key.

    Use this fixture in tests that instantiate GroqBackend but do NOT actually
    make network calls (e.g. they mock the underlying HTTP client).
    """
    monkeypatch.setenv("GROQ_API_KEY", "test_key_gsk_mock_1234567890abcdef")
