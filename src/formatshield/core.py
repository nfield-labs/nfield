"""
FormatShield core — the main entry point for intelligent structured generation routing.

Usage::

    import formatshield as fs
    from pydantic import BaseModel

    class MySchema(BaseModel):
        answer: str
        confidence: float

    result = await fs.generate(
        prompt="What is the capital of France?",
        schema=MySchema,
        model="groq/llama-3.3-70b-versatile",
    )
    print(result.parsed.answer)
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

from pydantic import BaseModel, ValidationError

from formatshield.backends.protocol import BackendName, get_backend_name_from_model
from formatshield.observability.logger import StructuredLogger
from formatshield.observability.metrics import MetricsCollector
from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.oracle.threshold_oracle import ThresholdOracle
from formatshield.scorer.complexity_scorer import ComplexityScorer
from formatshield.scorer.features import ComplexityFeatures, StreamEvent
from formatshield.ttf.failure_detector import FailureModeDetector

if TYPE_CHECKING:
    from formatshield.generator import AsyncFormatShieldGenerator, FormatShieldGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    """The complete result of a FormatShield generation call."""

    output: str
    """Raw JSON string returned by the backend."""

    parsed: BaseModel | dict[str, Any] | Any | None
    """Parsed Pydantic model instance (or plain dict) if schema was provided."""

    thinking: str | None
    """Thinking text from Pass 1 of TTF generation. ``None`` for direct routes."""

    routing: RoutingDecision
    """The routing decision made by ThresholdOracle for this request."""

    complexity_score: float
    """Scalar complexity score in [0, 1] computed by ComplexityScorer."""

    failure_modes: list[str]
    """Failure modes detected by FailureModeDetector."""

    latency_ms: float
    """Total wall-clock latency in milliseconds."""

    backend: str
    """Backend used for generation (e.g. ``"groq"``, ``"vllm"``)."""

    model: str
    """Full model identifier used."""

    schema_valid: bool
    """Whether the output passed Pydantic schema validation."""

    fallback_triggered: bool
    """Whether TTF failed and fell back to direct generation."""

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "output": self.output,
            "thinking": self.thinking,
            "routing": {
                "strategy": self.routing.strategy,
                "expected_accuracy_delta": self.routing.expected_accuracy_delta,
                "expected_overhead_pct": self.routing.expected_overhead_pct,
                "confidence": self.routing.confidence,
                "explanation": self.routing.explanation,
            },
            "complexity_score": self.complexity_score,
            "failure_modes": self.failure_modes,
            "latency_ms": self.latency_ms,
            "backend": self.backend,
            "model": self.model,
            "schema_valid": self.schema_valid,
            "fallback_triggered": self.fallback_triggered,
        }


# ---------------------------------------------------------------------------
# output_type helpers
# ---------------------------------------------------------------------------


def _build_schema_from_output_type(output_type: type[Any]) -> dict[str, Any]:
    """Derive a JSON Schema dict from a Python output type annotation.

    Supports: int, float, bool, str, Enum subclasses, Literal[...], list[T].

    Args:
        output_type: A Python type or type annotation.

    Returns:
        JSON Schema dict suitable for constrained generation.
    """
    # bool must come before int — issubclass(bool, int) is True
    if output_type is bool:
        return {"type": "boolean"}
    if output_type is int:
        return {"type": "integer"}
    if output_type is float:
        return {"type": "number"}
    if output_type is str:
        return {"type": "string"}
    if isinstance(output_type, type) and issubclass(output_type, enum.Enum):
        values = [e.value for e in output_type]
        type_name = type(values[0]).__name__ if values else "string"
        return {"type": type_name, "enum": values}
    origin = get_origin(output_type)
    if origin is Literal:
        return {"enum": list(get_args(output_type))}
    if origin is list:
        args = get_args(output_type)
        item_schema = _build_schema_from_output_type(args[0]) if args else {}
        return {"type": "array", "items": item_schema}
    return {}


def _cast_parsed(raw_output: str, output_type: type[Any]) -> Any:
    """Cast raw JSON string to the requested Python output type.

    Args:
        raw_output: Raw JSON string from the backend.
        output_type: Target Python type.

    Returns:
        Value cast to output_type, or raw_output string on parse failure.

    Raises:
        ValueError: If output is a Literal type but value not in allowed set.
    """
    try:
        value = json.loads(raw_output)
    except (json.JSONDecodeError, ValueError):
        return raw_output
    # bool before int — issubclass(bool, int) is True
    if output_type is bool:
        return bool(value)
    if output_type is int:
        return int(value)
    if output_type is float:
        return float(value)
    if output_type is str:
        return str(value)
    if isinstance(output_type, type) and issubclass(output_type, enum.Enum):
        return output_type(value)
    origin = get_origin(output_type)
    if origin is Literal:
        allowed = get_args(output_type)
        if value in allowed:
            return value
        raise ValueError(f"Output {value!r} is not in the allowed Literal values: {list(allowed)}")
    if origin is list:
        args = get_args(output_type)
        if args:
            return [_cast_parsed(json.dumps(item), args[0]) for item in value]
        return value
    return value


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def _build_backend(
    model: str,
    backend_name: BackendName,
    base_url: str | None,
    api_key: str | None,
) -> Any:
    """Instantiate the correct backend adapter from *backend_name*."""
    # Strip "backend/" prefix to get bare model name
    model_name = model.split("/", 1)[1] if "/" in model else model

    if backend_name == "groq":
        from formatshield.backends.groq_backend import GroqBackend

        return GroqBackend(api_key=api_key, model=model_name)

    if backend_name == "openrouter":
        from formatshield.backends.openrouter_backend import OpenRouterBackend

        return OpenRouterBackend(api_key=api_key, model=model_name)

    if backend_name == "ollama":
        from formatshield.backends.ollama_backend import OllamaBackend

        host = base_url or "http://localhost:11434"
        return OllamaBackend(host=host, model=model_name)

    if backend_name == "vllm":
        from formatshield.backends.vllm_backend import VLLMBackend

        url = base_url or "http://localhost:8000/v1"
        return VLLMBackend(base_url=url, model=model_name)

    if backend_name == "dryrun":
        from formatshield.backends.dryrun_backend import DryRunBackend

        return DryRunBackend()

    if backend_name == "openai":
        from formatshield.backends.openai_backend import OpenAIBackend

        return OpenAIBackend(api_key=api_key, model=model_name)

    if backend_name == "anthropic":
        from formatshield.backends.anthropic_backend import AnthropicBackend

        return AnthropicBackend(api_key=api_key, model=model_name)

    if backend_name == "cohere":
        from formatshield.backends.cohere_backend import (
            CohereBackend,  # type: ignore[import-not-found]
        )

        return CohereBackend(api_key=api_key, model=model_name)

    if backend_name == "mistral":
        from formatshield.backends.mistral_backend import (
            MistralBackend,  # type: ignore[import-not-found]
        )

        return MistralBackend(api_key=api_key, model=model_name)

    if backend_name == "together":
        from formatshield.backends.together_backend import (
            TogetherBackend,  # type: ignore[import-not-found]
        )

        return TogetherBackend(api_key=api_key, model=model_name)

    if backend_name == "fireworks":
        from formatshield.backends.openrouter_backend import OpenRouterBackend

        return OpenRouterBackend(api_key=api_key, model=f"fireworks/{model_name}")

    # Fallback: OpenRouter handles most OpenAI-compatible APIs
    from formatshield.backends.openrouter_backend import OpenRouterBackend

    return OpenRouterBackend(api_key=api_key, model=model)


# ---------------------------------------------------------------------------
# FormatShield main class
# ---------------------------------------------------------------------------


class FormatShield:
    """
    Intelligent routing layer for structured LLM generation.

    Scores each (prompt, schema) pair for complexity, detects failure modes,
    and routes to either direct constrained generation or two-pass
    Think-Then-Format (TTF) generation based on empirically-calibrated
    per-backend thresholds.

    Example::

        shield = FormatShield(model="groq/llama-3.3-70b-versatile", debug=True)
        result = await shield.generate(prompt, schema=MySchema)
        print(result.parsed)
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        latency_budget_ms: float | None = None,
        cost_aware: bool = False,
        ttf_fallback: bool = True,
        expose_thinking: bool = False,
        debug: bool = False,
        metrics: MetricsCollector | None = None,
        log_level: str = "WARNING",
        backend: Any | None = None,
    ) -> None:
        self.model = model
        self._latency_budget_ms = latency_budget_ms
        self._cost_aware = cost_aware
        self._ttf_fallback = ttf_fallback
        self._expose_thinking = expose_thinking
        self._debug = debug

        self.backend_name: BackendName = get_backend_name_from_model(model)
        self._backend = (
            backend
            if backend is not None
            else _build_backend(model, self.backend_name, base_url, api_key)
        )

        self._scorer = ComplexityScorer()
        self._oracle = ThresholdOracle()
        self._detector = FailureModeDetector()

        self._metrics = metrics or MetricsCollector()
        self._logger = StructuredLogger(level=log_level)

        logger.debug("FormatShield initialised: model=%s backend=%s", model, self.backend_name)

    # ------------------------------------------------------------------
    # Async generate
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        output_type: type[Any] | None = None,
        debug: bool | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
    ) -> GenerationResult:
        """Generate structured output, routing between TTF and direct.

        Args:
            prompt: The user prompt.
            schema: Pydantic model class or JSON Schema dict for output structure.
            output_type: Python type to cast the output to (int, float, bool, str,
                Enum subclass, Literal[...], or list[T]). When provided and schema
                is None, a JSON Schema is auto-derived from the type.
            debug: Override instance-level debug flag for this call.
            temperature: Sampling temperature passed to the backend.
            max_tokens: Maximum tokens to generate.
            seed: RNG seed for reproducible sampling.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling cutoff.
            frequency_penalty: Frequency penalty (OpenAI-compatible).
            presence_penalty: Presence penalty (OpenAI-compatible).
            stop: Stop sequence(s).
        """
        t_start = time.monotonic()
        use_debug = self._debug if debug is None else debug

        # Extract schema dict + model class
        schema_dict: dict[str, Any] | None = None
        schema_model: type[BaseModel] | None = None

        if schema is not None:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema_model = schema
                schema_dict = schema.model_json_schema()
            elif isinstance(schema, dict):
                schema_dict = schema

        # output_type: derive schema if not already provided
        if output_type is not None and schema_dict is None:
            schema_dict = _build_schema_from_output_type(output_type) or None

        # Complexity scoring
        features: ComplexityFeatures = self._scorer.score(
            prompt=prompt,
            schema=schema_dict,
            model_id=self.model,
        )
        complexity_score = self._scorer.compute_score(features)

        # Failure mode detection
        failure_modes = self._detector.detect(
            features=features,
            model_id=self.model,
            schema=schema_dict or {},
        )

        # Routing decision
        decision: RoutingDecision = self._oracle.predict(
            features=features,
            backend=self.backend_name,
            model_id=self.model,
            latency_budget_ms=self._latency_budget_ms,
        )

        # Override routing when failure modes demand direct
        if self._detector.should_override_to_direct(failure_modes):
            decision = RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=0.0,
                expected_overhead_pct=0.0,
                confidence=0.9,
                explanation=f"FailureModeDetector override: {failure_modes}",
                failure_modes=failure_modes,
            )

        if use_debug:
            self._print_routing_trace(features, complexity_score, decision)

        # Generation
        thinking: str | None = None
        output: str = ""
        fallback_triggered = False

        if decision.use_ttf:
            from formatshield.ttf.engine import TTFEngine

            engine = TTFEngine(
                backend=self._backend,
                ttf_fallback=self._ttf_fallback,
                expose_thinking=self._expose_thinking,
            )
            try:
                thinking, output = await engine.generate(
                    prompt=prompt,
                    schema=schema_dict,
                    schema_model=schema_model,
                )
                if not thinking and self._ttf_fallback:
                    fallback_triggered = True
            except Exception as exc:
                logger.warning("FormatShield: TTF failed (%s), falling back to direct", exc)
                output = await self._backend.generate(
                    prompt,
                    schema=schema_dict,
                    constraints="json" if schema_dict else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
                fallback_triggered = True
        else:
            output = await self._backend.generate(
                prompt,
                schema=schema_dict,
                constraints="json" if schema_dict else None,
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
                top_p=top_p,
                top_k=top_k,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
            )

        # Parse + validate
        parsed: BaseModel | dict[str, Any] | Any | None = None
        schema_valid = False

        if output:
            if schema_model is not None:
                try:
                    parsed = schema_model.model_validate_json(output)
                    schema_valid = True
                except (ValidationError, ValueError):
                    try:
                        parsed = json.loads(output)
                    except (json.JSONDecodeError, ValueError):
                        parsed = None
            elif output_type is not None:
                try:
                    parsed = _cast_parsed(output, output_type)
                    schema_valid = True
                except (ValueError, TypeError):
                    try:
                        parsed = json.loads(output)
                    except (json.JSONDecodeError, ValueError):
                        parsed = None
            else:
                try:
                    parsed = json.loads(output)
                    schema_valid = True
                except (json.JSONDecodeError, ValueError):
                    parsed = None

        latency_ms = (time.monotonic() - t_start) * 1000

        # Observability
        self._metrics.record_routing(decision.strategy, self.backend_name)
        self._metrics.record_latency(latency_ms, self.backend_name)
        if fallback_triggered:
            self._metrics.record_fallback()

        self._logger.log_generation(
            model=self.model,
            backend=self.backend_name,
            route=decision.strategy,
            latency_ms=latency_ms,
            schema_valid=schema_valid,
            fallback=fallback_triggered,
        )

        return GenerationResult(
            output=output,
            parsed=parsed,
            thinking=thinking,
            routing=decision,
            complexity_score=complexity_score,
            failure_modes=failure_modes,
            latency_ms=latency_ms,
            backend=self.backend_name,
            model=self.model,
            schema_valid=schema_valid,
            fallback_triggered=fallback_triggered,
        )

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def generate_sync(
        self,
        prompt: str,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        output_type: type[Any] | None = None,
        debug: bool | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
    ) -> GenerationResult:
        """Synchronous wrapper around :meth:`generate`.

        Safe to call from both sync code and from within an already-running
        event loop (e.g. pytest-asyncio, Jupyter) by running in a new thread.

        Args:
            prompt: The user prompt.
            schema: Pydantic model class or JSON Schema dict.
            output_type: Python type to cast output to (int, Enum, Literal, list[T]).
            debug: Override instance-level debug flag.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            seed: RNG seed for reproducible sampling.
            top_p: Nucleus sampling probability.
            top_k: Top-k sampling cutoff.
            frequency_penalty: Frequency penalty.
            presence_penalty: Presence penalty.
            stop: Stop sequence(s).
        """
        import threading

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            # We're inside an async context — run in a dedicated thread with its own loop
            result_holder: list[GenerationResult] = []
            error_holder: list[BaseException] = []

            def _run() -> None:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result_holder.append(
                        new_loop.run_until_complete(
                            self.generate(
                                prompt,
                                schema,
                                output_type,
                                debug,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                seed=seed,
                                top_p=top_p,
                                top_k=top_k,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                stop=stop,
                            )
                        )
                    )
                except Exception as exc:
                    error_holder.append(exc)
                finally:
                    new_loop.close()

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            thread.join(timeout=120)

            if error_holder:
                raise error_holder[0]
            if result_holder:
                return result_holder[0]
            raise TimeoutError("generate_sync timed out after 120 seconds")
        else:
            return asyncio.run(
                self.generate(
                    prompt,
                    schema,
                    output_type,
                    debug,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    top_p=top_p,
                    top_k=top_k,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                )
            )

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def stream(
        self,
        prompt: str,
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream generation events as async iterator of StreamEvent."""
        schema_dict: dict[str, Any] | None = None
        if schema is not None:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema_dict = schema.model_json_schema()
            elif isinstance(schema, dict):
                schema_dict = schema

        features = self._scorer.score(prompt, schema=schema_dict, model_id=self.model)
        failure_modes = self._detector.detect(features, self.model, schema_dict or {})
        decision = self._oracle.predict(features, self.backend_name, self.model)

        if self._detector.should_override_to_direct(failure_modes):
            decision = RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=0.0,
                expected_overhead_pct=0.0,
                confidence=0.9,
                explanation="FailureModeDetector override",
                failure_modes=failure_modes,
            )

        if decision.use_ttf:
            from formatshield.ttf.engine import TTFEngine

            engine = TTFEngine(self._backend, ttf_fallback=self._ttf_fallback)
            async for event in engine._stream_impl(prompt, schema_dict):
                if self._expose_thinking or event.type != "thinking":
                    yield event
        else:
            async for event in await self._backend.stream(
                prompt, schema=schema_dict, constraints="json" if schema_dict else None
            ):
                yield event

    # ------------------------------------------------------------------
    # Generator factories
    # ------------------------------------------------------------------

    def generator(
        self,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        output_type: type[Any] | None = None,
    ) -> FormatShieldGenerator:
        """Create a reusable synchronous generator with cached schema state.

        Schema features (depth, constraint count) are scored once at construction.
        Only prompt features recompute on each call.

        Args:
            schema: JSON Schema dict or Pydantic model class.
            output_type: Python type to cast output to (int, Enum, Literal, list[T]).

        Returns:
            :class:`~formatshield.generator.FormatShieldGenerator` bound to this shield.

        Example::

            gen = shield.generator(output_type=int)
            result = gen("What is 2+2?")
        """
        from formatshield.generator import FormatShieldGenerator

        return FormatShieldGenerator(self, schema=schema, output_type=output_type)

    def async_generator(
        self,
        schema: type[BaseModel] | dict[str, Any] | None = None,
        output_type: type[Any] | None = None,
    ) -> AsyncFormatShieldGenerator:
        """Create a reusable async generator with cached schema state.

        Args:
            schema: JSON Schema dict or Pydantic model class.
            output_type: Python type to cast output to (int, Enum, Literal, list[T]).

        Returns:
            :class:`~formatshield.generator.AsyncFormatShieldGenerator` bound to this shield.

        Example::

            gen = shield.async_generator(output_type=int)
            result = await gen("What is 2+2?")
            results = await gen.batch(["Q1", "Q2"], max_concurrency=5)
        """
        from formatshield.generator import AsyncFormatShieldGenerator

        return AsyncFormatShieldGenerator(self, schema=schema, output_type=output_type)

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str) -> FormatShield:
        """Load FormatShield from a YAML or JSON config file."""
        import json as _json
        from pathlib import Path

        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if path.suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import]

                config = yaml.safe_load(path.read_text())
            except ImportError as exc:
                raise ImportError("pip install pyyaml to load YAML configs") from exc
        else:
            config = _json.loads(path.read_text())

        return cls(**config)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _print_routing_trace(
        self,
        features: ComplexityFeatures,
        complexity_score: float,
        decision: RoutingDecision,
    ) -> None:
        print(
            f"[FormatShield] model={self.model}\n"
            f"[FormatShield] complexity_score={complexity_score:.3f} "
            f"(schema_depth={features.schema_depth}, "
            f"reasoning_ops={features.required_reasoning_ops}, "
            f"length_bucket={features.prompt_length_bucket})\n"
            f"[FormatShield] route={decision.strategy} | "
            f"expected_delta={decision.expected_accuracy_delta:+.3f} | "
            f"estimated_overhead={decision.expected_overhead_pct:.0f}%\n"
            f"[FormatShield] confidence={decision.confidence:.2f} | "
            f"explanation={decision.explanation!r}"
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


async def generate(
    prompt: str,
    schema: type[BaseModel] | dict[str, Any] | None = None,
    model: str = "groq/llama-3.3-70b-versatile",
    output_type: type[Any] | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    frequency_penalty: float | None = None,
    presence_penalty: float | None = None,
    stop: list[str] | str | None = None,
    **kwargs: Any,
) -> GenerationResult:
    """One-liner API: ``result = await fs.generate(prompt, MySchema, model='groq/llama3')``.

    Parameters
    ----------
    prompt:
        The user prompt.
    schema:
        Pydantic model class or JSON Schema dict.
    model:
        Model identifier in ``"provider/model"`` format.
    output_type:
        Python type to cast the output to (int, Enum, Literal, list[T]).
    temperature:
        Sampling temperature passed to the backend.
    max_tokens:
        Maximum tokens to generate.
    seed:
        RNG seed for reproducible sampling.
    top_p:
        Nucleus sampling probability.
    top_k:
        Top-k sampling cutoff.
    frequency_penalty:
        Frequency penalty (OpenAI-compatible).
    presence_penalty:
        Presence penalty (OpenAI-compatible).
    stop:
        Stop sequence(s).
    **kwargs:
        Additional keyword arguments forwarded to :class:`FormatShield`.
    """
    shield = FormatShield(model=model, **kwargs)
    return await shield.generate(
        prompt,
        schema,
        output_type=output_type,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        top_p=top_p,
        top_k=top_k,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
    )
