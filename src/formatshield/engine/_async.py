"""Async engine — the real pipeline runner behind every public entry point.

``AsyncFormatShield`` threads a single ``PipelineState`` through all seven
stages (S0 → S6) and returns an :class:`~formatshield.types.ExtractionResult`.
The sync ``FormatShield`` class and both ``nfield`` helpers delegate here, so
this module is the one place the stage order is defined.

Schema input is polymorphic: a JSON Schema ``dict``, a Pydantic model
(class or instance), or a Python ``dataclass`` (class or instance) are all
accepted and normalised to a JSON Schema dict before Stage 1.
"""

from __future__ import annotations

import asyncio
import dataclasses
import os
import types
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

from formatshield.config import ExtractionConfig
from formatshield.exceptions import SchemaError
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s0_resources import run_stage_0
from formatshield.pipeline.s1_schema import run_stage_1
from formatshield.pipeline.s2a_structure import run_stage_2a
from formatshield.pipeline.s2b_prepass import run_stage_2b
from formatshield.pipeline.s2c_packing import run_stage_2c
from formatshield.pipeline.s3_excerpt import run_stage_3
from formatshield.pipeline.s4_extract import run_stage_4
from formatshield.pipeline.s5_validate import run_stage_5
from formatshield.pipeline.s5b_recover import run_recovery_pass
from formatshield.pipeline.s6_assemble import run_stage_6
from formatshield.providers import from_model

if TYPE_CHECKING:
    from formatshield.providers._protocol import LLMProvider
    from formatshield.types import ExtractionResult

__all__ = ["AsyncFormatShield", "nfield_async"]

# Environment variable consulted when no model is passed to nfield / the engine.
_MODEL_ENV_VAR: str = "FORMATSHIELD_MODEL"

# Minimal Python type → JSON Schema type map for dataclass conversion.
_PRIMITIVE_JSON_TYPES: dict[type, str] = {
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
}

# Bound dataclass-schema recursion. A self-referential dataclass (e.g. a tree
# node whose field type is its own class) would otherwise recurse forever and
# crash with a RecursionError. The schema is caller-supplied, not untrusted
# document content, so this is a robustness guard, not a DoS defense — it turns a
# confusing stack overflow into a clean SchemaError. Real schemas nest a few
# levels. Same value as assembly._trie._MAX_PATH_DEPTH for one consistent
# nesting ceiling across the library; conversion adds ~2 frames per level, well
# under CPython's ~1000-frame recursion limit.
_MAX_SCHEMA_DEPTH: int = 256


def _check_schema_depth(depth: int) -> None:
    """Raise ``SchemaError`` once dataclass conversion nests past the cap.

    Args:
        depth: Current recursion depth in :func:`_python_type_to_node` /
            :func:`_dataclass_to_json_schema`.

    Raises:
        SchemaError: If ``depth`` exceeds ``_MAX_SCHEMA_DEPTH``.
    """
    if depth > _MAX_SCHEMA_DEPTH:
        raise SchemaError(
            f"Schema nests deeper than the maximum {_MAX_SCHEMA_DEPTH} levels.",
            hint="Check for a self-referential dataclass; flatten or break the cycle.",
        )


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def _resolve_model(model: str | None, config: ExtractionConfig) -> str:
    """Resolve the model string from the call, the environment, or the config.

    Resolution order: the explicit ``model`` argument, then the
    ``FORMATSHIELD_MODEL`` environment variable, then ``config.default_model``.

    Args:
        model: Model string passed by the caller, or ``None``.
        config: Active extraction configuration (may carry a default model).

    Returns:
        A non-empty ``"provider/model-name"`` string.

    Raises:
        SchemaError: If no model can be resolved from any source.

    Example:
        >>> _resolve_model("groq/llama-3.1-8b", ExtractionConfig())
        'groq/llama-3.1-8b'
    """
    if model:
        return model
    env_model = os.getenv(_MODEL_ENV_VAR)
    if env_model:
        return env_model
    if config.default_model:
        return config.default_model
    raise SchemaError(
        "No model specified for extraction.",
        hint=(
            "Pass model='groq/llama-3.1-8b', set the FORMATSHIELD_MODEL "
            "environment variable, or set ExtractionConfig(default_model=...)."
        ),
    )


# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------


def _python_type_to_node(annotation: Any, *, depth: int = 0) -> dict[str, Any]:
    """Map a Python type annotation to a JSON Schema node.

    Handles primitives, ``Optional[X]`` / ``X | None``, homogeneous ``list[X]``,
    and nested dataclasses. Unknown types fall back to ``{"type": "string"}``.

    Args:
        annotation: A type annotation from a dataclass field.
        depth: Current recursion depth, bounded by ``_MAX_SCHEMA_DEPTH``.

    Returns:
        A JSON Schema fragment describing the annotation.

    Raises:
        SchemaError: If conversion nests past ``_MAX_SCHEMA_DEPTH`` (e.g. a
            self-referential dataclass).
    """
    _check_schema_depth(depth)
    origin = get_origin(annotation)

    # Optional[X] (typing.Union) and X | None (types.UnionType) both unwrap here.
    if origin is Union or isinstance(annotation, types.UnionType):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_node(non_none[0], depth=depth + 1)
        return {"type": "string"}

    if origin in (list, tuple, set):
        item_args = get_args(annotation)
        item_node = (
            _python_type_to_node(item_args[0], depth=depth + 1)
            if item_args
            else {"type": "string"}
        )
        return {"type": "array", "items": item_node}

    if isinstance(annotation, type):
        if annotation in _PRIMITIVE_JSON_TYPES:
            return {"type": _PRIMITIVE_JSON_TYPES[annotation]}
        if dataclasses.is_dataclass(annotation):
            return _dataclass_to_json_schema(annotation, depth=depth + 1)

    return {"type": "string"}


def _dataclass_to_json_schema(cls: type, *, depth: int = 0) -> dict[str, Any]:
    """Convert a dataclass type to an equivalent JSON Schema object.

    Args:
        cls: A dataclass type.
        depth: Current recursion depth, bounded by ``_MAX_SCHEMA_DEPTH``.

    Returns:
        A JSON Schema ``object`` with one property per dataclass field. Fields
        without a default (or default factory) are listed as ``required``.

    Raises:
        SchemaError: If type hints cannot be resolved or conversion nests past
            ``_MAX_SCHEMA_DEPTH`` (e.g. a self-referential dataclass).
    """
    _check_schema_depth(depth)
    try:
        hints = get_type_hints(cls)
    except (NameError, TypeError) as exc:
        raise SchemaError(
            f"Could not resolve type hints for dataclass {cls.__name__!r}.",
            hint="Define the dataclass and any nested types at module scope.",
        ) from exc
    properties: dict[str, Any] = {}
    required: list[str] = []
    for f in dataclasses.fields(cls):
        properties[f.name] = _python_type_to_node(hints.get(f.name, str), depth=depth + 1)
        has_default = (
            f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
        )
        if not has_default:
            required.append(f.name)
    return {"type": "object", "properties": properties, "required": required}


def _normalize_schema(schema: object) -> dict[str, Any]:
    """Normalise any supported schema input to a JSON Schema dict.

    Accepts a JSON Schema ``dict`` (returned as-is), a Pydantic model
    (class or instance, via ``model_json_schema()``), or a Python dataclass
    (class or instance).

    Args:
        schema: The schema in one of the supported forms.

    Returns:
        A JSON Schema dict ready for Stage 1.

    Raises:
        SchemaError: If the schema is of an unsupported type.

    Example:
        >>> _normalize_schema({"type": "object", "properties": {}})
        {'type': 'object', 'properties': {}}
    """
    if isinstance(schema, dict):
        return schema

    model_json_schema = getattr(schema, "model_json_schema", None)
    if callable(model_json_schema):
        produced = model_json_schema()
        if not isinstance(produced, dict):
            raise SchemaError(
                "model_json_schema() did not return a dict.",
                hint="Ensure the object is a Pydantic v2 model.",
            )
        return produced

    if dataclasses.is_dataclass(schema):
        cls = schema if isinstance(schema, type) else type(schema)
        return _dataclass_to_json_schema(cls)

    raise SchemaError(
        f"Unsupported schema type: {type(schema).__name__}.",
        hint="Pass a JSON Schema dict, a Pydantic model, or a dataclass.",
    )


# ---------------------------------------------------------------------------
# Async engine
# ---------------------------------------------------------------------------


class AsyncFormatShield:
    """Async-native FormatShield engine: run the full S0-S6 pipeline.

    Construct once with a model (and optionally a reusable schema), then call
    :meth:`extract` for each document. When a schema is supplied at construction
    time it is normalised once and reused across calls.

    Args:
        model: Model string ``"provider/model-name"``. If ``None``, resolved
            from ``FORMATSHIELD_MODEL`` or ``config.default_model`` at init.
        schema: Optional reusable schema (dict / Pydantic model / dataclass).
        config: Optional :class:`~formatshield.config.ExtractionConfig`.
        context_window: The model's real context window in tokens (C_eff).
            Defaults to the provider's conservative default when omitted.
        max_output_tokens: The model's real output ceiling in tokens (M_O).
        api_key: Provider API key. ``None`` (default) reads it from the
            environment — the recommended path. Pass explicitly only for
            secret-vault / multi-tenant use; it is never logged.
        base_url: Override the provider API base URL (proxy / gateway /
            self-hosted compatible endpoint). ``None`` uses the SDK default.
        instructions: Optional caller steering, prepended to the built-in SFEP
            system prompt and counted in leaf overhead.

    Example:
        >>> # async with AsyncFormatShield("groq/llama-3.1-8b", schema=S) as fs:
        >>> #     result = await fs(document)
        >>> isinstance(AsyncFormatShield, type)
        True
    """

    def __init__(
        self,
        model: str | None = None,
        schema: object | None = None,
        *,
        config: ExtractionConfig | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        instructions: str = "",
    ) -> None:
        self._config: ExtractionConfig = config or ExtractionConfig()
        self._model: str = _resolve_model(model, self._config)
        self._provider: LLMProvider = from_model(
            self._model,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            max_retries=self._config.max_api_retries,
            api_key=api_key,
            base_url=base_url,
        )
        # Optional stronger model the recovery pass escalates stragglers to. Built once
        # here (not per call); it uses its own default specs and the same credentials.
        self._fallback_provider: LLMProvider | None = (
            from_model(
                self._config.fallback_model,
                max_retries=self._config.max_api_retries,
                api_key=api_key,
                base_url=base_url,
            )
            if self._config.fallback_model
            else None
        )
        self._instructions = instructions
        self._schema: dict[str, Any] | None = (
            _normalize_schema(schema) if schema is not None else None
        )
        # chars_per_token is a property of the model's tokenizer — the Normalized
        # Sequence Length (chars / token; arXiv:2411.12240) — and of the language,
        # never of the document. It is measured once and reused across extract()
        # calls so a reused engine calibrates a single time.
        self._chars_per_token: float | None = None
        self._c_eff: int = 0
        self._m_o: int = 0
        self._c_usable: float = 0.0
        # Serializes the one-time Stage 0 calibration so concurrent extract()
        # calls on a shared engine measure chars_per_token once, not once each.
        self._calibration_lock = asyncio.Lock()

    @property
    def model(self) -> str:
        """Return the resolved model string for this engine."""
        return self._model

    async def extract(self, document: str, schema: object | None = None) -> ExtractionResult:
        """Run the full extraction pipeline on a single document.

        Threads one ``PipelineState`` through stages S0 → S6. A schema passed
        here overrides the construction-time schema; otherwise the cached one
        is reused.

        Args:
            document: The source document text.
            schema: Optional per-call schema override (dict / Pydantic / dataclass).

        Returns:
            The :class:`~formatshield.types.ExtractionResult`.

        Raises:
            SchemaError: If no schema is available from the call or construction.

        Example:
            >>> # result = await engine.extract("invoice text", schema=Invoice)
            >>> # result.status
        """
        schema_dict = self._resolve_schema(schema)
        config = self._config
        provider = self._provider

        state = await self._calibrated_state()
        state.instructions = self._instructions
        state.inject_dependencies = config.inject_dependencies
        state.knowledge_fallback = config.knowledge_fallback
        state.strict_validation = config.strict_validation
        state.ground_values = config.ground_values
        state.grounding_min_score = config.grounding_min_score
        state.max_concurrent_calls = config.max_concurrent_calls
        state = run_stage_1(state, schema_dict)
        state = run_stage_2a(state)
        state = run_stage_2b(state, document, config)
        state = run_stage_2c(state, config)
        state = run_stage_3(state)
        state = await run_stage_4(state, provider)
        state = await run_stage_5(state, provider, config)
        state = await run_recovery_pass(
            state, provider, config, fallback_provider=self._fallback_provider
        )
        return run_stage_6(state)

    def _resolve_schema(self, schema: object | None) -> dict[str, Any]:
        """Pick the per-call schema, falling back to the cached one."""
        if schema is not None:
            return _normalize_schema(schema)
        if self._schema is not None:
            return self._schema
        raise SchemaError(
            "No schema provided.",
            hint="Pass schema=... to extract() or to the constructor.",
        )

    async def _calibrated_state(self) -> PipelineState:
        """Return a fresh ``PipelineState`` carrying Stage 0 calibration.

        The first call runs Stage 0 (one provider call to measure
        ``chars_per_token``) and caches the result on the engine. Later calls
        skip that round trip and build a fresh state from the cached values, so
        a reused engine calibrates only once. The lock + double-checked guard
        keeps that "once" true even when extract() calls run concurrently.
        """
        if self._chars_per_token is None:
            async with self._calibration_lock:
                # Re-check inside the lock: a racing call may have just filled it.
                if self._chars_per_token is None:
                    state = await run_stage_0(self._provider, self._config)
                    self._chars_per_token = state.chars_per_token
                    self._c_eff = state.C_eff
                    self._m_o = state.M_O
                    self._c_usable = state.C_usable
                    return state
        return PipelineState(
            chars_per_token=self._chars_per_token,
            C_eff=self._c_eff,
            M_O=self._m_o,
            C_usable=self._c_usable,
        )

    async def __call__(self, document: str, schema: object | None = None) -> ExtractionResult:
        """Alias for :meth:`extract` so ``await engine(document)`` works."""
        return await self.extract(document, schema)

    async def __aenter__(self) -> AsyncFormatShield:
        """Enter the async context manager (returns ``self``)."""
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the async context manager.

        The provider's network client is created lazily and holds no
        long-lived resources here, so there is nothing to close.
        """
        return None


# ---------------------------------------------------------------------------
# One-shot async entry point
# ---------------------------------------------------------------------------


async def nfield_async(
    document: str,
    schema: object,
    model: str | None = None,
    *,
    config: ExtractionConfig | None = None,
    context_window: int | None = None,
    max_output_tokens: int | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    instructions: str = "",
) -> ExtractionResult:
    """Extract N structured fields from a document (async, one-shot).

    Creates a temporary :class:`AsyncFormatShield`, runs the pipeline once, and
    returns the result. For repeated extraction on the same schema, construct an
    :class:`AsyncFormatShield` and reuse it instead.

    Args:
        document: The source document text.
        schema: The target schema (dict / Pydantic model / dataclass).
        model: Model string ``"provider/model-name"``. If ``None``, resolved
            from ``FORMATSHIELD_MODEL`` or ``config.default_model``.
        config: Optional extraction configuration.
        context_window: The model's real context window in tokens (C_eff).
        max_output_tokens: The model's real output ceiling in tokens (M_O).
        api_key: Provider API key. ``None`` reads it from the environment.
        base_url: Override the provider API base URL. ``None`` uses the default.
        instructions: Optional caller steering, prepended to the SFEP prompt.

    Returns:
        The :class:`~formatshield.types.ExtractionResult`.

    Raises:
        SchemaError: If no model or schema can be resolved.

    Example:
        >>> # result = await nfield_async(doc, MySchema, "groq/llama-3.1-8b")
        >>> callable(nfield_async)
        True
    """
    engine = AsyncFormatShield(
        model,
        schema,
        config=config,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        api_key=api_key,
        base_url=base_url,
        instructions=instructions,
    )
    return await engine.extract(document)
