"""Async engine - the real pipeline runner behind every public entry point.

``AsyncNField`` threads a single ``PipelineState`` through all seven
stages (S0 → S6) and returns an :class:`~nfield.types.ExtractionResult`.
The sync ``NField`` class and both ``nfield`` helpers delegate here, so
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
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

from nfield.config import ExtractionConfig
from nfield.exceptions import SchemaError
from nfield.pipeline.s0_resources import run_stage_0
from nfield.pipeline.s1_schema import run_stage_1
from nfield.pipeline.s2a_structure import run_stage_2a
from nfield.pipeline.s2b_prepass import run_stage_2b
from nfield.pipeline.s2c_packing import run_stage_2c
from nfield.pipeline.s3_excerpt import run_stage_3
from nfield.pipeline.s4_extract import run_stage_4
from nfield.pipeline.s5_validate import run_stage_5
from nfield.pipeline.s5b_recover import run_recovery_pass
from nfield.pipeline.s6_assemble import run_stage_6
from nfield.providers import from_model
from nfield.providers._cache import resolve_cache
from nfield.schema._preflight import preflight_schema

if TYPE_CHECKING:
    from nfield.providers._protocol import LLMProvider
    from nfield.types import ExtractionResult

__all__ = ["AsyncNField", "nfield_async"]

# Environment variable consulted when no model is passed to nfield / the engine.
_MODEL_ENV_VAR: str = "NFIELD_MODEL"

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
# document content, so this is a robustness guard, not a DoS defense - it turns a
# confusing stack overflow into a clean SchemaError. Real schemas nest a few
# levels. Same value as assembly._trie._MAX_PATH_DEPTH for one consistent
# nesting ceiling across the library; conversion adds ~2 frames per level, well
# under CPython's ~1000-frame recursion limit.
_MAX_SCHEMA_DEPTH: int = 256

# Default ceiling on documents extracted at once by extract_batch. Each document itself
# fans out to several leaf calls, so doc-level concurrency multiplies the in-flight API
# calls; a small bound keeps that product under provider rate limits. Raise it on
# higher-throughput plans (5-10 is the common API-concurrency range).
_DEFAULT_BATCH_CONCURRENCY: int = 4


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
# Input checks
# ---------------------------------------------------------------------------


def _require_text_document(document: object) -> None:
    """Reject a document that is not text, with a clear, actionable message.

    The pipeline reads the document as a string. A non-string (``None``, a number, a
    ``Path``, raw bytes) otherwise fails deep inside with a cryptic error; this names
    the problem at the boundary instead. Only the type is checked here; whether an
    empty string is allowed depends on the mode (see ``_require_document_matches_mode``).

    Args:
        document: The value passed as the document.

    Raises:
        TypeError: If *document* is not a ``str``.
    """
    if not isinstance(document, str):
        raise TypeError(
            f"document must be text (str), got {type(document).__name__}. Read a file "
            "first with load_document('path'); convert PDF/DOCX to text yourself."
        )


def _require_document_matches_mode(document: str, closed_book: bool) -> None:
    """Reject a document that contradicts the extraction mode.

    The document and ``closed_book`` are mutually exclusive: closed-book fills the
    schema from the model's own knowledge (no document), while document mode needs
    text to read. Each error names the fix so the caller lands in the right mode.

    Args:
        document: The source text (already type-checked).
        closed_book: Whether the engine is in closed-book mode.

    Raises:
        ValueError: If a document is passed in closed-book mode, or absent otherwise.
    """
    has_document = bool(document.strip())
    if closed_book and has_document:
        raise ValueError(
            "closed_book=True fills the schema from the model's knowledge and ignores "
            "the document; remove it (pass ''), or set closed_book=False to extract "
            "from it."
        )
    if not closed_book and not has_document:
        raise ValueError(
            "no document to extract from; pass a document, or set closed_book=True to "
            "fill the schema from the model's knowledge."
        )


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def _resolve_model(model: str | None, config: ExtractionConfig) -> str:
    """Resolve the model string from the call, the environment, or the config.

    Resolution order: the explicit ``model`` argument, then the
    ``NFIELD_MODEL`` environment variable, then ``config.default_model``.

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
            "Pass model='groq/llama-3.1-8b', set the NFIELD_MODEL "
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


class AsyncNField:
    """Async-native nfield engine: run the full S0-S6 pipeline.

    Construct once with a model (and optionally a reusable schema), then call
    :meth:`extract` for each document. When a schema is supplied at construction
    time it is normalised once and reused across calls.

    Args:
        model: Model string ``"provider/model-name"``. If ``None``, resolved
            from ``NFIELD_MODEL`` or ``config.default_model`` at init.
        schema: Optional reusable schema (dict / Pydantic model / dataclass).
        config: Optional :class:`~nfield.config.ExtractionConfig`.
        context_window: The model's real context window in tokens (C_eff).
            Defaults to the provider's conservative default when omitted.
        max_output_tokens: The model's real output ceiling in tokens (M_O).
        api_key: Provider API key. ``None`` (default) reads it from the
            environment - the recommended path. Pass explicitly only for
            secret-vault / multi-tenant use; it is never logged.
        base_url: Override the provider API base URL (proxy / gateway /
            self-hosted compatible endpoint). ``None`` uses the SDK default.
        instructions: Optional caller steering, prepended to the built-in SFEP
            system prompt and counted in leaf overhead.

    Example:
        >>> # async with AsyncNField("groq/llama-3.1-8b", schema=S) as fs:
        >>> #     result = await fs(document)
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
        # Shared by both providers; keys carry the model name, so they never collide.
        self._cache = resolve_cache(self._config.cache)
        self._provider: LLMProvider = from_model(
            self._model,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
            max_retries=self._config.max_api_retries,
            api_key=api_key,
            base_url=base_url,
            reasoning_model=self._config.reasoning_model,
            cache=self._cache,
        )
        # Optional stronger model the recovery pass escalates stragglers to. Built once
        # here (not per call); it uses its own default specs and the same credentials.
        self._fallback_provider: LLMProvider | None = (
            from_model(
                self._config.fallback_model,
                max_retries=self._config.max_api_retries,
                api_key=api_key,
                base_url=base_url,
                reasoning_model=self._config.reasoning_model,
                cache=self._cache,
            )
            if self._config.fallback_model
            else None
        )
        self._instructions = instructions
        self._schema: dict[str, Any] | None = (
            _normalize_schema(schema) if schema is not None else None
        )
        # chars_per_token comes from the provider's estimator (Stage 0), which
        # refines it from each response's real prompt-token count - so a reused
        # engine sharpens its budget across documents.

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
            The :class:`~nfield.types.ExtractionResult`.

        Raises:
            SchemaError: If no schema is available from the call or construction.
            TypeError: If ``document`` is not a string.
            ValueError: If ``document`` and ``closed_book`` contradict each other.

        Example:
            >>> # result = await engine.extract("invoice text", schema=Invoice)
            >>> # result.status
        """
        _require_text_document(document)
        schema_dict = self._resolve_schema(schema)
        config = self._config
        provider = self._provider
        _require_document_matches_mode(document, config.closed_book)

        # Reject a provably-unsatisfiable schema before spending any API call.
        if config.validate_schema:
            preflight_schema(schema_dict)

        state = run_stage_0(self._provider, self._config)
        state.instructions = self._instructions
        state.inject_dependencies = config.inject_dependencies
        state.knowledge_fallback = config.knowledge_fallback
        state.strict_validation = config.strict_validation
        # Closed-book has no source to ground against, so grounding is forced off.
        state.ground_values = config.ground_values and not config.closed_book
        state.grounding_min_score = config.grounding_min_score
        # Closed-book has no document to locate values in, so provenance is forced off.
        state.include_provenance = config.provenance and not config.closed_book
        state.max_concurrent_calls = config.max_concurrent_calls
        state.closed_book = config.closed_book
        state.self_consistency = config.self_consistency
        # Closed-book ignores the document: feed retrieval an empty source.
        source = "" if config.closed_book else document
        state = run_stage_1(state, schema_dict)
        state = run_stage_2a(state)
        state = run_stage_2b(state, source, config)
        state = run_stage_2c(state, config)
        state = run_stage_3(state)
        state = await run_stage_4(state, provider)
        state = await run_stage_5(state, provider, config)
        state = await run_recovery_pass(
            state, provider, config, fallback_provider=self._fallback_provider
        )
        return run_stage_6(state)

    @overload
    async def extract_batch(
        self,
        documents: list[str],
        schema: object | None = ...,
        *,
        max_concurrent: int | None = ...,
        return_exceptions: Literal[False] = ...,
    ) -> list[ExtractionResult]: ...

    @overload
    async def extract_batch(
        self,
        documents: list[str],
        schema: object | None = ...,
        *,
        max_concurrent: int | None = ...,
        return_exceptions: Literal[True],
    ) -> list[ExtractionResult | BaseException]: ...

    async def extract_batch(
        self,
        documents: list[str],
        schema: object | None = None,
        *,
        max_concurrent: int | None = None,
        return_exceptions: bool = False,
    ) -> list[ExtractionResult] | list[ExtractionResult | BaseException]:
        """Extract many documents concurrently with one reused, calibrated engine.

        Runs each document through :meth:`extract`, bounded by a semaphore so a large
        batch does not fan out into a provider rate-limit storm. The engine calibrates
        once (Stage 0) and every document reuses it.

        Args:
            documents: The source documents to extract.
            schema: Optional schema override applied to every document; falls back to
                the construction-time schema.
            max_concurrent: Max documents in flight at once. Defaults to
                ``_DEFAULT_BATCH_CONCURRENCY``. Each document still fans out to its own
                bounded leaf calls underneath.
            return_exceptions: When ``True``, a document that raises yields its exception
                in place (the batch always completes). When ``False`` (default), the
                first failure is re-raised after all documents have run.

        Returns:
            One result per input document, in input order. With
            ``return_exceptions=True`` a failed document's slot holds the exception.

        Example:
            >>> # results = await engine.extract_batch([doc1, doc2], max_concurrent=2)
        """
        semaphore = asyncio.BoundedSemaphore(max_concurrent or _DEFAULT_BATCH_CONCURRENCY)

        async def _one(document: str) -> ExtractionResult:
            async with semaphore:
                return await self.extract(document, schema)

        settled = await asyncio.gather(*(_one(doc) for doc in documents), return_exceptions=True)
        if return_exceptions:
            return list(settled)
        # Surface the first failure, but only after every document has settled.
        resolved: list[ExtractionResult] = []
        for outcome in settled:
            if isinstance(outcome, BaseException):
                raise outcome
            resolved.append(outcome)
        return resolved

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

    async def __call__(self, document: str, schema: object | None = None) -> ExtractionResult:
        """Alias for :meth:`extract` so ``await engine(document)`` works."""
        return await self.extract(document, schema)

    async def __aenter__(self) -> AsyncNField:
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

    Creates a temporary :class:`AsyncNField`, runs the pipeline once, and
    returns the result. For repeated extraction on the same schema, construct an
    :class:`AsyncNField` and reuse it instead.

    Args:
        document: The source document text.
        schema: The target schema (dict / Pydantic model / dataclass).
        model: Model string ``"provider/model-name"``. If ``None``, resolved
            from ``NFIELD_MODEL`` or ``config.default_model``.
        config: Optional extraction configuration.
        context_window: The model's real context window in tokens (C_eff).
        max_output_tokens: The model's real output ceiling in tokens (M_O).
        api_key: Provider API key. ``None`` reads it from the environment.
        base_url: Override the provider API base URL. ``None`` uses the default.
        instructions: Optional caller steering, prepended to the SFEP prompt.

    Returns:
        The :class:`~nfield.types.ExtractionResult`.

    Raises:
        SchemaError: If no model or schema can be resolved.

    Example:
        >>> # result = await nfield_async(doc, MySchema, "groq/llama-3.1-8b")
    """
    engine = AsyncNField(
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
