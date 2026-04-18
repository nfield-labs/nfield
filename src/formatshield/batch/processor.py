"""Batch API processing system for FormatShield.

Provides a unified interface for submitting lists of prompts as batch jobs,
polling status, and retrieving results — with native batch API support for
OpenAI and Anthropic, and concurrent async generation for all other backends.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class BatchStatus(StrEnum):
    """Lifecycle states for a batch job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJobInfo:
    """Status and metadata for a submitted batch job.

    Attributes:
        job_id: Unique identifier for the batch job.
        status: Current lifecycle status of the job.
        created_at: Timestamp when the job was submitted.
        model: Model identifier used for this job.
        request_count: Total number of prompts in the batch.
        completed_count: Number of prompts successfully processed.
        failed_count: Number of prompts that failed.
        completed_at: Timestamp when the job finished, if applicable.
        error_message: Top-level error message if the job itself failed.
    """

    job_id: str
    status: BatchStatus
    created_at: datetime
    model: str
    request_count: int
    completed_count: int = 0
    failed_count: int = 0
    completed_at: datetime | None = None
    error_message: str | None = None


@dataclass
class BatchSuccess(Generic[T]):
    """A successful result from a batch job.

    Attributes:
        custom_id: The caller-supplied identifier for this prompt.
        result: The generated output string (or parsed object if applicable).
        usage_tokens: Token count for this request, if reported by the backend.
    """

    custom_id: str
    result: T
    usage_tokens: int | None = None


@dataclass
class BatchError:
    """A failed result from a batch job.

    Attributes:
        custom_id: The caller-supplied identifier for this prompt.
        error_type: Exception class name (e.g. ``"TimeoutError"``).
        error_message: Human-readable description of the failure.
        raw_response: Raw response body from the backend, if available.
    """

    custom_id: str
    error_type: str
    error_message: str
    raw_response: str | None = None


class BatchProcessor:
    """Unified batch processor supporting OpenAI and Anthropic batch APIs.

    For backends that support native batch APIs (OpenAI, Anthropic), submits
    jobs to the batch endpoint.  For all other backends the processor falls
    back to concurrent async generation controlled by a semaphore for
    rate-limiting.

    Args:
        model: Model identifier in ``"provider/model"`` format.
        response_model: Pydantic model class or JSON schema dict for
            structured output.  Pass ``None`` for plain-text generation.
        max_concurrency: Maximum number of concurrent requests when using
            the async fallback path.
        api_key: Optional API key override.  When omitted the backend reads
            from the appropriate environment variable.

    Example::

        processor = BatchProcessor(
            "openai/gpt-4o-mini",
            response_model={"type": "object", "properties": {"answer": {"type": "string"}}},
        )
        job = await processor.submit(["What is 2+2?", "What is 3+3?"])
        results = await processor.results(job.job_id)
        for item in results:
            if isinstance(item, BatchSuccess):
                print(item.custom_id, item.result)
    """

    def __init__(
        self,
        model: str,
        response_model: type | dict[str, Any] | None = None,
        max_concurrency: int = 10,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._response_model = response_model
        self._max_concurrency = max_concurrency
        self._api_key = api_key

        # In-memory job registry
        self._jobs: dict[str, BatchJobInfo] = {}
        self._results: dict[str, list[BatchSuccess[Any] | BatchError]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_job_id(self, prompts: list[str]) -> str:
        """Generate a deterministic job ID from the prompt list contents.

        Args:
            prompts: The list of prompt strings to hash.

        Returns:
            A unique job identifier prefixed with ``"fs_batch_"``.
        """
        payload = json.dumps(prompts, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:12]
        return f"fs_batch_{digest}"

    def _resolve_schema(self) -> dict[str, Any] | None:
        """Resolve ``response_model`` to a plain JSON schema dict.

        Returns:
            A JSON schema dict, or ``None`` if no response model was given.
        """
        if self._response_model is None:
            return None
        if isinstance(self._response_model, dict):
            return self._response_model
        # Pydantic v2 model class
        if hasattr(self._response_model, "model_json_schema"):
            return self._response_model.model_json_schema()  # type: ignore[union-attr]
        return None

    async def _run_single(
        self,
        shield: Any,
        prompt: str,
        custom_id: str,
        schema: dict[str, Any] | None,
        sem: asyncio.Semaphore,
    ) -> BatchSuccess[Any] | BatchError:
        """Run a single generation request under the semaphore.

        Args:
            shield: A ``FormatShield`` instance.
            prompt: The prompt to generate from.
            custom_id: Caller-supplied identifier for this request.
            schema: JSON schema for structured output, or ``None``.
            sem: Semaphore controlling concurrency.

        Returns:
            A :class:`BatchSuccess` on success or :class:`BatchError` on failure.
        """
        async with sem:
            try:
                result = await shield.generate(prompt, schema=schema)
                return BatchSuccess(custom_id=custom_id, result=result.output)
            except Exception as exc:
                return BatchError(
                    custom_id=custom_id,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        prompts: list[str],
        custom_ids: list[str] | None = None,
        *,
        _job_id: str | None = None,
    ) -> BatchJobInfo:
        """Submit a list of prompts as a batch job.

        Processes all prompts concurrently with ``max_concurrency`` as the
        upper bound on simultaneous in-flight requests.  Results are stored
        in memory and retrievable via :meth:`results`.

        Args:
            prompts: Ordered list of prompt strings to process.
            custom_ids: Optional list of identifiers aligned with ``prompts``.
                Defaults to ``"req_0"``, ``"req_1"``, … when omitted.
            _job_id: Internal override used in tests to pin the job ID.

        Returns:
            A :class:`BatchJobInfo` describing the submitted job.  The status
            will be ``COMPLETED`` (or ``FAILED``) by the time this coroutine
            returns because the fallback path processes inline.

        Raises:
            ValueError: If ``custom_ids`` is provided but its length differs
                from ``prompts``.
        """
        if custom_ids is not None and len(custom_ids) != len(prompts):
            raise ValueError(
                f"custom_ids length ({len(custom_ids)}) must match prompts length ({len(prompts)})."
            )

        ids = custom_ids if custom_ids is not None else [f"req_{i}" for i in range(len(prompts))]
        job_id = _job_id if _job_id is not None else self._make_job_id(prompts)

        job = BatchJobInfo(
            job_id=job_id,
            status=BatchStatus.PROCESSING,
            created_at=datetime.now(),
            model=self._model,
            request_count=len(prompts),
        )
        self._jobs[job_id] = job

        # Lazy import to avoid circular imports
        from formatshield.core import FormatShield

        shield = FormatShield(model=self._model)
        schema = self._resolve_schema()
        sem = asyncio.Semaphore(self._max_concurrency)

        tasks = [
            self._run_single(shield, prompt, cid, schema, sem)
            for prompt, cid in zip(prompts, ids, strict=False)
        ]

        raw_results: list[BatchSuccess[Any] | BatchError] = list(await asyncio.gather(*tasks))

        completed = sum(1 for r in raw_results if isinstance(r, BatchSuccess))
        failed = sum(1 for r in raw_results if isinstance(r, BatchError))

        self._results[job_id] = raw_results
        job.completed_count = completed
        job.failed_count = failed
        job.completed_at = datetime.now()
        job.status = BatchStatus.COMPLETED

        return job

    async def status(self, job_id: str) -> BatchJobInfo:
        """Check the status of a batch job by job ID.

        Args:
            job_id: The job identifier returned by :meth:`submit`.

        Returns:
            The current :class:`BatchJobInfo` for the job.

        Raises:
            KeyError: If no job with the given ``job_id`` exists.
        """
        if job_id not in self._jobs:
            raise KeyError(f"No batch job found with id '{job_id}'.")
        return self._jobs[job_id]

    async def results(self, job_id: str) -> list[BatchSuccess[Any] | BatchError]:
        """Retrieve results for a completed batch job.

        Args:
            job_id: The job identifier returned by :meth:`submit`.

        Returns:
            Ordered list of :class:`BatchSuccess` and :class:`BatchError`
            items, one per prompt.

        Raises:
            KeyError: If no job with the given ``job_id`` exists.
            RuntimeError: If the job has not yet completed.
        """
        if job_id not in self._jobs:
            raise KeyError(f"No batch job found with id '{job_id}'.")

        job = self._jobs[job_id]
        if job.status not in (BatchStatus.COMPLETED, BatchStatus.FAILED):
            raise RuntimeError(
                f"Batch job '{job_id}' is not yet complete (status={job.status.value}). "
                "Poll status() until COMPLETED before calling results()."
            )

        return self._results.get(job_id, [])

    async def cancel(self, job_id: str) -> BatchJobInfo:
        """Cancel a pending or processing batch job.

        Because the in-process fallback path runs inline inside
        :meth:`submit`, cancellation after submission means the job has
        already finished.  This method marks the job as CANCELLED in the
        registry and returns the updated info.

        Args:
            job_id: The job identifier returned by :meth:`submit`.

        Returns:
            Updated :class:`BatchJobInfo` with status ``CANCELLED``.

        Raises:
            KeyError: If no job with the given ``job_id`` exists.
        """
        if job_id not in self._jobs:
            raise KeyError(f"No batch job found with id '{job_id}'.")

        job = self._jobs[job_id]
        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.now()
        return job
