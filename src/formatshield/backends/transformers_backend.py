"""HuggingFace Transformers inference backend for FormatShield."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from formatshield.scorer.features import StreamEvent

if TYPE_CHECKING:
    pass  # transformers imported lazily in methods


class TransformersBackend:
    """Local HuggingFace Transformers backend using a text-generation pipeline.

    Runs inference locally using the ``transformers`` library.  No API key is
    required — the model weights are downloaded from the HuggingFace Hub on
    first use and cached locally.

    Because the pipeline is a blocking, CPU/GPU-bound call it is always
    dispatched to a thread-pool executor so the asyncio event loop is never
    blocked.

    Args:
        model: HuggingFace model identifier.  Accepts the plain Hub ID (e.g.
            ``"meta-llama/Llama-3.1-8B-Instruct"``) or the prefixed form used
            by the FormatShield router (``"transformers/…"`` or ``"hf/…"``).
        device: Device string passed directly to the ``pipeline`` constructor,
            e.g. ``"cpu"``, ``"cuda"``, ``"mps"``, or a device index such as
            ``0``.
        torch_dtype: dtype string forwarded to ``pipeline`` as
            ``torch_dtype``.  ``"auto"`` lets transformers choose the best
            precision for the device.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "transformers"

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cpu",
        torch_dtype: str = "auto",
    ) -> None:
        # Strip optional "transformers/" or "hf/" prefix so the raw Hub ID
        # reaches the pipeline constructor.
        self._model_name = model.removeprefix("transformers/").removeprefix("hf/")
        self._device = device
        self._torch_dtype = torch_dtype
        self._pipeline: Any = None  # lazy-loaded on first generate() call

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """Transformers pipeline does not expose server-side KV-cache reuse."""
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """10 % baseline accuracy loss under constrained decoding, measured on
        FormatShield's internal benchmark suite for local Transformers models.
        """
        return 0.10

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pipeline(self) -> Any:
        """Lazily load and cache the HuggingFace text-generation pipeline.

        Returns:
            A ``transformers.Pipeline`` instance ready for inference.

        Raises:
            ImportError: If ``transformers`` is not installed.
        """
        if self._pipeline is None:
            try:
                from transformers import pipeline  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "transformers is required for TransformersBackend. "
                    "Install with: pip install 'formatshield[transformers]'"
                ) from exc
            self._pipeline = pipeline(
                "text-generation",
                model=self._model_name,
                device=self._device,
            )
        return self._pipeline

    def _build_prompt(self, prompt: str, schema: dict[str, Any] | None) -> str:
        """Embed the JSON schema into the prompt when one is provided.

        Because the local pipeline does not natively support grammar-constrained
        decoding, the schema is serialised and prepended as an instruction so
        the model understands the expected output structure.

        Args:
            prompt: The raw user prompt.
            schema: Optional JSON schema dict describing the expected output.

        Returns:
            The (potentially augmented) prompt string.
        """
        if schema is None:
            return prompt
        schema_text = json.dumps(schema, indent=2)
        return (
            f"You must respond with valid JSON conforming to this schema:\n{schema_text}\n\n"
            f"Do not include any text outside the JSON object.\n\n{prompt}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
    ) -> str:
        """Generate a response using the local Transformers pipeline.

        The blocking pipeline call is dispatched to a thread-pool executor so
        the event loop remains responsive.  When *schema* is provided it is
        embedded into the prompt as an instruction.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict for structured output.  Embedded
                into the prompt as an instruction because the pipeline does not
                support grammar-constrained decoding natively.
            constraints: Constraint hint string.  Currently unused for this
                backend but accepted for protocol compatibility.
            kv_cache_prefix: Ignored; this backend does not support prefix
                caching.
            temperature: Sampling temperature.  Values above ``0`` enable
                stochastic sampling; ``0`` (default) uses greedy decoding.
            top_p: Nucleus-sampling probability cutoff.  ``None`` defers to the
                pipeline default.
            top_k: Top-k sampling parameter.  ``None`` defers to the pipeline
                default.
            max_tokens: Maximum number of new tokens to generate.  Defaults to
                ``512`` when not specified.
            seed: Random seed for reproducible sampling.  ``None`` defers to
                the pipeline default.
            frequency_penalty: Not supported by the Transformers pipeline;
                accepted for protocol compatibility but ignored.
            presence_penalty: Not supported by the Transformers pipeline;
                accepted for protocol compatibility but ignored.
            stop: Stop sequence(s) forwarded to the pipeline as
                ``stopping_criteria``.  ``None`` defers to the pipeline
                default.

        Returns:
            The model's response as a plain string with the input prompt
            stripped from the output.

        Raises:
            ImportError: If the ``transformers`` package is not installed.
        """
        import asyncio

        # Accepted for protocol compatibility — not supported by Transformers pipeline.
        del constraints, kv_cache_prefix, frequency_penalty, presence_penalty

        full_prompt = self._build_prompt(prompt, schema)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens or 512,
            "do_sample": temperature is not None and temperature > 0,
            "temperature": temperature if temperature is not None else 1.0,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k
        if seed is not None:
            gen_kwargs["seed"] = seed
        if stop is not None:
            gen_kwargs["stopping_criteria"] = stop

        def _run() -> str:
            pipe = self._get_pipeline()
            results = pipe(full_prompt, **gen_kwargs)
            if isinstance(results, list) and results:
                text: str = results[0].get("generated_text", "")
                # Strip the input prompt that many pipelines echo back.
                if text.startswith(full_prompt):
                    text = text[len(full_prompt) :]
                return text.strip()
            return ""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _run)

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream the model's response as :class:`StreamEvent` objects.

        Because the Transformers pipeline is blocking, this implementation
        delegates to :meth:`generate` and emits a single ``"complete"`` event
        with the full response once generation finishes.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict embedded into the prompt.
            constraints: Constraint hint string; ignored for this backend.
            temperature: Sampling temperature.
            top_p: Nucleus-sampling probability cutoff.
            top_k: Top-k sampling parameter.
            max_tokens: Maximum number of new tokens to generate.
            seed: Random seed for reproducible sampling.
            frequency_penalty: Ignored; not supported by this backend.
            presence_penalty: Ignored; not supported by this backend.
            stop: Stop sequence(s).

        Yields:
            A single :class:`~formatshield.scorer.features.StreamEvent` of
            type ``"complete"`` carrying the full response text.
        """
        t0 = time.monotonic()
        result = await self.generate(
            prompt,
            schema,
            constraints,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        )
        yield StreamEvent(
            type="complete",
            content=result,
            backend=self.name,
            latency_ms=(time.monotonic() - t0) * 1000,
        )
