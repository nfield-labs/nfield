"""llama.cpp inference backend for FormatShield."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

from formatshield.scorer.features import StreamEvent


class LlamaCppBackend:
    """Local llama.cpp backend via the ``llama-cpp-python`` binding.

    Runs GGUF-quantised models entirely on local hardware with no API key
    requirement.  JSON grammar mode is activated automatically when a schema
    is provided, constraining the model's output to valid JSON at the
    token-sampling level.

    Because the llama.cpp inference call is blocking and CPU/GPU-bound it is
    always dispatched to a thread-pool executor so the asyncio event loop
    remains responsive.

    Args:
        model: Path to the GGUF model file.  Accepts both a bare path (e.g.
            ``"models/llama-3.1-8b.gguf"``) and the ``"llamacpp/"``-prefixed
            format used by the FormatShield router.
        n_ctx: Context window size in tokens.  Larger values consume more
            VRAM/RAM.
        n_gpu_layers: Number of transformer layers to offload to the GPU.
            ``0`` runs entirely on CPU.
        verbose: When ``True``, llama.cpp prints per-token progress to stderr.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "llamacpp"

    def __init__(
        self,
        model: str = "models/llama-3.1-8b.gguf",
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        # Strip optional "llamacpp/" prefix so the bare file path reaches Llama().
        self._model_path = model.removeprefix("llamacpp/")
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._verbose = verbose
        self._llm: Any = None  # lazy-loaded on first generate() call

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """llama.cpp does not expose server-side KV-cache prefix reuse via
        the Python binding used by this backend.
        """
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """12 % baseline accuracy loss under constrained JSON grammar decoding,
        measured on FormatShield's internal benchmark suite for GGUF models.
        """
        return 0.12

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_llm(self) -> Any:
        """Lazily load and cache the llama.cpp ``Llama`` instance.

        Returns:
            A ``llama_cpp.Llama`` object ready for inference.

        Raises:
            ImportError: If ``llama-cpp-python`` is not installed.
        """
        if self._llm is None:
            try:
                from llama_cpp import Llama  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "llama-cpp-python is required for LlamaCppBackend. "
                    "Install with: pip install 'formatshield[llamacpp]'"
                ) from exc
            self._llm = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_gpu_layers=self._n_gpu_layers,
                verbose=self._verbose,
            )
        return self._llm

    def _build_user_content(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        use_grammar_mode: bool,
    ) -> str:
        """Build the user message content, embedding the schema when needed.

        When grammar mode is active (``response_format={"type": "json_object"}``),
        llama.cpp handles structural constraints at the token level, so only a
        brief instruction is prepended.  Without grammar mode the full schema is
        embedded verbatim so the model can infer the expected structure.

        Args:
            prompt: The raw user prompt.
            schema: Optional JSON schema dict.
            use_grammar_mode: Whether llama.cpp JSON grammar mode will be used.

        Returns:
            The (potentially augmented) user message string.
        """
        if schema is None:
            return prompt
        if use_grammar_mode:
            schema_instruction = json.dumps(schema, indent=2)
            return f"Respond with JSON matching this schema:\n{schema_instruction}\n\n{prompt}"
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
        """Generate a response using the local llama.cpp engine.

        When *schema* is provided or *constraints* is ``"json"``, JSON grammar
        mode is activated via ``response_format={"type": "json_object"}``,
        which constrains token sampling to structurally valid JSON.  The
        blocking ``create_chat_completion`` call is dispatched to a thread-pool
        executor so the event loop remains responsive.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.  Activates JSON grammar mode and
                embeds the schema in the user message as an instruction.
            constraints: Pass ``"json"`` to activate JSON grammar mode even
                without an explicit schema.
            kv_cache_prefix: Ignored; this backend does not support prefix
                caching via the Python binding.
            temperature: Sampling temperature.  Defaults to ``0.0`` for
                greedy/deterministic decoding.
            top_p: Nucleus-sampling probability cutoff.  ``None`` defers to the
                llama.cpp default.
            top_k: Top-k sampling parameter.  ``None`` defers to the llama.cpp
                default.
            max_tokens: Maximum number of tokens to generate.  Defaults to
                ``512`` when not specified.
            seed: Random seed for reproducible sampling.  ``None`` defers to
                the llama.cpp default.
            frequency_penalty: Frequency penalty applied to repeated tokens.
                ``None`` defers to the llama.cpp default.
            presence_penalty: Presence penalty applied to tokens that have
                already appeared.  ``None`` defers to the llama.cpp default.
            stop: Stop sequence(s).  A single string is wrapped in a list
                automatically.  ``None`` defers to the llama.cpp default.

        Returns:
            The model's response as a plain string.

        Raises:
            ImportError: If ``llama-cpp-python`` is not installed.
        """
        import asyncio

        use_grammar = schema is not None or constraints == "json"

        def _run() -> str:
            llm = self._get_llm()
            kwargs: dict[str, Any] = {
                "max_tokens": max_tokens or 512,
                "temperature": temperature if temperature is not None else 0.0,
            }
            if top_p is not None:
                kwargs["top_p"] = top_p
            if top_k is not None:
                kwargs["top_k"] = top_k
            if seed is not None:
                kwargs["seed"] = seed
            if frequency_penalty is not None:
                kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                kwargs["presence_penalty"] = presence_penalty
            if stop is not None:
                kwargs["stop"] = stop if isinstance(stop, list) else [stop]
            if use_grammar:
                kwargs["response_format"] = {"type": "json_object"}

            user_content = self._build_user_content(prompt, schema, use_grammar)
            result = llm.create_chat_completion(
                messages=[{"role": "user", "content": user_content}],
                **kwargs,
            )
            content: str = result["choices"][0]["message"]["content"] or ""
            return content

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

        Because the llama.cpp Python binding's streaming support is handled
        through synchronous iteration, this implementation delegates to
        :meth:`generate` and emits a single ``"complete"`` event once
        generation finishes.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict; activates JSON grammar mode.
            constraints: Pass ``"json"`` to activate JSON grammar mode.
            temperature: Sampling temperature.
            top_p: Nucleus-sampling probability cutoff.
            top_k: Top-k sampling parameter.
            max_tokens: Maximum number of tokens to generate.
            seed: Random seed for reproducible sampling.
            frequency_penalty: Frequency penalty applied to repeated tokens.
            presence_penalty: Presence penalty applied to already-seen tokens.
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
