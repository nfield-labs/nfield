"""AWS Bedrock inference backend for FormatShield."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from typing import Any

from formatshield.scorer.features import StreamEvent


class BedrockBackend:
    """AWS Bedrock inference backend using the Converse API.

    Routes requests through Amazon Bedrock's managed inference service.
    Authentication is resolved from explicit keyword arguments, then from
    the standard AWS environment variables.  The backend uses
    ``boto3.Session.client("bedrock-runtime")`` and is thread-safe across
    concurrent async calls via :func:`asyncio.get_event_loop().run_in_executor`.

    Args:
        model: Bedrock model ID.  Accepts both plain IDs (e.g.
            ``"anthropic.claude-3-5-sonnet-20241022-v2:0"``) and the
            ``"bedrock/<model-id>"`` prefixed format used by the FormatShield
            router.
        region: AWS region name.  Defaults to the ``AWS_DEFAULT_REGION``
            environment variable, or ``"us-east-1"`` if that is not set.
        aws_access_key_id: AWS access key.  Defaults to the
            ``AWS_ACCESS_KEY_ID`` environment variable.
        aws_secret_access_key: AWS secret key.  Defaults to the
            ``AWS_SECRET_ACCESS_KEY`` environment variable.
    """

    #: Backend identifier consumed by the FormatShield router.
    name: str = "bedrock"

    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
    ) -> None:
        self._model_id = model.removeprefix("bedrock/")
        self._region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._aws_access_key_id = aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
        self._aws_secret_access_key = aws_secret_access_key or os.environ.get(
            "AWS_SECRET_ACCESS_KEY"
        )
        self._client: Any = None

    # ------------------------------------------------------------------
    # Capability properties
    # ------------------------------------------------------------------

    @property
    def supports_kv_cache_reuse(self) -> bool:
        """Bedrock does not expose server-side KV-cache prefix reuse."""
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        """16 % baseline accuracy loss for structured-output generation on Bedrock."""
        return 0.16

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        """Return (and lazily create) the boto3 Bedrock Runtime client.

        Raises:
            ImportError: If ``boto3`` is not installed.
        """
        if self._client is None:
            try:
                import boto3  # pyright: ignore[reportMissingImports]
            except ImportError as exc:
                raise ImportError(
                    "boto3 is required for BedrockBackend. "
                    "Install with: pip install 'formatshield[bedrock]'"
                ) from exc
            session_kwargs: dict[str, Any] = {"region_name": self._region}
            if self._aws_access_key_id:
                session_kwargs["aws_access_key_id"] = self._aws_access_key_id
            if self._aws_secret_access_key:
                session_kwargs["aws_secret_access_key"] = self._aws_secret_access_key
            session = boto3.Session(**session_kwargs)
            self._client = session.client("bedrock-runtime")
        return self._client

    def _build_messages(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build the Bedrock Converse API messages list.

        When a JSON schema is provided it is embedded directly in the user
        message so the model understands the required output structure.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema to embed in the message.

        Returns:
            A list of message dicts suitable for the Converse API.
        """
        content = prompt
        if schema:
            schema_text = json.dumps(schema, indent=2)
            content = (
                f"You must respond with valid JSON conforming to this schema:\n{schema_text}\n\n"
                f"Do not include any text outside the JSON object.\n\n{prompt}"
            )
        return [{"role": "user", "content": [{"text": content}]}]

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
        """Generate a response via the Bedrock Converse API.

        The synchronous boto3 call is dispatched to a thread-pool executor so
        that the async event loop is never blocked.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.  When supplied, the schema is
                embedded in the user message.
            constraints: Reserved for future grammar-constraint support;
                currently unused by Bedrock.
            kv_cache_prefix: Ignored; Bedrock does not support prefix caching.
            temperature: Sampling temperature.  Defaults to ``0.0`` for
                deterministic output.
            top_p: Nucleus sampling probability.  ``None`` defers to the API
                default.
            top_k: Ignored; the Converse API does not expose top-k.
            max_tokens: Maximum number of tokens to generate.  Defaults to
                ``512``.
            seed: Ignored; the Converse API does not expose a seed parameter.
            frequency_penalty: Ignored; not supported by the Converse API.
            presence_penalty: Ignored; not supported by the Converse API.
            stop: Stop sequence(s).  ``None`` defers to the API default.

        Returns:
            The model's response as a plain string.

        Raises:
            ImportError: If ``boto3`` is not installed.
            RuntimeError: If the Bedrock API call fails.
        """
        messages = self._build_messages(prompt, schema)
        inference_config: dict[str, Any] = {
            "temperature": temperature if temperature is not None else 0.0,
            "maxTokens": max_tokens or 512,
        }
        if top_p is not None:
            inference_config["topP"] = top_p
        if stop is not None:
            inference_config["stopSequences"] = stop if isinstance(stop, list) else [stop]

        def _run() -> str:
            client = self._get_client()
            try:
                response = client.converse(
                    modelId=self._model_id,
                    messages=messages,
                    inferenceConfig=inference_config,
                )
            except Exception as exc:
                raise RuntimeError(f"Bedrock API error: {exc}") from exc
            text: str = response["output"]["message"]["content"][0]["text"]
            return text

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

        Falls back to a single ``complete`` event wrapping the full response
        because the Bedrock Converse API's streaming interface requires a
        separate synchronous executor integration that is out of scope for
        this adapter.

        Args:
            prompt: The user prompt.
            schema: Optional JSON schema dict.
            constraints: Reserved for future use; currently unused.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability.
            top_k: Ignored.
            max_tokens: Maximum tokens to generate.
            seed: Ignored.
            frequency_penalty: Ignored.
            presence_penalty: Ignored.
            stop: Stop sequence(s).

        Yields:
            A single :class:`StreamEvent` of type ``"complete"``.
        """
        result = await self.generate(
            prompt,
            schema,
            constraints,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
        )
        yield StreamEvent(type="complete", content=result, backend=self.name, latency_ms=0.0)
