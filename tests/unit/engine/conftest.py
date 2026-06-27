"""Shared fixtures for engine tests: a deterministic in-memory provider.

The mock provider echoes a fixed SFEP response on every ``complete`` call, so
the full S0-S6 pipeline runs end-to-end without any network access. Tests patch
``from_model`` (as seen inside the async engine) to return it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable


class MockProvider:
    """Provider stub that returns a canned SFEP response for every call.

    A small context window keeps the capacity math simple; the canned SFEP
    should cover the field paths a test's schema produces.
    """

    model_name = "mock/echo"

    def __init__(
        self,
        sfep_text: str,
        *,
        context_window: int = 8192,
        max_output_tokens: int = 8192,
    ) -> None:
        self._sfep = sfep_text
        self.calls = 0
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        self.last_messages: list[dict[str, str]] = []

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.calls += 1
        self.last_messages = messages
        return self._sfep


@pytest.fixture
def install_provider(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], MockProvider]:
    """Return a helper that patches ``from_model`` to yield a MockProvider.

    Usage::

        provider = install_provider("name = Alice\\nage = 30")
        result = nfield(doc, schema, "mock/echo")
    """

    def _install(sfep_text: str) -> MockProvider:
        provider = MockProvider(sfep_text)
        monkeypatch.setattr(
            "nfield.engine._async.from_model",
            lambda _model, **_kwargs: provider,
        )
        return provider

    return _install
