"""
Regression test — Loop 1 gap: FormatShield.__init__() must accept backend= kwarg.

Gap found: FormatShield did not accept a pre-built backend instance.
This prevented DryRunBackend injection in e2e and unit tests, requiring
the hacky FormatShield.__new__() + manual attribute setting pattern.

Fix applied: Added backend= parameter to FormatShield.__init__(); when
provided, it overrides _build_backend() entirely. Also added "dryrun" to
BackendName and _PREFIX_TO_BACKEND so model="dryrun/test" auto-resolves.

Without this fix: tests/e2e/test_full_pipeline.py fails with:
  TypeError: FormatShield.__init__() got an unexpected keyword argument 'backend'
"""

from __future__ import annotations

import pytest

from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.core import FormatShield


def test_backend_kwarg_accepted() -> None:
    """FormatShield(model=..., backend=DryRunBackend()) must not raise TypeError."""
    backend = DryRunBackend()
    shield = FormatShield(model="dryrun/test", backend=backend)
    assert shield._backend is backend


def test_injected_backend_is_used_not_replaced() -> None:
    """When backend= is supplied, _build_backend() must NOT be called."""
    backend = DryRunBackend()
    shield = FormatShield(model="groq/llama-3.1-70b-versatile", backend=backend)
    # Even though model says groq/, the injected DryRunBackend is used
    assert shield._backend is backend
    assert shield._backend.name == "dryrun"


def test_dryrun_model_prefix_resolves_correctly() -> None:
    """model='dryrun/...' without explicit backend= must auto-resolve to DryRunBackend."""
    shield = FormatShield(model="dryrun/test")
    assert shield._backend.name == "dryrun"
    assert shield.backend_name == "dryrun"


def test_generate_sync_with_injected_backend_succeeds() -> None:
    """generate_sync() works when backend= is injected — full end-to-end sanity check."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    result = shield.generate_sync("What is 2+2?")
    assert result is not None
    assert result.backend == "dryrun"


@pytest.mark.asyncio
async def test_generate_async_with_injected_backend_succeeds() -> None:
    """generate() async works with injected DryRunBackend."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    result = await shield.generate("What is 2+2?")
    assert result is not None
    assert result.backend == "dryrun"
