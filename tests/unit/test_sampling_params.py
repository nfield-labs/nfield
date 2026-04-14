"""Tests for sampling parameter passthrough."""

import asyncio

from formatshield.backends.dryrun_backend import DryRunBackend


def test_dryrun_accepts_temperature():
    backend = DryRunBackend()
    result = asyncio.run(backend.generate("test", temperature=0.7))
    assert isinstance(result, str)


def test_dryrun_accepts_max_tokens():
    backend = DryRunBackend()
    result = asyncio.run(backend.generate("test", max_tokens=100))
    assert isinstance(result, str)


def test_dryrun_accepts_all_sampling_params():
    backend = DryRunBackend()
    result = asyncio.run(
        backend.generate(
            "test",
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            max_tokens=200,
            seed=42,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
        )
    )
    assert isinstance(result, str)


def test_dryrun_default_behavior_unchanged():
    backend = DryRunBackend()
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = asyncio.run(backend.generate("test", schema=schema))
    assert isinstance(result, str)


def test_dryrun_accepts_top_k():
    backend = DryRunBackend()
    result = asyncio.run(backend.generate("test", top_k=50))
    assert isinstance(result, str)


def test_dryrun_accepts_seed():
    backend = DryRunBackend()
    result = asyncio.run(backend.generate("test", seed=123))
    assert isinstance(result, str)


def test_dryrun_accepts_stop_string():
    backend = DryRunBackend()
    result = asyncio.run(backend.generate("test", stop="STOP"))
    assert isinstance(result, str)


def test_dryrun_accepts_stop_list():
    backend = DryRunBackend()
    result = asyncio.run(backend.generate("test", stop=["STOP", "END"]))
    assert isinstance(result, str)


def test_dryrun_accepts_frequency_penalty():
    backend = DryRunBackend()
    result = asyncio.run(backend.generate("test", frequency_penalty=0.5))
    assert isinstance(result, str)


def test_dryrun_accepts_presence_penalty():
    backend = DryRunBackend()
    result = asyncio.run(backend.generate("test", presence_penalty=0.5))
    assert isinstance(result, str)


def test_dryrun_stream_accepts_all_sampling_params():
    """DryRunBackend.stream() should accept all sampling params without error."""
    backend = DryRunBackend()

    async def _run() -> list[str]:
        events = []
        async for event in await backend.stream(
            "test",
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_tokens=100,
            seed=1,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END"],
        ):
            events.append(event.type)
        return events

    event_types = asyncio.run(_run())
    assert "complete" in event_types


def test_sampling_params_are_keyword_only():
    """Verify sampling params cannot be passed positionally."""
    import inspect

    from formatshield.backends.dryrun_backend import DryRunBackend

    sig = inspect.signature(DryRunBackend.generate)
    params = sig.parameters
    sampling_params = [
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "stop",
    ]
    for name in sampling_params:
        assert name in params, f"'{name}' not in DryRunBackend.generate signature"
        assert params[name].kind == inspect.Parameter.KEYWORD_ONLY, (
            f"'{name}' should be keyword-only"
        )
