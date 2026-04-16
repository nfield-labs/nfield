"""
FormatShield DSL response types — Partial, Maybe, IterableModel, CitationMixin.

These types mirror the Instructor DSL pattern but integrate FormatShield's
routing layer. They are thin wrappers that carry typing metadata used by
``core.generate()`` to adjust schema generation and response parsing.

Usage::

    from formatshield.dsl import Maybe, Partial, IterableModel

    # Optional result — won't raise on model uncertainty
    result = await fs.generate(prompt, Maybe[MyModel], model=...)
    if result.parsed.result is not None:
        print(result.parsed.result.answer)
    else:
        print("Model flagged uncertainty:", result.parsed.error_message)

    # Streaming partial model — update UI progressively
    async for partial in fs.stream_partial(prompt, Partial[MyModel], model=...):
        print(partial)  # MyModel with partial fields filled in

    # Iterable — stream a list of items one by one
    async for item in fs.stream_iterable(prompt, IterableModel[MyModel], model=...):
        print(item)  # individual MyModel instances
"""

from formatshield.dsl.iterable import IterableModel
from formatshield.dsl.maybe import Maybe, MaybeResult
from formatshield.dsl.partial import Partial

__all__ = [
    "IterableModel",
    "Maybe",
    "MaybeResult",
    "Partial",
]
