"""Raw-prompt baseline - the format-tax floor.

No enforcement of any kind: ask the model for JSON in plain text and parse
whatever comes back. This is the lower bound the design calls the "format-tax
floor" - what you get with a single call and no structured-output tooling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import _common

if TYPE_CHECKING:
    from ._base import AdapterOutput

_TEMPERATURE: float = 0.0


@dataclass(frozen=True, slots=True)
class RawPromptAdapter:
    """Single uncontrolled call to the model; parse JSON out of the text.

    Args:
        api_key: Provider credential; ``None`` uses the SDK's env pickup.
        base_url: Optional provider base URL.
    """

    name: str = field(default="raw_prompt", init=False)
    api_key: str | None = None
    base_url: str | None = None

    def run(
        self,
        document: str,
        schema: dict[str, Any],
        *,
        model: str,
        context_window: int,
        max_output_tokens: int,
        instructions: str = "",
    ) -> AdapterOutput:
        """Extract by asking for JSON in free text and parsing the response."""
        started = time.perf_counter()
        try:
            client = _common.groq_client(self.api_key, self.base_url)
            response = client.chat.completions.create(
                model=_common.model_id(model),
                # SDK message params are TypedDicts; plain dicts are accepted at runtime.
                messages=_common.messages(  # type: ignore[arg-type]
                    document,
                    schema,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    instructions=instructions,
                ),
                max_tokens=max_output_tokens,
                temperature=_TEMPERATURE,
            )
            data = _common.parse_json_object(response.choices[0].message.content or "")
        except Exception as exc:  # record fairly, never abort the sweep
            return _common.failure_output(schema, round(time.perf_counter() - started, 3), exc)
        return _common.success_output(data, schema, round(time.perf_counter() - started, 3))
