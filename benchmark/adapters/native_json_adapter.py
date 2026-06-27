"""Native JSON-mode baseline - the provider's ``response_format`` enforcement.

The same single call as the raw baseline, but with the provider's JSON mode
turned on so the model is constrained to emit syntactically valid JSON. This
isolates the value of provider-level JSON enforcement: valid syntax does not
imply correct *values*, which is exactly the JSON-pass-vs-VA gap the benchmark
reports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import _common

if TYPE_CHECKING:
    from ._base import AdapterOutput

_TEMPERATURE: float = 0.0
_JSON_RESPONSE_FORMAT: dict[str, str] = {"type": "json_object"}


@dataclass(frozen=True, slots=True)
class NativeJsonAdapter:
    """Single call with provider JSON mode; parse the guaranteed-valid JSON.

    Args:
        api_key: Provider credential; ``None`` uses the SDK's env pickup.
        base_url: Optional provider base URL.
    """

    name: str = field(default="native_json", init=False)
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
        """Extract via the provider's JSON response format in a single call."""
        started = time.perf_counter()
        try:
            client = _common.groq_client(self.api_key, self.base_url)
            response = client.chat.completions.create(  # type: ignore[call-overload]
                model=_common.model_id(model),
                # SDK message params are TypedDicts; plain dicts are accepted at runtime.
                messages=_common.messages(
                    document,
                    schema,
                    context_window=context_window,
                    max_output_tokens=max_output_tokens,
                    instructions=instructions,
                ),
                max_tokens=max_output_tokens,
                temperature=_TEMPERATURE,
                response_format=_JSON_RESPONSE_FORMAT,
            )
            data = _common.parse_json_object(response.choices[0].message.content or "")
        except Exception as exc:  # record fairly, never abort the sweep
            return _common.failure_output(schema, round(time.perf_counter() - started, 3), exc)
        return _common.success_output(data, schema, round(time.perf_counter() - started, 3))
