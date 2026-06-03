"""Live end-to-end test of the public ``nfield`` API against a real Groq model.

The smallest possible full-stack check: a 3-field schema over a one-line
document, driven entirely through the public entry point. It proves the engine
wires the provider, the pipeline, and the result together correctly.

Requires GROQ_API_KEY. Skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from formatshield import ExtractionResult, ExtractionStatus, nfield
from formatshield.config import ExtractionConfig

# Load .env at import time (same convention as the other live tests).
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
_MODEL = "groq/llama-3.1-8b-instant"

_DOCUMENT = "INVOICE #4471\nVendor: Acme Corporation\nTotal Due: 1284.50 USD\nStatus: paid\n"

_SCHEMA = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string"},
        "vendor": {"type": "string"},
        "total": {"type": "number"},
    },
    "required": ["vendor"],
}


@pytest.mark.skipif(not _GROQ_API_KEY, reason="GROQ_API_KEY not set")
def test_nfield_end_to_end_live():
    result = nfield(_DOCUMENT, _SCHEMA, _MODEL, config=ExtractionConfig(max_retry_rounds=1))

    assert isinstance(result, ExtractionResult)
    assert result.status in {
        ExtractionStatus.SUCCESS,
        ExtractionStatus.PARTIAL,
        ExtractionStatus.FAILED,
    }
    assert result.metadata.fields_total == 3
    # The vendor is unambiguous in the document; a working pipeline recovers it.
    assert result.data.get("vendor"), f"vendor not extracted: {result.data!r}"
