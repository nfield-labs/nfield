"""Live review test for the Section 5 public API against a real Groq model.

Drives the real model entirely through ``nfield`` / ``AsyncFormatShield`` (no
raw stage calls), using a Pydantic model that REUSES a sub-model (``Party`` for
both buyer and seller). That diamond is the regression guard: a correct
flattener yields all eight leaf fields; the old global-ref bug dropped the
seller branch to six.

Requires GROQ_API_KEY. Skips otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from formatshield import AsyncFormatShield, ExtractionResult, ExtractionStatus, nfield
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

pydantic = pytest.importorskip("pydantic")


class Party(pydantic.BaseModel):
    name: str
    city: str


class Contract(pydantic.BaseModel):
    contract_id: str
    buyer: Party
    seller: Party  # reused sub-model → diamond; both branches must survive
    total_amount: float
    currency: str
    signed: bool


_DOCUMENT = (
    "CONTRACT C-9001\n"
    "Buyer: Acme Corporation, located in Boston.\n"
    "Seller: Globex Incorporated, located in Chicago.\n"
    "Total Amount: 50000.00 USD.\n"
    "Signed: yes.\n"
)

# contract_id, buyer.name, buyer.city, seller.name, seller.city,
# total_amount, currency, signed
_EXPECTED_FIELDS = 8


@pytest.mark.skipif(not _GROQ_API_KEY, reason="GROQ_API_KEY not set")
def test_public_nfield_pydantic_diamond_live():
    result = nfield(_DOCUMENT, Contract, _MODEL, config=ExtractionConfig(max_retry_rounds=1))

    assert isinstance(result, ExtractionResult)
    assert result.status in {
        ExtractionStatus.SUCCESS,
        ExtractionStatus.PARTIAL,
        ExtractionStatus.FAILED,
    }
    # The diamond must fully expand: both buyer and seller sub-trees are present.
    assert result.metadata.fields_total == _EXPECTED_FIELDS
    # Buyer is unambiguous in the document; a working pipeline recovers it.
    assert result.data.get("buyer", {}).get("name"), f"buyer.name missing: {result.data!r}"


@pytest.mark.skipif(not _GROQ_API_KEY, reason="GROQ_API_KEY not set")
async def test_reused_engine_calibrates_once_live():
    engine = AsyncFormatShield(_MODEL, Contract, config=ExtractionConfig(max_retry_rounds=0))
    first = await engine.extract(_DOCUMENT)
    second = await engine.extract(_DOCUMENT.replace("Boston", "Seattle"))
    assert first.metadata.fields_total == _EXPECTED_FIELDS
    assert second.metadata.fields_total == _EXPECTED_FIELDS
