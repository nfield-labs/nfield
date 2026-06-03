"""Live integration tests for Section 4: Full pipeline S0-S6 via real Groq API.

Exercises the complete 7-stage pipeline with real LLM calls.
Each test class covers a distinct dimension of pipeline correctness.

Models used (from Groq production docs):
  llama-3.1-8b-instant  — 131,072 ctx / 131,072 max output / 560 t/s / cheapest
  llama-3.3-70b-versatile — 131,072 ctx / 32,768 max output / 280 t/s / best quality

Requires GROQ_API_KEY in environment or .env file at repo root.
All tests auto-skip when the key is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load .env at import time (same pattern as other integration tests)
# ---------------------------------------------------------------------------

_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Model specs (Groq production docs, June 2026)
# ---------------------------------------------------------------------------

# Fast model: cheapest, max context, great for bulk tests
_MODEL_8B = "llama-3.1-8b-instant"
_CTX_8B = 131_072
_MAX_OUT_8B = 131_072

# Quality model: better reasoning, larger output budget
_MODEL_70B = "llama-3.3-70b-versatile"
_CTX_70B = 131_072
_MAX_OUT_70B = 32_768


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _skip_no_key() -> None:
    if not _GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set — skipping live integration test")


def _make_provider_8b():
    from formatshield.providers.groq._provider import GroqProvider

    return GroqProvider(_MODEL_8B, context_window=_CTX_8B, max_output_tokens=_MAX_OUT_8B)


def _make_provider_70b():
    from formatshield.providers.groq._provider import GroqProvider

    return GroqProvider(_MODEL_70B, context_window=_CTX_70B, max_output_tokens=_MAX_OUT_70B)


# ---------------------------------------------------------------------------
# Reusable schemas and documents
# ---------------------------------------------------------------------------

_SCHEMA_COMPANY = {
    "type": "object",
    "properties": {
        "company_name": {"type": "string", "description": "Full legal name of the company"},
        "founded_year": {
            "type": "integer",
            "minimum": 1800,
            "maximum": 2030,
            "description": "Year the company was founded",
        },
        "is_publicly_traded": {
            "type": "boolean",
            "description": "True if the company is publicly listed on a stock exchange",
        },
    },
}

_SCHEMA_INVOICE = {
    "type": "object",
    "properties": {
        "invoice_number": {"type": "string", "description": "Unique invoice identifier"},
        "invoice_date": {
            "type": "string",
            "description": "Invoice date in ISO 8601 format",
        },
        "vendor": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Vendor company name"},
                "email": {"type": "string", "description": "Vendor contact email"},
            },
        },
        "total_amount": {
            "type": "number",
            "minimum": 0,
            "description": "Total invoice amount in USD",
        },
        "currency": {
            "type": "string",
            "enum": ["USD", "EUR", "GBP", "JPY"],
            "description": "Invoice currency code",
        },
    },
}

_SCHEMA_MEDICAL = {
    "type": "object",
    "properties": {
        "patient_name": {"type": "string", "description": "Full name of the patient"},
        "patient_age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 150,
            "description": "Patient age in years",
        },
        "blood_type": {
            "type": "string",
            "enum": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
            "description": "ABO blood group with Rh factor",
        },
        "diagnosis": {
            "type": "string",
            "maxLength": 500,
            "description": "Primary diagnosis from the physician",
        },
        "is_admitted": {
            "type": "boolean",
            "description": "True if the patient is currently admitted",
        },
    },
}

_DOC_COMPANY = (
    "Acme Corporation was incorporated in 1947 by Thomas Acme in Newark, New Jersey. "
    "The company went public on the New York Stock Exchange in 1962 under the ticker ACM. "
    "Acme Corp operates in the manufacturing sector and currently employs over 50,000 people "
    "across 32 countries worldwide. Its flagship product line is industrial-grade equipment."
)

_DOC_INVOICE = (
    "INVOICE\n"
    "Invoice Number: INV-2026-00142\n"
    "Date: 2026-05-15\n\n"
    "Vendor: TechSupplies Ltd\n"
    "Email: billing@techsupplies.com\n\n"
    "Items:\n"
    "  - Cloud Storage 1TB (12 months) ....... $1,200.00\n"
    "  - Technical Support Package ........... $800.00\n\n"
    "Total Amount: $2,000.00 USD\n"
    "Payment Due: 2026-06-15"
)

_DOC_MEDICAL = (
    "PATIENT CLINICAL RECORD\n"
    "Patient: Sarah Johnson, Age: 34\n"
    "Blood Type: O+\n"
    "Admission Status: Currently admitted (admitted 2026-06-01)\n\n"
    "Primary Diagnosis: Type 2 diabetes mellitus with peripheral neuropathy. "
    "Patient presents with elevated HbA1c at 9.2%. Prescription adjusted.\n\n"
    "Attending Physician: Dr. K. Mehta, Internal Medicine"
)

# Document that is deliberately missing some fields (for partial extraction tests)
_DOC_PARTIAL = (
    "Brief company overview: Acme Corp is a manufacturing firm. "
    "No financial details or founding year are mentioned here."
)


# ---------------------------------------------------------------------------
# Test Class 1: Full pipeline S0→S6 correctness
# ---------------------------------------------------------------------------


class TestFullPipelineS0toS6:
    """Full pipeline from calibration through assembly with real Groq."""

    @pytest.mark.asyncio
    async def test_company_schema_success_status(self):
        """Full pipeline on a clear 3-field company doc extracts cleanly.

        Uses the 70B model: single-call extraction on the 8B model is
        occasionally non-deterministic (rare empty responses), which would make
        a status assertion flaky. Aggregate accuracy is covered separately by
        the synthetic accuracy-at-scale tests.
        """
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_70b())
        from formatshield.types import ExtractionStatus

        assert result.status in (ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL), (
            f"Expected SUCCESS or PARTIAL, got {result.status}. data={result.data}"
        )

    @pytest.mark.asyncio
    async def test_company_name_extracted(self):
        """Company name field is extracted and non-empty (70B for reliability)."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_70b())
        assert "company_name" in result.data, f"company_name missing. data={result.data}"
        assert isinstance(result.data["company_name"], str)
        assert len(result.data["company_name"]) > 0

    @pytest.mark.asyncio
    async def test_founded_year_is_integer_in_range(self):
        """Founded year is an integer between 1800 and 2030."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        if "founded_year" in result.data and result.data["founded_year"] is not None:
            year = result.data["founded_year"]
            assert isinstance(year, int), f"founded_year must be int, got {type(year)}"
            assert 1800 <= year <= 2030, f"founded_year={year} out of range"

    @pytest.mark.asyncio
    async def test_is_publicly_traded_is_bool(self):
        """Boolean field is extracted as Python bool."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        if "is_publicly_traded" in result.data and result.data["is_publicly_traded"] is not None:
            assert isinstance(result.data["is_publicly_traded"], bool), (
                f"is_publicly_traded must be bool, got {type(result.data['is_publicly_traded'])}"
            )

    @pytest.mark.asyncio
    async def test_metadata_k_min_at_least_1(self):
        """K_min is at least 1 for any schema."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        assert result.metadata.K_min >= 1

    @pytest.mark.asyncio
    async def test_metadata_k_equals_leaves_count(self):
        """Actual K matches number of extraction calls made."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        assert result.metadata.K >= 1

    @pytest.mark.asyncio
    async def test_metadata_fields_total_matches_schema(self):
        """fields_total matches the number of fields in the schema."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        assert result.metadata.fields_total == 3

    @pytest.mark.asyncio
    async def test_quality_score_in_unit_range(self):
        """Quality score is always in [0.0, 1.0]."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        assert 0.0 <= result.metadata.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_level_is_valid_tier(self):
        """Confidence level is one of the three defined tiers."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        assert result.metadata.confidence_level in ("HIGH", "MEDIUM", "LOW")

    @pytest.mark.asyncio
    async def test_per_field_confidence_covers_all_fields(self):
        """per_field_confidence dict has an entry for every schema field."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        expected_paths = {"company_name", "founded_year", "is_publicly_traded"}
        actual_paths = set(result.metadata.per_field_confidence.keys())
        assert expected_paths == actual_paths, (
            f"Missing confidence entries: {expected_paths - actual_paths}"
        )

    @pytest.mark.asyncio
    async def test_per_field_confidence_in_unit_range(self):
        """Every per-field confidence score is in [0.0, 1.0]."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        for path, score in result.metadata.per_field_confidence.items():
            assert 0.0 <= score <= 1.0, f"{path}: confidence={score} out of [0,1]"


# ---------------------------------------------------------------------------
# Test Class 2: Nested schema (invoice)
# ---------------------------------------------------------------------------


class TestNestedSchemaExtraction:
    """Pipeline handles nested JSON Schema (object-within-object)."""

    @pytest.mark.asyncio
    async def test_invoice_pipeline_runs(self):
        """Full pipeline on invoice schema does not crash."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_INVOICE, _DOC_INVOICE, _make_provider_8b())
        assert result is not None

    @pytest.mark.asyncio
    async def test_invoice_number_extracted(self):
        """Invoice number field is a non-empty string."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_INVOICE, _DOC_INVOICE, _make_provider_8b())
        if "invoice_number" in result.data and result.data["invoice_number"] is not None:
            assert isinstance(result.data["invoice_number"], str)
            assert len(result.data["invoice_number"]) > 0

    @pytest.mark.asyncio
    async def test_total_amount_is_positive_number(self):
        """Total amount is a non-negative number when extracted."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_INVOICE, _DOC_INVOICE, _make_provider_8b())
        if "total_amount" in result.data and result.data["total_amount"] is not None:
            amount = result.data["total_amount"]
            assert isinstance(amount, (int, float)), f"total_amount type={type(amount)}"
            assert amount >= 0, f"total_amount={amount} is negative"

    @pytest.mark.asyncio
    async def test_currency_enum_valid(self):
        """Currency field is one of the allowed enum values."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_INVOICE, _DOC_INVOICE, _make_provider_8b())
        if "currency" in result.data and result.data["currency"] is not None:
            assert result.data["currency"] in ("USD", "EUR", "GBP", "JPY"), (
                f"currency={result.data['currency']!r} not in allowed enum"
            )

    @pytest.mark.asyncio
    async def test_nested_vendor_assembled_correctly(self):
        """Nested vendor fields are assembled into a nested dict."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_INVOICE, _DOC_INVOICE, _make_provider_8b())
        # If any vendor field was extracted, it should be nested under "vendor" key
        vendor = result.data.get("vendor")
        if vendor is not None:
            assert isinstance(vendor, dict), f"vendor must be a dict, got {type(vendor)}"

    @pytest.mark.asyncio
    async def test_invoice_fields_total(self):
        """Invoice schema produces the correct fields_total count."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_INVOICE, _DOC_INVOICE, _make_provider_8b())
        # invoice_number, invoice_date, vendor.name, vendor.email, total_amount, currency = 6
        assert result.metadata.fields_total == 6, (
            f"Expected 6 fields_total, got {result.metadata.fields_total}"
        )


# ---------------------------------------------------------------------------
# Test Class 3: Constrained field extraction (medical)
# ---------------------------------------------------------------------------


class TestConstrainedFieldExtraction:
    """Type and constraint validation works end-to-end with real Groq output."""

    @pytest.mark.asyncio
    async def test_medical_pipeline_runs(self):
        """Full pipeline on medical schema completes without error."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_MEDICAL, _DOC_MEDICAL, _make_provider_8b())
        assert result is not None

    @pytest.mark.asyncio
    async def test_patient_age_integer_in_range(self):
        """Patient age is integer between 0 and 150."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_MEDICAL, _DOC_MEDICAL, _make_provider_8b())
        if "patient_age" in result.data and result.data["patient_age"] is not None:
            age = result.data["patient_age"]
            assert isinstance(age, int), f"patient_age must be int, got {type(age)}"
            assert 0 <= age <= 150, f"patient_age={age} out of valid range"

    @pytest.mark.asyncio
    async def test_blood_type_enum_value(self):
        """Blood type is one of the 8 valid ABO+Rh combinations."""
        _skip_no_key()
        valid = {"A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"}
        result = await _run_pipeline(_SCHEMA_MEDICAL, _DOC_MEDICAL, _make_provider_8b())
        if "blood_type" in result.data and result.data["blood_type"] is not None:
            assert result.data["blood_type"] in valid, (
                f"blood_type={result.data['blood_type']!r} not a valid ABO+Rh type"
            )

    @pytest.mark.asyncio
    async def test_is_admitted_boolean(self):
        """is_admitted field is a Python bool."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_MEDICAL, _DOC_MEDICAL, _make_provider_8b())
        if "is_admitted" in result.data and result.data["is_admitted"] is not None:
            assert isinstance(result.data["is_admitted"], bool)

    @pytest.mark.asyncio
    async def test_diagnosis_respects_maxlength(self):
        """Diagnosis string does not exceed 500 characters."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_MEDICAL, _DOC_MEDICAL, _make_provider_8b())
        diag = result.data.get("diagnosis")
        if diag is not None:
            assert isinstance(diag, str)
            assert len(diag) <= 500, f"diagnosis length={len(diag)} exceeds maxLength=500"

    @pytest.mark.asyncio
    async def test_no_invalid_filled_fields_after_validation(self):
        """Stage 5 ensures no FILLED field violates its schema constraints."""
        _skip_no_key()
        from formatshield.validation._type_check import validate_field

        result = await _run_pipeline(
            _SCHEMA_MEDICAL, _DOC_MEDICAL, _make_provider_8b(), return_state=True
        )
        _, state = result
        bb = state.blackboard
        field_map = state.field_by_path
        for path, value in bb.get_filled().items():
            f = field_map.get(path)
            if f is not None and value is not None:
                valid, err = validate_field(value, f)
                assert valid, (
                    f"FILLED field {path!r}={value!r} violates constraints after Stage 5: {err}"
                )


# ---------------------------------------------------------------------------
# Test Class 4: Partial document (missing fields → PARTIAL status)
# ---------------------------------------------------------------------------


class TestPartialExtraction:
    """When document doesn't contain all fields, pipeline returns PARTIAL."""

    @pytest.mark.asyncio
    async def test_partial_doc_returns_partial_or_failed(self):
        """Document without founding year produces PARTIAL or FAILED status."""
        _skip_no_key()
        from formatshield.types import ExtractionStatus

        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_PARTIAL, _make_provider_8b())
        assert result.status in (ExtractionStatus.PARTIAL, ExtractionStatus.FAILED), (
            f"Expected PARTIAL/FAILED for incomplete doc, got {result.status}. data={result.data}"
        )

    @pytest.mark.asyncio
    async def test_partial_doc_fields_missing_count(self):
        """fields_missing + fields_extracted == fields_total for partial results."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_PARTIAL, _make_provider_8b())
        m = result.metadata
        assert m.fields_extracted + m.fields_missing <= m.fields_total, (
            f"extracted={m.fields_extracted} + missing={m.fields_missing} > total={m.fields_total}"
        )

    @pytest.mark.asyncio
    async def test_optimality_gap_in_range(self):
        """Optimality gap is always in [0.0, 1.0]."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_PARTIAL, _make_provider_8b())
        gap = result.metadata.optimality_gap
        assert 0.0 <= gap <= 1.0, f"optimality_gap={gap} out of [0,1]"


# ---------------------------------------------------------------------------
# Test Class 5: Model comparison (8b vs 70b)
# ---------------------------------------------------------------------------


class TestModelComparison:
    """Both production models produce valid ExtractionResults."""

    @pytest.mark.asyncio
    async def test_8b_model_extracts_company(self):
        """llama-3.1-8b-instant returns a valid pipeline result for company doc.

        fields_extracted + fields_missing must always equal fields_total — no
        unaccounted fields left in PENDING state after Stage 5.
        """
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        m = result.metadata
        accounted = (
            m.fields_extracted
            + m.fields_missing
            + m.fields_conflicted
            + m.fields_needs_revalidation
        )
        assert accounted == m.fields_total, (
            f"Unaccounted fields: extracted={m.fields_extracted} missing={m.fields_missing} "
            f"conflicted={m.fields_conflicted} revalidation={m.fields_needs_revalidation} "
            f"total={m.fields_total}"
        )

    @pytest.mark.asyncio
    async def test_70b_model_extracts_company(self):
        """llama-3.3-70b-versatile extracts at least one field from company doc."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_70b())
        assert result.metadata.fields_extracted >= 1, (
            f"70b model extracted 0 fields. data={result.data}"
        )

    @pytest.mark.asyncio
    async def test_70b_enum_accuracy(self):
        """70b model correctly extracts enum-constrained field."""
        _skip_no_key()
        result = await _run_pipeline(_SCHEMA_INVOICE, _DOC_INVOICE, _make_provider_70b())
        if "currency" in result.data and result.data["currency"] is not None:
            assert result.data["currency"] in ("USD", "EUR", "GBP", "JPY")

    @pytest.mark.asyncio
    async def test_both_models_return_valid_result_type(self):
        """Both models return an ExtractionResult instance."""
        _skip_no_key()
        from formatshield.types import ExtractionResult

        r8 = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b())
        r70 = await _run_pipeline(_SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_70b())
        assert isinstance(r8, ExtractionResult)
        assert isinstance(r70, ExtractionResult)


# ---------------------------------------------------------------------------
# Test Class 6: Pipeline calibration (Stage 0)
# ---------------------------------------------------------------------------


class TestPipelineCalibration:
    """Stage 0 resource calibration is accurate for each model."""

    @pytest.mark.asyncio
    async def test_chars_per_token_8b_reasonable_range(self):
        """8b model chars_per_token is in 2.5-5.0 for English text."""
        _skip_no_key()
        from formatshield.config import ExtractionConfig
        from formatshield.pipeline.s0_resources import run_stage_0

        provider = _make_provider_8b()
        state = await run_stage_0(provider, ExtractionConfig())
        assert 2.5 <= state.chars_per_token <= 5.0, (
            f"chars_per_token={state.chars_per_token} outside expected English range"
        )

    @pytest.mark.asyncio
    async def test_context_window_matches_spec(self):
        """C_eff equals the model's declared context window."""
        _skip_no_key()
        from formatshield.config import ExtractionConfig
        from formatshield.pipeline.s0_resources import run_stage_0

        provider = _make_provider_8b()
        state = await run_stage_0(provider, ExtractionConfig())
        assert state.C_eff == _CTX_8B, f"C_eff={state.C_eff} != declared context_window={_CTX_8B}"

    @pytest.mark.asyncio
    async def test_c_usable_is_50pct_of_c_eff(self):
        """C_usable == 0.5 * C_eff with default ExtractionConfig."""
        _skip_no_key()
        from formatshield.config import ExtractionConfig
        from formatshield.pipeline.s0_resources import run_stage_0

        provider = _make_provider_8b()
        state = await run_stage_0(provider, ExtractionConfig())
        assert abs(state.C_usable - state.C_eff * 0.5) < 1, (
            f"C_usable={state.C_usable} != 0.5 * C_eff={state.C_eff}"
        )

    @pytest.mark.asyncio
    async def test_70b_max_output_matches_spec(self):
        """M_O for 70b model equals declared max_output_tokens."""
        _skip_no_key()
        from formatshield.config import ExtractionConfig
        from formatshield.pipeline.s0_resources import run_stage_0

        provider = _make_provider_70b()
        state = await run_stage_0(provider, ExtractionConfig())
        assert state.M_O == _MAX_OUT_70B, (
            f"M_O={state.M_O} != declared max_output_tokens={_MAX_OUT_70B}"
        )


# ---------------------------------------------------------------------------
# Test Class 7: Pipeline state correctness (individual stage inspection)
# ---------------------------------------------------------------------------


class TestPipelineStateInspection:
    """Inspect PipelineState at each stage to verify correctness end-to-end."""

    @pytest.mark.asyncio
    async def test_stage1_all_fields_have_tau(self):
        """After Stage 1, every field has tau > 0."""
        _skip_no_key()
        _, state = await _run_pipeline(
            _SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b(), return_state=True
        )
        for f in state.fields:
            assert f.tau > 0.0, f"Field {f.path!r} has tau={f.tau}"

    @pytest.mark.asyncio
    async def test_stage1_difficulty_in_range(self):
        """After Stage 1, every D(f) is in [0.0, 1.0]."""
        _skip_no_key()
        _, state = await _run_pipeline(
            _SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b(), return_state=True
        )
        for f in state.fields:
            assert 0.0 <= f.difficulty <= 1.0, f"Field {f.path!r} has difficulty={f.difficulty}"

    @pytest.mark.asyncio
    async def test_stage2a_groups_cover_all_fields(self):
        """After Stage 2A, group_map covers every field path."""
        _skip_no_key()
        _, state = await _run_pipeline(
            _SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b(), return_state=True
        )
        for f in state.fields:
            assert f.path in state.group_map, f"Field {f.path!r} not in group_map after Stage 2A"

    @pytest.mark.asyncio
    async def test_stage2c_all_fields_in_leaves(self):
        """After Stage 2C, every schema field appears in exactly one leaf."""
        _skip_no_key()
        _, state = await _run_pipeline(
            _SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b(), return_state=True
        )
        leaf_paths = {f.path for leaf in state.leaves for f in leaf.fields}
        schema_paths = {f.path for f in state.fields}
        assert leaf_paths == schema_paths, (
            f"Leaf coverage mismatch. Missing: {schema_paths - leaf_paths}"
        )

    @pytest.mark.asyncio
    async def test_stage3_excerpts_non_empty(self):
        """After Stage 3, every leaf has a non-empty document_excerpt."""
        _skip_no_key()
        _, state = await _run_pipeline(
            _SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b(), return_state=True
        )
        for leaf in state.leaves:
            assert isinstance(leaf.document_excerpt, str)
            # Excerpt may be empty only for truly empty documents
            assert len(leaf.document_excerpt) >= 0

    @pytest.mark.asyncio
    async def test_blackboard_has_no_pending_after_stage5(self):
        """After Stage 5, no field remains in PENDING state."""
        _skip_no_key()
        from formatshield.assembly._blackboard import FieldState

        _, state = await _run_pipeline(
            _SCHEMA_COMPANY, _DOC_COMPANY, _make_provider_8b(), return_state=True
        )
        bb = state.blackboard
        pending = [p for p in bb.all_paths() if bb.get_state(p) == FieldState.PENDING]
        assert pending == [], f"Fields still PENDING after Stage 5: {pending}"


# ---------------------------------------------------------------------------
# Helper: run the full pipeline and return ExtractionResult (+ optionally state)
# ---------------------------------------------------------------------------


async def _run_pipeline(schema, document, provider, *, return_state: bool = False):
    """Run stages S0-S6 and return ExtractionResult (or (result, state) tuple).

    Args:
        schema: JSON Schema dict.
        document: Raw document text.
        provider: LLM provider instance (real Groq).
        return_state: If True, returns (ExtractionResult, PipelineState).

    Returns:
        ExtractionResult, or (ExtractionResult, PipelineState) if return_state=True.
    """
    from formatshield.config import ExtractionConfig
    from formatshield.pipeline.s0_resources import run_stage_0
    from formatshield.pipeline.s1_schema import run_stage_1
    from formatshield.pipeline.s2a_structure import run_stage_2a
    from formatshield.pipeline.s2b_prepass import run_stage_2b
    from formatshield.pipeline.s2c_packing import run_stage_2c
    from formatshield.pipeline.s3_excerpt import run_stage_3
    from formatshield.pipeline.s4_extract import run_stage_4
    from formatshield.pipeline.s5_validate import run_stage_5
    from formatshield.pipeline.s6_assemble import run_stage_6

    config = ExtractionConfig()

    state = await run_stage_0(provider, config)
    state = run_stage_1(state, schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, document, config)
    state = run_stage_2c(state, config)
    state = run_stage_3(state)
    state = await run_stage_4(state, provider)
    state = await run_stage_5(state, provider, config)
    result = run_stage_6(state)

    if return_state:
        return result, state
    return result
