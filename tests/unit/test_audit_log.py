"""Unit tests for tamper-evident audit logging."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.core import FormatShield
from formatshield.observability.audit_log import (
    FileAuditLogger,
    InMemoryAuditLogger,
    verify_audit_manifest,
    write_audit_manifest,
)


def test_audit_logger_chain_verifies_after_multiple_events() -> None:
    logger = InMemoryAuditLogger()
    logger.record("event.one", {"x": 1})
    logger.record("event.two", {"y": 2})

    assert len(logger.events()) == 2
    assert logger.verify_chain() is True


def test_core_emits_audit_events_on_generation() -> None:
    audit = InMemoryAuditLogger()

    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        shield = FormatShield(
            model="dryrun/test",
            backend=DryRunBackend(base_latency_ms=0.0),
            audit_logger=audit,
        )

    result = shield.generate_sync(
        "Hello",
        schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        },
    )

    assert result.routing.strategy in {"direct", "ttf"}
    event_types = [event.event_type for event in audit.events()]
    assert "routing.decision" in event_types
    assert "generation.complete" in event_types
    assert audit.verify_chain() is True


def test_file_audit_logger_persists_events_and_chain(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit-events.ndjson"

    logger = FileAuditLogger(audit_path)
    logger.record("routing.decision", {"strategy": "ttf"})
    logger.record("generation.complete", {"schema_valid": True})

    reloaded = FileAuditLogger(audit_path)
    events = reloaded.events()

    assert len(events) == 2
    assert events[0].event_type == "routing.decision"
    assert events[1].event_type == "generation.complete"
    assert reloaded.verify_chain() is True


def test_file_audit_logger_detects_tampered_event(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit-events.ndjson"

    logger = FileAuditLogger(audit_path)
    logger.record("event.one", {"x": 1})
    logger.record("event.two", {"y": 2})

    lines = audit_path.read_text(encoding="utf-8").splitlines()
    second = json.loads(lines[1])
    second["payload"] = {"y": 999}
    lines[1] = json.dumps(second, sort_keys=True, separators=(",", ":"))
    audit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    reloaded = FileAuditLogger(audit_path)
    assert len(reloaded.events()) == 2
    assert reloaded.verify_chain() is False


def test_audit_manifest_round_trip_verification(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit-events.ndjson"
    manifest_path = tmp_path / "audit-manifest.json"

    logger = FileAuditLogger(audit_path)
    logger.record("routing.decision", {"strategy": "ttf"})
    logger.record("generation.complete", {"schema_valid": True})

    manifest = write_audit_manifest(
        audit_path=audit_path,
        manifest_path=manifest_path,
        signing_key="test-secret",
        signing_key_id="test-kid",
    )

    assert manifest.event_count == 2
    assert manifest.signature is not None
    assert manifest.signature_key_id == "test-kid"

    valid, issues, loaded = verify_audit_manifest(
        audit_path=audit_path,
        manifest_path=manifest_path,
        signing_key="test-secret",
        expected_signing_key_id="test-kid",
    )

    assert loaded is not None
    assert valid is True
    assert issues == []


def test_audit_manifest_detects_audit_drift(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit-events.ndjson"
    manifest_path = tmp_path / "audit-manifest.json"

    logger = FileAuditLogger(audit_path)
    logger.record("event.one", {"x": 1})
    write_audit_manifest(audit_path=audit_path, manifest_path=manifest_path)

    logger.record("event.two", {"y": 2})

    valid, issues, _ = verify_audit_manifest(
        audit_path=audit_path,
        manifest_path=manifest_path,
    )

    assert valid is False
    assert any("Event count mismatch" in issue for issue in issues)


def test_audit_manifest_detects_signature_mismatch(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit-events.ndjson"
    manifest_path = tmp_path / "audit-manifest.json"

    logger = FileAuditLogger(audit_path)
    logger.record("event.one", {"x": 1})

    write_audit_manifest(
        audit_path=audit_path,
        manifest_path=manifest_path,
        signing_key="correct-secret",
        signing_key_id="correct-kid",
    )

    valid, issues, _ = verify_audit_manifest(
        audit_path=audit_path,
        manifest_path=manifest_path,
        signing_key="wrong-secret",
        expected_signing_key_id="correct-kid",
    )

    assert valid is False
    assert any("signature mismatch" in issue.lower() for issue in issues)


def test_audit_manifest_detects_signing_key_id_mismatch(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit-events.ndjson"
    manifest_path = tmp_path / "audit-manifest.json"

    logger = FileAuditLogger(audit_path)
    logger.record("event.one", {"x": 1})

    write_audit_manifest(
        audit_path=audit_path,
        manifest_path=manifest_path,
        signing_key="secret",
        signing_key_id="kid-a",
    )

    valid, issues, _ = verify_audit_manifest(
        audit_path=audit_path,
        manifest_path=manifest_path,
        signing_key="secret",
        expected_signing_key_id="kid-b",
    )

    assert valid is False
    assert any("key id mismatch" in issue.lower() for issue in issues)
