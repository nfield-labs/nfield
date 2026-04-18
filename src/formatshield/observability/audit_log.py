"""Tamper-evident audit logging for routing and policy events."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class AuditEvent:
    """Single immutable audit event in the chain."""

    event_id: str
    timestamp_utc: str
    event_type: str
    payload: dict[str, Any]
    prev_hash: str
    event_hash: str

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditManifest:
    """Portable integrity manifest for an audit event file."""

    version: str
    created_at_utc: str
    audit_path: str
    event_count: int
    first_event_id: str | None
    last_event_id: str | None
    first_hash: str
    last_hash: str
    chain_valid: bool
    event_type_counts: dict[str, int]
    audit_sha256: str
    manifest_payload_hash: str
    signature_alg: str | None
    signature_key_id: str | None
    signature: str | None

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def _payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def _compute_event_hash(
    prev_hash: str,
    event_id: str,
    timestamp_utc: str,
    event_type: str,
    payload: dict[str, Any],
) -> str:
    payload_json = _payload_json(payload)
    hash_input = "|".join([prev_hash, event_id, timestamp_utc, event_type, payload_json])
    return hashlib.sha256(hash_input.encode("utf-8")).hexdigest()


def _build_event(prev_hash: str, event_type: str, payload: dict[str, Any]) -> AuditEvent:
    timestamp_utc = datetime.now(UTC).isoformat()
    event_id = uuid.uuid4().hex
    payload_copy = dict(payload)
    event_hash = _compute_event_hash(
        prev_hash=prev_hash,
        event_id=event_id,
        timestamp_utc=timestamp_utc,
        event_type=event_type,
        payload=payload_copy,
    )
    return AuditEvent(
        event_id=event_id,
        timestamp_utc=timestamp_utc,
        event_type=event_type,
        payload=payload_copy,
        prev_hash=prev_hash,
        event_hash=event_hash,
    )


def _verify_event_chain(events: list[AuditEvent]) -> bool:
    prev_hash = "GENESIS"
    for event in events:
        expected = _compute_event_hash(
            prev_hash=prev_hash,
            event_id=event.event_id,
            timestamp_utc=event.timestamp_utc,
            event_type=event.event_type,
            payload=event.payload,
        )
        if event.prev_hash != prev_hash or event.event_hash != expected:
            return False
        prev_hash = event.event_hash
    return True


def _event_from_json(payload: dict[str, Any]) -> AuditEvent:
    return AuditEvent(
        event_id=str(payload["event_id"]),
        timestamp_utc=str(payload["timestamp_utc"]),
        event_type=str(payload["event_type"]),
        payload=dict(payload["payload"]),
        prev_hash=str(payload["prev_hash"]),
        event_hash=str(payload["event_hash"]),
    )


def _manifest_payload(manifest: AuditManifest) -> dict[str, Any]:
    return {
        "version": manifest.version,
        "audit_path": manifest.audit_path,
        "event_count": manifest.event_count,
        "first_event_id": manifest.first_event_id,
        "last_event_id": manifest.last_event_id,
        "first_hash": manifest.first_hash,
        "last_hash": manifest.last_hash,
        "chain_valid": manifest.chain_valid,
        "event_type_counts": dict(manifest.event_type_counts),
        "audit_sha256": manifest.audit_sha256,
        "signature_alg": manifest.signature_alg,
        "signature_key_id": manifest.signature_key_id,
    }


def _sha256_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    if not path.exists():
        return _sha256_text("")
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _hmac_signature(message: str, key: str) -> str:
    return hmac.new(key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def _manifest_from_json(payload: dict[str, Any]) -> AuditManifest:
    return AuditManifest(
        version=str(payload["version"]),
        created_at_utc=str(payload["created_at_utc"]),
        audit_path=str(payload["audit_path"]),
        event_count=int(payload["event_count"]),
        first_event_id=str(payload["first_event_id"]) if payload.get("first_event_id") else None,
        last_event_id=str(payload["last_event_id"]) if payload.get("last_event_id") else None,
        first_hash=str(payload["first_hash"]),
        last_hash=str(payload["last_hash"]),
        chain_valid=bool(payload["chain_valid"]),
        event_type_counts={str(k): int(v) for k, v in dict(payload["event_type_counts"]).items()},
        audit_sha256=str(payload["audit_sha256"]),
        manifest_payload_hash=str(payload["manifest_payload_hash"]),
        signature_alg=(str(payload["signature_alg"]) if payload.get("signature_alg") else None),
        signature_key_id=(
            str(payload["signature_key_id"]) if payload.get("signature_key_id") else None
        ),
        signature=(str(payload["signature"]) if payload.get("signature") else None),
    )


def build_audit_manifest(
    audit_path: str | Path,
    *,
    signing_key: str | None = None,
    signing_key_id: str | None = None,
) -> AuditManifest:
    """Build a manifest snapshot for an audit NDJSON file."""
    audit_file = Path(audit_path)
    logger = FileAuditLogger(audit_file)
    events = logger.events()

    event_type_counts: dict[str, int] = {}
    for event in events:
        event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1

    first_event_id = events[0].event_id if events else None
    last_event_id = events[-1].event_id if events else None
    first_hash = events[0].prev_hash if events else "GENESIS"
    last_hash = events[-1].event_hash if events else "GENESIS"
    chain_valid = logger.verify_chain()
    signature_alg = "hmac-sha256" if signing_key else None
    signature_key_id = signing_key_id if signing_key else None

    manifest = AuditManifest(
        version="1",
        created_at_utc=datetime.now(UTC).isoformat(),
        audit_path=str(audit_file),
        event_count=len(events),
        first_event_id=first_event_id,
        last_event_id=last_event_id,
        first_hash=first_hash,
        last_hash=last_hash,
        chain_valid=chain_valid,
        event_type_counts=event_type_counts,
        audit_sha256=_sha256_file(audit_file),
        manifest_payload_hash="",
        signature_alg=signature_alg,
        signature_key_id=signature_key_id,
        signature=None,
    )
    payload_hash = _sha256_text(_canonical_json(_manifest_payload(manifest)))
    signature = _hmac_signature(payload_hash, signing_key) if signing_key else None

    return AuditManifest(
        version=manifest.version,
        created_at_utc=manifest.created_at_utc,
        audit_path=manifest.audit_path,
        event_count=manifest.event_count,
        first_event_id=manifest.first_event_id,
        last_event_id=manifest.last_event_id,
        first_hash=manifest.first_hash,
        last_hash=manifest.last_hash,
        chain_valid=manifest.chain_valid,
        event_type_counts=manifest.event_type_counts,
        audit_sha256=manifest.audit_sha256,
        manifest_payload_hash=payload_hash,
        signature_alg=signature_alg,
        signature_key_id=signature_key_id,
        signature=signature,
    )


def write_audit_manifest(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signing_key: str | None = None,
    signing_key_id: str | None = None,
) -> AuditManifest:
    """Write a JSON manifest for the provided audit file and return it."""
    manifest = build_audit_manifest(
        audit_path,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
    )
    target = Path(manifest_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.model_dump(), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify_audit_manifest(
    audit_path: str | Path,
    manifest_path: str | Path,
    *,
    signing_key: str | None = None,
    expected_signing_key_id: str | None = None,
) -> tuple[bool, list[str], AuditManifest | None]:
    """Verify manifest integrity and audit-file drift.

    Returns ``(is_valid, issues, manifest)`` where ``manifest`` is ``None``
    if the manifest cannot be parsed.
    """
    issues: list[str] = []
    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        return False, [f"Manifest file not found: {manifest_file}"], None

    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest = _manifest_from_json(payload)
    except Exception as exc:
        return False, [f"Failed to read manifest: {exc}"], None

    payload_hash = _sha256_text(_canonical_json(_manifest_payload(manifest)))
    if payload_hash != manifest.manifest_payload_hash:
        issues.append("Manifest payload hash mismatch")

    if expected_signing_key_id is not None and expected_signing_key_id != manifest.signature_key_id:
        issues.append(
            "Manifest signing key id mismatch "
            f"(expected '{expected_signing_key_id}', got '{manifest.signature_key_id}')"
        )

    if manifest.signature_alg == "hmac-sha256":
        if not signing_key:
            issues.append("Manifest is signed but no signing key was provided")
        elif not manifest.signature:
            issues.append("Manifest signature is missing")
        else:
            expected_signature = _hmac_signature(manifest.manifest_payload_hash, signing_key)
            if not hmac.compare_digest(expected_signature, manifest.signature):
                issues.append("Manifest signature mismatch")
        if manifest.signature_key_id is None:
            issues.append("Manifest signature key id is missing")
    elif manifest.signature is not None and manifest.signature_alg is None:
        issues.append("Manifest signature present without signature algorithm")
    elif manifest.signature_key_id is not None and manifest.signature_alg is None:
        issues.append("Manifest signature key id present without signature algorithm")

    current = build_audit_manifest(audit_path, signing_key=None)
    if current.event_count != manifest.event_count:
        issues.append(
            f"Event count mismatch: expected {manifest.event_count}, observed {current.event_count}"
        )
    if current.first_event_id != manifest.first_event_id:
        issues.append("First event id mismatch")
    if current.last_event_id != manifest.last_event_id:
        issues.append("Last event id mismatch")
    if current.first_hash != manifest.first_hash:
        issues.append("First hash mismatch")
    if current.last_hash != manifest.last_hash:
        issues.append("Last hash mismatch")
    if current.chain_valid != manifest.chain_valid:
        issues.append("Chain validity mismatch")
    if current.event_type_counts != manifest.event_type_counts:
        issues.append("Event-type distribution mismatch")
    if current.audit_sha256 != manifest.audit_sha256:
        issues.append("Audit file checksum mismatch")

    return len(issues) == 0, issues, manifest


@runtime_checkable
class AuditLoggerProtocol(Protocol):
    """Protocol for append-only audit loggers."""

    def record(self, event_type: str, payload: dict[str, Any]) -> AuditEvent:
        ...

    def events(self) -> list[AuditEvent]:
        ...


class InMemoryAuditLogger:
    """Thread-safe, hash-chained in-memory audit logger.

    Each event hash includes the previous event hash, making the chain
    tamper-evident within a process.
    """

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._last_hash = "GENESIS"
        self._lock = threading.Lock()

    def record(self, event_type: str, payload: dict[str, Any]) -> AuditEvent:
        event = _build_event(prev_hash=self._last_hash, event_type=event_type, payload=payload)

        with self._lock:
            self._events.append(event)
            self._last_hash = event.event_hash

        return event

    def events(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._events)

    def verify_chain(self) -> bool:
        """Return True when all event hashes are internally consistent."""
        return _verify_event_chain(self.events())


class FileAuditLogger:
    """Thread-safe, hash-chained audit logger persisted as NDJSON.

    Existing events are loaded from disk on startup and can be verified with
    :meth:`verify_chain` to detect tampering.
    """

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[AuditEvent] = []
        self._last_hash = "GENESIS"
        self._lock = threading.Lock()
        self._load_existing_events()

    @property
    def file_path(self) -> Path:
        return self._file_path

    def _load_existing_events(self) -> None:
        if not self._file_path.exists():
            return

        loaded: list[AuditEvent] = []
        for line in self._file_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            event = _event_from_json(payload)
            loaded.append(event)

        self._events = loaded
        self._last_hash = loaded[-1].event_hash if loaded else "GENESIS"

    def record(self, event_type: str, payload: dict[str, Any]) -> AuditEvent:
        with self._lock:
            event = _build_event(prev_hash=self._last_hash, event_type=event_type, payload=payload)
            line = json.dumps(event.model_dump(), sort_keys=True, separators=(",", ":"))
            with self._file_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.write("\n")
            self._events.append(event)
            self._last_hash = event.event_hash
            return event

    def events(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._events)

    def verify_chain(self) -> bool:
        return _verify_event_chain(self.events())
