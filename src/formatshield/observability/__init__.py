"""FormatShield observability package."""

from formatshield.observability.audit_log import (
    AuditEvent,
    AuditLoggerProtocol,
    AuditManifest,
    FileAuditLogger,
    InMemoryAuditLogger,
    build_audit_manifest,
    verify_audit_manifest,
    write_audit_manifest,
)
from formatshield.observability.logger import StructuredLogger
from formatshield.observability.metrics import MetricsCollector, PrometheusMetrics
from formatshield.observability.otel import FormatShieldTracer, get_tracer

__all__ = [
    "AuditEvent",
    "AuditLoggerProtocol",
    "AuditManifest",
    "FileAuditLogger",
    "FormatShieldTracer",
    "InMemoryAuditLogger",
    "MetricsCollector",
    "PrometheusMetrics",
    "StructuredLogger",
    "build_audit_manifest",
    "get_tracer",
    "verify_audit_manifest",
    "write_audit_manifest",
]
