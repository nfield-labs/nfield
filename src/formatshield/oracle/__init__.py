"""FormatShield oracle package."""

from formatshield.oracle.backend_registry import BackendCapability, BackendRegistry, get_registry
from formatshield.oracle.context import RoutingContext, TelemetryRecord
from formatshield.oracle.oracle_x import OracleX
from formatshield.oracle.routing_decision import RoutingDecision

__all__ = [
    "BackendCapability",
    "BackendRegistry",
    "OracleX",
    "RoutingContext",
    "RoutingDecision",
    "TelemetryRecord",
    "get_registry",
]
