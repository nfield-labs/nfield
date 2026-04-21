"""FormatShield oracle package."""

from formatshield.oracle.backend_registry import BackendCapability, BackendRegistry, get_registry
from formatshield.oracle.context import RoutingContext, TelemetryRecord
from formatshield.oracle.oracle_x import OracleX
from formatshield.oracle.phi_calibrator import (
    OutcomeRecord,
    PhiComponents,
    PhiOracleCalibrator,
    build_phi_calibrator,
)
from formatshield.oracle.routing_decision import RoutingDecision

__all__ = [
    "BackendCapability",
    "BackendRegistry",
    "OracleX",
    "OutcomeRecord",
    "PhiComponents",
    "PhiOracleCalibrator",
    "RoutingContext",
    "RoutingDecision",
    "TelemetryRecord",
    "build_phi_calibrator",
    "get_registry",
]
