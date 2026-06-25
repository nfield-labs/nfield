"""NField extraction pipeline — stages S0 through S6."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "PipelineState",
    "run_stage_0",
    "run_stage_1",
    "run_stage_2a",
    "run_stage_2b",
    "run_stage_2c",
    "run_stage_3",
    "run_stage_4",
    "run_stage_5",
    "run_stage_6",
]


def __getattr__(name: str) -> object:
    if name == "PipelineState":
        from nfield.pipeline._state import PipelineState

        return PipelineState
    if name == "run_stage_0":
        from nfield.pipeline.s0_resources import run_stage_0

        return run_stage_0
    if name == "run_stage_1":
        from nfield.pipeline.s1_schema import run_stage_1

        return run_stage_1
    if name == "run_stage_2a":
        from nfield.pipeline.s2a_structure import run_stage_2a

        return run_stage_2a
    if name == "run_stage_2b":
        from nfield.pipeline.s2b_prepass import run_stage_2b

        return run_stage_2b
    if name == "run_stage_2c":
        from nfield.pipeline.s2c_packing import run_stage_2c

        return run_stage_2c
    if name == "run_stage_3":
        from nfield.pipeline.s3_excerpt import run_stage_3

        return run_stage_3
    if name == "run_stage_4":
        from nfield.pipeline.s4_extract import run_stage_4

        return run_stage_4
    if name == "run_stage_5":
        from nfield.pipeline.s5_validate import run_stage_5

        return run_stage_5
    if name == "run_stage_6":
        from nfield.pipeline.s6_assemble import run_stage_6

        return run_stage_6
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
