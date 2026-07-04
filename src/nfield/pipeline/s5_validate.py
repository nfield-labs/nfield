"""Stage 5: Validation.

Zero API calls. For every leaf, validates extracted values against type and
constraint rules. Filled values that fail are marked ``FAILED``; fields the model
left ``PENDING`` are marked ``FAILED`` too. When grounding is enabled, a filled value
the leaf's excerpt does not support is also marked ``FAILED`` (anti-hallucination), so
the recovery pass re-extracts it. All re-extraction is performed by the recovery pass
(Stage 5.5), so this stage only settles state.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nfield.validation._grounding import grounding_score, is_groundable
from nfield.validation._type_check import validate_field

if TYPE_CHECKING:
    from nfield.config import ExtractionConfig
    from nfield.pipeline._state import PipelineState
    from nfield.providers._protocol import LLMProvider
    from nfield.schema._types import CapacityLeaf

__all__ = ["run_stage_5"]

logger = logging.getLogger(__name__)


async def run_stage_5(
    state: PipelineState,
    provider: LLMProvider,
    config: ExtractionConfig,
) -> PipelineState:
    """Validate all extracted values (no API calls).

    For each leaf, type- and constraint-checks the filled values and settles each
    field's state. Re-extraction of failures is deferred to the recovery pass.

    Args:
        state: Pipeline state from Stage 4 (blackboard has extracted values).
        provider: LLM provider (unused here; kept for stage-signature uniformity).
        config: Extraction configuration (unused here; kept for uniformity).

    Returns:
        Updated ``PipelineState``.

    """
    assert state.blackboard is not None, "Blackboard must be initialised"

    doc_text = "\n".join(s.text for s in state.segments)
    for leaf in state.leaves:
        _validate_leaf(leaf, state, doc_text)

    if state.ground_values:
        _ground_all(state)

    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_leaf(leaf: CapacityLeaf, state: PipelineState, doc_text: str) -> None:
    """Validate a leaf's fields without any API call.

    Filled values are type- and constraint-checked; an invalid one is marked
    ``FAILED``. A field left ``PENDING`` (extracted but never returned) is also
    marked ``FAILED`` so the recovery pass treats it as missing. ``EMPTY``,
    ``FAILED``, ``CONFLICT`` and ``NEEDS_REVALIDATION`` fields are left untouched
    for the recovery pass to re-extract.

    Args:
        leaf: The leaf whose fields to validate.
        state: Pipeline state holding the blackboard.
        doc_text: The whole document text, for grounding array-quality checks.
    """
    from nfield.assembly._blackboard import FieldState

    bb = state.blackboard
    if bb is None:
        return
    from nfield.pipeline.s4_extract import _array_quality_error

    filled = bb.get_filled()
    for f in leaf.fields:
        field_state = bb.get_state(f.path)
        if field_state == FieldState.FILLED:
            value = filled.get(f.path)
            valid, err = validate_field(value, f)
            if not valid:
                bb.mark_failed(f.path, err or "validation failed")
                continue
            # A string that merely restates the field's own name ("Borrower" for
            # parties.borrower) is the document's placeholder term, not a value.
            if isinstance(value, str) and _restates_field_name(value, f.path):
                bb.mark_failed(
                    f.path,
                    f"the value {value!r} restates the field name, not the "
                    "document's actual value; extract the concrete value",
                )
                continue
            # An array can be type-valid yet hold document furniture (labels, list
            # ordinals) or reworded text; fail it with the reason so the recovery
            # pass re-extracts. Checked against the WHOLE document, not the leaf
            # excerpt - window-continued items live outside the excerpt. The value
            # is stashed so recovery can restore it if re-extraction yields worse.
            if isinstance(value, list) and isinstance(f.constraints.get("items"), dict):
                quality_err = _array_quality_error(value, doc_text or leaf.document_excerpt)
                if quality_err is not None:
                    state.quality_failed_values[f.path] = value
                    bb.mark_failed(f.path, quality_err)
        elif field_state == FieldState.PENDING:
            bb.mark_failed(f.path, "field not extracted")


def _restates_field_name(value: str, path: str) -> bool:
    """True when *value* is just the field's own key rendered as words.

    ``"Borrower"`` for ``parties.borrower`` or ``"Use of Proceeds"`` for
    ``terms.use_of_proceeds`` answer the question with its own words; articles
    are ignored so ``"The Borrower"`` matches too. A value containing anything
    beyond the key (a real name contains more) never matches. Article stripping
    is English-only, a heuristic sharpener rather than the core key match.
    """
    key = path.rsplit(".", 1)[-1].replace("_", " ").casefold()
    words = [w for w in value.replace("_", " ").casefold().split() if w not in ("the", "a", "an")]
    return " ".join(words) == key


def _ground_all(state: PipelineState) -> None:
    """Score each filled, groundable value against the excerpt it was extracted from.

    Runs after type/constraint validation has settled every leaf. For each filled
    value of a groundable type, the support score is taken as the **maximum** over the
    excerpts of all leaves that contain the field (a field split across leaves is
    grounded if any of its excerpts supports it). The score is recorded on
    ``state.grounding_scores`` for the Stage 6 hallucination metric; a value scoring
    below ``state.grounding_min_score`` is marked ``FAILED`` so the recovery pass
    re-extracts it (``EMPTY``/``FAILED``/``None`` and non-groundable types are skipped).

    Args:
        state: Pipeline state (blackboard, leaves, grounding threshold/scores).
    """
    from nfield.assembly._blackboard import FieldState

    bb = state.blackboard
    if bb is None:
        return
    filled = bb.get_filled()

    best: dict[str, float] = {}
    for leaf in state.leaves:
        excerpt = leaf.document_excerpt
        for f in leaf.fields:
            value = filled.get(f.path)
            if value is None or not is_groundable(f, value):
                continue
            score = grounding_score(value, excerpt, f.type)
            if score > best.get(f.path, -1.0):
                best[f.path] = score

    for path, score in best.items():
        state.grounding_scores[path] = score
        if score < state.grounding_min_score and bb.get_state(path) == FieldState.FILLED:
            bb.mark_failed(
                path,
                f"ungrounded: value not supported by the document (score {score:.2f})",
            )
