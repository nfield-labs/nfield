"""Blackboard state machine for per-field extraction state tracking.

The Blackboard is the shared data structure that accumulates extraction
results across all capacity leaves and retry rounds. It tracks the state
of each field throughout the pipeline using a finite state machine with
6 states per field.

State transitions
-----------------

    EMPTY ──write()──► PENDING ──write()──► FILLED
                                         │
                      mark_failed() ─────┼───► FAILED
                      mark_needs_revalidation() ► NEEDS_REVALIDATION
                      write() (conflict) ─────► CONFLICT

Notes
-----
* ``write_raw()`` is the dep-change-safe variant: it does NOT transition
  a FILLED field back to PENDING. Used when updating a dependency whose
  change may invalidate a dependent field.
* Cross-leaf conflict detection: if two leaves extract different non-None
  values for the same field, the state transitions to ``CONFLICT`` and
  both values are stored for reporting.
* Once a field is ``FAILED`` or ``CONFLICT``, it can only transition to
  ``NEEDS_REVALIDATION`` (for human review), not back to ``FILLED``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from nfield.exceptions import AssemblyError

__all__ = [
    "Blackboard",
    "FieldState",
]


# ---------------------------------------------------------------------------
# FieldState enum
# ---------------------------------------------------------------------------


class FieldState(Enum):
    """State of a single field in the extraction blackboard.

    Attributes:
        EMPTY: Field has not been seen in any extraction output yet.
        PENDING: Field has been written at least once but not confirmed.
        FILLED: Field has a validated value (at least type-valid).
        FAILED: Field extraction failed and retry did not recover it.
        CONFLICT: Two or more leaves extracted different values for this field.
        NEEDS_REVALIDATION: Field is flagged for human or semantic review.

    Example:
        >>> FieldState.FILLED.value
        'filled'
    """

    EMPTY = "empty"
    PENDING = "pending"
    FILLED = "filled"
    FAILED = "failed"
    CONFLICT = "conflict"
    NEEDS_REVALIDATION = "needs_revalidation"


# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------
#
# State transitions are enforced inline in each write/mark_* method below, not via
# a transition table: the real rules are value-dependent (a same-value re-write is a
# no-op, a different value escalates to CONFLICT, a transient flag tags a FAILED),
# which a flat state->states table cannot express. The legal moves, for reference:
#   EMPTY    -> PENDING | FILLED | FAILED
#   PENDING  -> FILLED | FAILED | CONFLICT
#   FILLED   -> CONFLICT | NEEDS_REVALIDATION   (+ reopen_for_retry -> PENDING)
#   FAILED   -> FILLED | NEEDS_REVALIDATION     (+ reopen_for_retry -> PENDING)
#   CONFLICT -> NEEDS_REVALIDATION              (+ reopen_for_retry -> PENDING)
#   NEEDS_REVALIDATION -> (terminal)            (+ reopen_for_retry -> PENDING)


class Blackboard:
    """Per-field state machine tracking extraction results across all leaves.

    The Blackboard is initialized with the complete list of field paths
    from Stage 1 and accumulates values written by Stage 4 (extraction)
    and Stage 5 (validation + retry).

    Attributes:
        _states: Mapping of field path to current FieldState.
        _values: Mapping of field path to current typed value.
        _errors: Mapping of failed field paths to error messages.
        _conflict_values: Mapping of conflicted field paths to all seen values.

    Example:
        >>> bb = Blackboard(["name", "age"])
        >>> bb.write("name", "Alice")
        >>> bb.get_filled()
        {'name': 'Alice'}
        >>> bb.get_missing()
        ['age']
    """

    def __init__(self, paths: list[str]) -> None:
        """Initialise a Blackboard for the given field paths.

        Args:
            paths: All field paths from the flattened schema (Stage 1 output).
                All paths start in ``EMPTY`` state.

        Raises:
            ValueError: If *paths* contains duplicates.
        """
        if len(paths) != len(set(paths)):
            duplicates = [p for p in paths if paths.count(p) > 1]
            raise ValueError(f"Blackboard paths must be unique; duplicates: {duplicates}")
        self._states: dict[str, FieldState] = dict.fromkeys(paths, FieldState.EMPTY)
        self._values: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._conflict_values: dict[str, list[Any]] = {}
        # Paths whose FAILED state is a transient API/call failure (the call never
        # returned), tracked apart from a genuine "absent in document" failure.
        self._call_failed: set[str] = set()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def write(self, path: str, value: Any) -> None:
        """Write a value for a field, transitioning its state.

        Transitions:
        * ``EMPTY`` / ``PENDING``    → ``FILLED`` (or ``CONFLICT`` if value differs)
        * ``FILLED`` with same value → no-op
        * ``FILLED`` with new value  → ``CONFLICT``
        * ``FAILED``                 → ``FILLED`` (retry recovered this field)

        Args:
            path: Dot-notation field path.
            value: Typed Python value from the SFEP parser.

        Raises:
            AssemblyError: If the path is not registered in this blackboard.
        """
        self._require_path(path)
        state = self._states[path]

        if state == FieldState.FILLED:
            existing = self._values.get(path)
            if existing == value:
                return  # Same value from a second leaf - no conflict
            # Different value from a second leaf - conflict
            self._conflict_values.setdefault(path, [existing])
            if value not in self._conflict_values[path]:
                self._conflict_values[path].append(value)
            self._states[path] = FieldState.CONFLICT
            return

        if state == FieldState.CONFLICT:
            # Already conflicted - accumulate additional values
            if value not in self._conflict_values.get(path, []):
                self._conflict_values.setdefault(path, []).append(value)
            return

        if state == FieldState.NEEDS_REVALIDATION:
            # Terminal state - no further writes
            return

        # EMPTY / PENDING / FAILED → FILLED
        self._values[path] = value
        self._states[path] = FieldState.FILLED
        self._call_failed.discard(path)

    def write_raw(self, path: str, value: Any) -> None:
        """Dependency-change-safe write: does not overwrite a FILLED field.

        Used when updating dependency values that may propagate to dependent
        fields. Prevents overwriting a valid extracted value with a stale
        dependency update.

        If the field is ``EMPTY`` or ``PENDING``, behaves like :meth:`write`.
        If the field is already ``FILLED``, the write is silently discarded
        and the field is flagged ``NEEDS_REVALIDATION`` (since its dependency
        changed, its value may be stale).

        Args:
            path: Dot-notation field path.
            value: New typed Python value.

        Raises:
            AssemblyError: If the path is not registered in this blackboard.
        """
        self._require_path(path)
        state = self._states[path]

        if state == FieldState.FILLED:
            # Dependency changed while this field already has a value -
            # flag for revalidation without overwriting
            self._states[path] = FieldState.NEEDS_REVALIDATION
            return

        if state in (FieldState.FAILED, FieldState.CONFLICT, FieldState.NEEDS_REVALIDATION):
            return  # Cannot update terminal/conflict states

        # EMPTY / PENDING → write normally
        self._values[path] = value
        self._states[path] = FieldState.FILLED

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def mark_failed(self, path: str, error: str, *, transient: bool = False) -> None:
        """Transition a field to ``FAILED`` state with an error message.

        Args:
            path: Dot-notation field path.
            error: Human-readable description of the failure.
            transient: ``True`` when the failure is a call/API error (the request
                never returned) rather than the field being absent from the
                document. Tracked separately so reporting and recovery can tell a
                network blip from genuinely missing data.

        Raises:
            AssemblyError: If the path is not registered.
        """
        self._require_path(path)
        state = self._states[path]
        if state not in (
            FieldState.EMPTY,
            FieldState.PENDING,
            FieldState.FILLED,
            FieldState.FAILED,
        ):
            return  # Cannot transition from CONFLICT or NEEDS_REVALIDATION to FAILED
        self._states[path] = FieldState.FAILED
        self._errors[path] = error
        if transient:
            self._call_failed.add(path)
        else:
            self._call_failed.discard(path)

    def mark_needs_revalidation(self, path: str) -> None:
        """Transition a field to ``NEEDS_REVALIDATION`` state.

        Args:
            path: Dot-notation field path.

        Raises:
            AssemblyError: If the path is not registered.
        """
        self._require_path(path)
        self._states[path] = FieldState.NEEDS_REVALIDATION

    def mark_pending(self, path: str) -> None:
        """Transition a field from ``EMPTY`` to ``PENDING`` state.

        Used to indicate that extraction for this field is in-flight.

        Args:
            path: Dot-notation field path.

        Raises:
            AssemblyError: If the path is not registered.
        """
        self._require_path(path)
        if self._states[path] == FieldState.EMPTY:
            self._states[path] = FieldState.PENDING

    def reopen_for_retry(self, path: str) -> bool:
        """Reopen a FAILED / CONFLICT / NEEDS_REVALIDATION field for re-extraction.

        A controlled escape hatch for the retry orchestrator: it moves a field
        that the normal FSM treats as settled (or terminal) back to ``PENDING`` so
        a subsequent :meth:`write` can record a fresh value. Clears the field's
        prior error and any stored conflicting values, since the retry supersedes
        them. Fields in ``EMPTY``/``PENDING``/``FILLED`` are left unchanged.

        Args:
            path: Dot-notation field path.

        Returns:
            ``True`` if the field was reopened, ``False`` if its state was not
            eligible (so the caller knows whether a retry will be applied).

        Raises:
            AssemblyError: If the path is not registered.
        """
        self._require_path(path)
        if self._states[path] in (
            FieldState.FAILED,
            FieldState.CONFLICT,
            FieldState.NEEDS_REVALIDATION,
        ):
            self._states[path] = FieldState.PENDING
            self._errors.pop(path, None)
            self._conflict_values.pop(path, None)
            return True
        return False

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_missing(self) -> list[str]:
        """Return paths of fields still in ``EMPTY`` state after extraction.

        Returns:
            Sorted list of dot-notation paths that were never extracted.

        Example:
            >>> bb = Blackboard(["a", "b"])
            >>> bb.write("a", 1)
            >>> bb.get_missing()
            ['b']
        """
        return sorted(p for p, s in self._states.items() if s == FieldState.EMPTY)

    def get_conflicts(self) -> list[str]:
        """Return paths of fields in ``CONFLICT`` state.

        Returns:
            Sorted list of dot-notation paths with conflicting values.

        Example:
            >>> bb = Blackboard(["x"])
            >>> bb.write("x", 1)
            >>> bb.write("x", 2)
            >>> bb.get_conflicts()
            ['x']
        """
        return sorted(p for p, s in self._states.items() if s == FieldState.CONFLICT)

    def get_needs_revalidation(self) -> list[str]:
        """Return paths of fields flagged for revalidation.

        Returns:
            Sorted list of dot-notation paths in NEEDS_REVALIDATION state.
        """
        return sorted(p for p, s in self._states.items() if s == FieldState.NEEDS_REVALIDATION)

    def get_failed(self) -> list[str]:
        """Return paths of fields in ``FAILED`` state.

        Returns:
            Sorted list of dot-notation paths that failed extraction.
        """
        return sorted(p for p, s in self._states.items() if s == FieldState.FAILED)

    def get_call_failed(self) -> list[str]:
        """Return paths whose ``FAILED`` state is a transient call/API failure.

        These are fields the model never got a chance to answer (the request
        failed), as opposed to fields it answered ``NULL`` (absent from the
        document). Used to report API failures distinctly from missing data.

        Returns:
            Sorted list of dot-notation paths still FAILED due to a call error.
        """
        return sorted(p for p in self._call_failed if self._states.get(p) == FieldState.FAILED)

    def get_filled(self) -> dict[str, Any]:
        """Return fields that hold a real (non-empty) extracted value.

        ``None`` and ``""`` are excluded on purpose: the recovery pass marks
        tree-backtracked "confirmed absent" fields ``FILLED`` with ``None``
        (:meth:`write_raw`), and an empty string carries no extracted value. Both
        would overstate the extraction rate, so they are omitted here and counted
        as missing by the quality metrics - the same empties the scorer discounts.

        Returns:
            Dict of ``{path: value}`` for ``FILLED`` fields whose value is not
            ``None`` or ``""``.

        Example:
            >>> bb = Blackboard(["name", "nickname"])
            >>> bb.write("name", "Alice")
            >>> bb.write_raw("nickname", None)  # confirmed absent
            >>> bb.get_filled()
            {'name': 'Alice'}
        """
        return {
            p: self._values[p]
            for p, s in self._states.items()
            if s == FieldState.FILLED and self._values.get(p) not in (None, "")
        }

    def get_value(self, path: str) -> Any:
        """Return the last value written for a field, regardless of its state.

        Unlike :meth:`get_filled`, this returns the stored value even for a ``FAILED``
        or ``NEEDS_REVALIDATION`` field, so a retry can show the model the exact value
        its previous attempt produced. Returns ``None`` if nothing was ever written.

        Args:
            path: Dot-notation field path.

        Returns:
            The stored value, or ``None`` if the field has no recorded value.
        """
        return self._values.get(path)

    def get_conflict_values(self, path: str) -> list[Any]:
        """Return all conflicting values seen for a field.

        Args:
            path: Dot-notation field path.

        Returns:
            List of all values written to a CONFLICT field.
        """
        return list(self._conflict_values.get(path, []))

    def get_state(self, path: str) -> FieldState:
        """Return the current state of a field.

        Args:
            path: Dot-notation field path.

        Returns:
            Current :class:`FieldState` for the path.

        Raises:
            AssemblyError: If the path is not registered.
        """
        self._require_path(path)
        return self._states[path]

    def get_error(self, path: str) -> str | None:
        """Return the error message for a failed field.

        Args:
            path: Dot-notation field path.

        Returns:
            Error message string, or ``None`` if the field did not fail.
        """
        return self._errors.get(path)

    def all_paths(self) -> list[str]:
        """Return all registered field paths in sorted order.

        Returns:
            Sorted list of all paths registered at construction.
        """
        return sorted(self._states)

    def summary(self) -> dict[str, int]:
        """Return a count of fields in each state.

        Returns:
            Dict mapping state name to field count.

        Example:
            >>> bb = Blackboard(["a", "b", "c"])
            >>> bb.write("a", 1)
            >>> bb.mark_failed("b", "parse error")
            >>> bb.summary()
            {'empty': 1, 'pending': 0, 'filled': 1, 'failed': 1, 'conflict': 0, 'needs_revalidation': 0}
        """
        counts: dict[str, int] = {s.value: 0 for s in FieldState}
        for state in self._states.values():
            counts[state.value] += 1
        return counts

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _require_path(self, path: str) -> None:
        """Assert that *path* is registered in this blackboard.

        Args:
            path: Path to check.

        Raises:
            AssemblyError: If the path was not registered at construction.
        """
        if path not in self._states:
            raise AssemblyError(
                f"Unknown field path {path!r} - "
                "path must be registered at Blackboard construction",
                path=path,
            )
