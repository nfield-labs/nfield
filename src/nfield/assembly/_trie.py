"""Radix Trie Assembler — flat key=value pairs to nested JSON.

Implements the inverse of the SFEP Bijection (Theorem 1): given a flat dict
of dot-notation paths and typed Python values, reconstructs the original
nested JSON structure.

Algorithm
---------
1. :func:`parse_path_segments` splits ``"a.b[0].c"`` into ``["a", "b", 0, "c"]``.
2. :class:`RadixTrie` inserts each path into a tree structure via depth-bounded
   recursive descent.
3. :meth:`RadixTrie.build` returns the assembled nested dict/list structure.

Resource bounds (paths come from LLM output, which the source document can
influence, so they are untrusted): :func:`parse_path_segments` rejects a path
deeper than ``_MAX_PATH_DEPTH`` segments or carrying an array index above
``_MAX_ARRAY_INDEX``. Without these, a single crafted line like
``items[999999999] = x`` would grow a list to a billion elements (memory
exhaustion) and a path like ``a.a.a…`` (thousands deep) would overflow the
recursion stack. Both now raise :class:`AssemblyError` instead.

Invariant (Theorem 1 — SFEP Bijection)
---------------------------------------
``assemble_json(parse_sfep(sfep_text)) == original_json``

provided the schema is losslessly representable in SFEP format (i.e. no
duplicate field paths and no unbounded dynamic keys).

Handles
-------
* Nested objects: ``a.b.c``
* Integer-indexed arrays: ``items[0].name``, ``items[1].name``
* Homogeneous arrays (flattener ``[]`` suffix): ``segments[].capex`` → index 0
* Mixed nesting: ``a.b[0].c.d[1]``
* Single top-level fields: ``name``
* Empty input: returns ``{}``
"""

from __future__ import annotations

import re
from typing import Any

from nfield.exceptions import AssemblyError

__all__ = [
    "RadixTrie",
    "assemble_json",
    "parse_path_segments",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches a key optionally followed by one or more bracket groups. Brackets may
# be indexed ("items[0]") or empty ("segments[]"). The schema flattener emits
# empty brackets for a homogeneous array (an array whose elements share one
# shape, e.g. "items[].name"); these map to index 0 because the SFEP wire format
# carries no per-element index for such arrays.
_RE_KEY_WITH_INDICES: re.Pattern[str] = re.compile(r"^([^\[]+)((?:\[\d*\])*)$")

# Matches individual bracket groups: "[0]", "[12]", or empty "[]"
_RE_BRACKET_INDEX: re.Pattern[str] = re.compile(r"\[(\d*)\]")

# Resource bounds on untrusted (LLM-produced) paths. An array index drives list
# growth (``array.append`` per slot) and path depth drives recursion, so both must
# be capped or a crafted path can exhaust memory or the stack. The limits are far
# above any real schema — they only fire on adversarial input, where they raise
# AssemblyError instead of crashing the process.
#
# _MAX_ARRAY_INDEX: a list grown to index N costs ~N*8 bytes of pointers, so the cap
# bounds one array's worst case to ~800 KB (100k * 8B). Real arrays hold tens to a
# few thousand elements; homogeneous arrays collapse to index 0.
#
# _MAX_PATH_DEPTH: insertion recurses once per segment, and CPython's recursion limit
# is ~1000 frames (already partly used by the pipeline/asyncio stack), so the cap is
# kept well below it. Real JSON nests ~5-15 deep. (A deeper need would call for an
# iterative rewrite, not a higher cap.)
_MAX_ARRAY_INDEX: int = 100_000
_MAX_PATH_DEPTH: int = 256


# ---------------------------------------------------------------------------
# Path segment parser
# ---------------------------------------------------------------------------


def parse_path_segments(path: str) -> list[str | int]:
    """Split a dot-notation path into a list of string keys and integer indices.

    Parses ``"a.b[0].c[1][2]"`` into ``["a", "b", 0, "c", 1, 2]``.

    Args:
        path: Dot-notation path string from SFEP output.

    Returns:
        Ordered list of path segments. String elements are object keys;
        integer elements are array indices.

    Raises:
        AssemblyError: If the path is empty, malformed, deeper than
            ``_MAX_PATH_DEPTH`` segments, or carries an array index above
            ``_MAX_ARRAY_INDEX`` (the resource bounds on untrusted input).

    Example:
        >>> parse_path_segments("address.city")
        ['address', 'city']
        >>> parse_path_segments("items[0].name")
        ['items', 0, 'name']
        >>> parse_path_segments("matrix[0][1]")
        ['matrix', 0, 1]
        >>> parse_path_segments("segments[].capex")
        ['segments', 0, 'capex']
    """
    if not path or not path.strip():
        raise AssemblyError("Cannot parse empty path", path=path)

    segments: list[str | int] = []

    for dot_part in path.split("."):
        if not dot_part:
            raise AssemblyError(
                f"Path contains empty segment: {path!r}",
                path=path,
            )

        match = _RE_KEY_WITH_INDICES.match(dot_part)
        if match is None:
            raise AssemblyError(
                f"Path segment {dot_part!r} is malformed in path {path!r}",
                path=path,
            )

        key = match.group(1)
        index_str = match.group(2)

        if not key:
            raise AssemblyError(
                f"Empty key before index in segment {dot_part!r}",
                path=path,
            )

        segments.append(key)

        # Append each bracket as an integer index. Empty brackets ("[]",
        # homogeneous array) map to index 0 — a single representative element.
        for idx_match in _RE_BRACKET_INDEX.finditer(index_str):
            raw = idx_match.group(1)
            idx = int(raw) if raw else 0
            if idx > _MAX_ARRAY_INDEX:
                raise AssemblyError(
                    f"Array index {idx} exceeds the maximum {_MAX_ARRAY_INDEX} in path {path!r}",
                    path=path,
                )
            segments.append(idx)

    # Bound recursion depth: insertion recurses once per segment, so a very deep
    # path would overflow the stack. Reject before any allocation happens.
    if len(segments) > _MAX_PATH_DEPTH:
        raise AssemblyError(
            f"Path depth {len(segments)} exceeds the maximum {_MAX_PATH_DEPTH} in path {path!r}",
            path=path,
        )

    return segments


# ---------------------------------------------------------------------------
# RadixTrie
# ---------------------------------------------------------------------------

# Internal node type aliases
_ObjectNode = dict[str, Any]
_ArrayNode = list[Any]
_TrieNode = _ObjectNode | _ArrayNode


class RadixTrie:
    """Radix trie for assembling dot-notation paths into a nested dict/list.

    Usage:
        >>> trie = RadixTrie()
        >>> trie.insert("a.b", 1)
        >>> trie.insert("a.c", 2)
        >>> trie.build()
        {'a': {'b': 1, 'c': 2}}

    Attributes:
        _root: Root dict node of the trie.
    """

    def __init__(self) -> None:
        self._root: _ObjectNode = {}

    def insert(self, path: str, value: Any) -> None:
        """Insert a path=value pair into the trie.

        Args:
            path: Dot-notation path string.
            value: Typed Python value to store at this path.

        Raises:
            AssemblyError: If the path is malformed or conflicts with an
                existing value (e.g. trying to nest under a leaf value).

        Example:
            >>> trie = RadixTrie()
            >>> trie.insert("a.b.c", 42)
            >>> trie.build()
            {'a': {'b': {'c': 42}}}
        """
        segments = parse_path_segments(path)
        self._insert_segments(self._root, segments, value, path)

    def build(self) -> dict[str, Any]:
        """Traverse the trie and produce the nested JSON structure.

        Returns:
            Nested dict matching the original JSON Schema structure.

        Example:
            >>> trie = RadixTrie()
            >>> trie.insert("x", 1)
            >>> trie.build()
            {'x': 1}
        """
        return dict(self._root)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _insert_segments(
        self,
        node: _ObjectNode,
        segments: list[str | int],
        value: Any,
        original_path: str,
    ) -> None:
        """Recursively insert segments starting at *node*.

        Args:
            node: Current dict node in the trie.
            segments: Remaining path segments to process.
            value: Value to store at the leaf.
            original_path: Full original path for error messages.
        """
        if not segments:
            return

        current_seg = segments[0]
        remaining = segments[1:]

        if not isinstance(current_seg, str):
            raise AssemblyError(
                f"Expected string key at root level, got index {current_seg!r} "
                f"in path {original_path!r}",
                path=original_path,
            )

        if not remaining:
            # Leaf node — store the value
            node[current_seg] = value
            return

        next_seg = remaining[0]

        if isinstance(next_seg, int):
            # Next segment is an array index → current key maps to a list
            node.setdefault(current_seg, [])
            existing = node[current_seg]
            if not isinstance(existing, list):
                raise AssemblyError(
                    f"Path conflict at {current_seg!r} in {original_path!r}: "
                    f"expected list, found {type(existing).__name__}",
                    path=original_path,
                )
            self._insert_into_array(existing, remaining, value, original_path)
        else:
            # Next segment is a string key → current key maps to a dict
            node.setdefault(current_seg, {})
            existing = node[current_seg]
            if not isinstance(existing, dict):
                raise AssemblyError(
                    f"Path conflict at {current_seg!r} in {original_path!r}: "
                    f"expected dict, found {type(existing).__name__}",
                    path=original_path,
                )
            self._insert_segments(existing, remaining, value, original_path)

    def _insert_into_array(
        self,
        array: _ArrayNode,
        segments: list[str | int],
        value: Any,
        original_path: str,
    ) -> None:
        """Insert into an array node starting with an index segment.

        Args:
            array: Current list node.
            segments: Remaining segments, first of which must be an int.
            value: Value to store.
            original_path: Full path for error context.
        """
        if not segments or not isinstance(segments[0], int):
            raise AssemblyError(
                f"Expected array index in {original_path!r}",
                path=original_path,
            )

        idx = segments[0]
        remaining = segments[1:]

        # Grow array to accommodate this index
        while len(array) <= idx:
            array.append(None)

        if not remaining:
            # Leaf at this index
            array[idx] = value
            return

        next_seg = remaining[0]

        if isinstance(next_seg, int):
            # Nested array
            if array[idx] is None:
                array[idx] = []
            if not isinstance(array[idx], list):
                raise AssemblyError(
                    f"Path conflict at index {idx} in {original_path!r}: expected list",
                    path=original_path,
                )
            self._insert_into_array(array[idx], remaining, value, original_path)
        else:
            # Object inside array
            if array[idx] is None:
                array[idx] = {}
            if not isinstance(array[idx], dict):
                raise AssemblyError(
                    f"Path conflict at index {idx} in {original_path!r}: expected dict",
                    path=original_path,
                )
            self._insert_segments(array[idx], remaining, value, original_path)


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


def assemble_json(pairs: dict[str, Any]) -> dict[str, Any]:
    """Assemble a flat SFEP result dict into a nested JSON structure.

    This is the public entry point for the assembly stage. Inserts all
    ``path=value`` pairs into a :class:`RadixTrie` and returns the built
    nested dict.

    Args:
        pairs: Flat dict mapping dot-notation paths to typed Python values.
            Produced by :func:`~nfield.extraction._sfep.parse_sfep`.

    Returns:
        Nested dict matching the original JSON Schema structure.

    Raises:
        AssemblyError: If any path is malformed or conflicts with another path.

    Example:
        >>> assemble_json({"a.b": 1, "a.c": 2})
        {'a': {'b': 1, 'c': 2}}
        >>> assemble_json({"items[0].name": "x", "items[1].name": "y"})
        {'items': [{'name': 'x'}, {'name': 'y'}]}
        >>> assemble_json({})
        {}
    """
    if not pairs:
        return {}

    trie = RadixTrie()
    for path, value in pairs.items():
        trie.insert(path, value)

    return trie.build()
