"""
FormatShield JSON schema analyzer.

Provides :class:`SchemaAnalyzer`, which inspects a JSON Schema dict to extract
two numeric features used by :class:`~formatshield.scorer.ComplexityScorer`:

* **max nesting depth** – how deeply nested the schema is (a flat object has
  depth 1; each additional level of ``properties``/``items``/``anyOf`` etc.
  adds one more).
* **constraint count** – the total number of constrained fields across the
  entire schema, defined as required-array members + enum-typed fields +
  pattern-constrained fields.

The analyzer handles the following JSON Schema keywords recursively:
``properties``, ``items``, ``anyOf``, ``oneOf``, ``allOf``, ``$defs``,
``definitions``, ``if`` / ``then`` / ``else``, and ``additionalProperties``
(when it is itself a schema object).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SchemaAnalyzer:
    """Analyse a JSON Schema dict and return depth and constraint-count metrics.

    Instances are stateless; the same :class:`SchemaAnalyzer` can be reused
    for multiple schema dicts without any side effects.

    Example::

        analyzer = SchemaAnalyzer()
        depth, constraints = analyzer.analyze(schema)
    """

    # JSON Schema keywords whose *values* are themselves sub-schemas or
    # lists of sub-schemas and should be recursed into.
    _ARRAY_KEYWORDS: frozenset[str] = frozenset({"anyOf", "oneOf", "allOf"})

    _SINGLE_KEYWORDS: frozenset[str] = frozenset(
        {
            "if",
            "then",
            "else",
            "not",
            "additionalProperties",
            "contains",
            "propertyNames",
            "unevaluatedItems",
            "unevaluatedProperties",
        }
    )

    # Keywords that hold named sub-schemas (mapping str → schema dict).
    _MAP_KEYWORDS: frozenset[str] = frozenset({"$defs", "definitions", "properties"})

    def analyze(self, schema: dict) -> tuple[int, int]:  # type: ignore[type-arg]
        """Return ``(max_depth, constraint_count)`` for *schema*.

        Parameters
        ----------
        schema:
            A JSON Schema represented as a Python dict.  Must not be ``None``,
            but may be empty (returns ``(0, 0)``).

        Returns
        -------
        tuple[int, int]
            ``(max_depth, constraint_count)`` where both values are ≥ 0.
            Returns ``(0, 0)`` on any error (malformed schema, recursion
            limit exceeded, etc.).
        """
        if not isinstance(schema, dict):
            return 0, 0
        try:
            depth = self._compute_depth(schema, current_depth=0)
            constraints = self._count_constraints(schema)
            return depth, constraints
        except Exception:
            logger.debug("SchemaAnalyzer.analyze: unexpected error", exc_info=True)
            return 0, 0

    # ------------------------------------------------------------------
    # Depth computation
    # ------------------------------------------------------------------

    def _compute_depth(self, node: Any, current_depth: int) -> int:
        """Recursively compute the maximum nesting depth starting from *node*.

        ``current_depth`` is the depth of *node* itself (the root starts at 0).

        The depth increases by 1 whenever we descend into a named property
        (via ``properties``), an array item schema (via ``items`` /
        ``prefixItems``), or a combinator branch (``anyOf`` / ``oneOf`` /
        ``allOf``).  Named-definition registries (``$defs`` / ``definitions``)
        are traversed at the *same* depth level since they are merely reusable
        definitions, not structural nesting.
        """
        if not isinstance(node, dict):
            return current_depth

        max_depth = current_depth

        # --- properties: each child property adds one depth level -----------
        properties = node.get("properties")
        if isinstance(properties, dict):
            for child_schema in properties.values():
                child_depth = self._compute_depth(child_schema, current_depth + 1)
                if child_depth > max_depth:
                    max_depth = child_depth

        # --- items (JSON Schema draft-4 through draft-7: single schema) -----
        items = node.get("items")
        if isinstance(items, dict):
            child_depth = self._compute_depth(items, current_depth + 1)
            if child_depth > max_depth:
                max_depth = child_depth
        elif isinstance(items, list):
            # prefixItems-style (tuple validation) draft ≤ 2019-09
            for item_schema in items:
                child_depth = self._compute_depth(item_schema, current_depth + 1)
                if child_depth > max_depth:
                    max_depth = child_depth

        # --- prefixItems (draft 2020-12) ------------------------------------
        prefix_items = node.get("prefixItems")
        if isinstance(prefix_items, list):
            for item_schema in prefix_items:
                child_depth = self._compute_depth(item_schema, current_depth + 1)
                if child_depth > max_depth:
                    max_depth = child_depth

        # --- combinators: anyOf / oneOf / allOf (each branch +1 depth) -----
        for keyword in ("anyOf", "oneOf", "allOf"):
            branches = node.get(keyword)
            if isinstance(branches, list):
                for branch in branches:
                    child_depth = self._compute_depth(branch, current_depth + 1)
                    if child_depth > max_depth:
                        max_depth = child_depth

        # --- definition registries: traverse at the *same* depth level ------
        for registry_key in ("$defs", "definitions"):
            registry = node.get(registry_key)
            if isinstance(registry, dict):
                for def_schema in registry.values():
                    child_depth = self._compute_depth(def_schema, current_depth)
                    if child_depth > max_depth:
                        max_depth = child_depth

        # --- single-schema keywords (same depth level) ----------------------
        for keyword in (
            "if",
            "then",
            "else",
            "not",
            "contains",
            "propertyNames",
            "unevaluatedItems",
            "unevaluatedProperties",
        ):
            sub = node.get(keyword)
            if isinstance(sub, dict):
                child_depth = self._compute_depth(sub, current_depth)
                if child_depth > max_depth:
                    max_depth = child_depth

        # --- additionalProperties as schema ---------------------------------
        additional = node.get("additionalProperties")
        if isinstance(additional, dict):
            child_depth = self._compute_depth(additional, current_depth)
            if child_depth > max_depth:
                max_depth = child_depth

        return max_depth

    # ------------------------------------------------------------------
    # Constraint counting
    # ------------------------------------------------------------------

    def _count_constraints(self, node: Any) -> int:
        """Recursively count constrained fields in *node*.

        A field is considered *constrained* if it appears in any of:
        * A ``required`` array (each member counts once).
        * A property whose schema contains an ``enum`` keyword.
        * A property whose schema contains a ``pattern`` keyword.
        * A property whose schema contains a ``const`` keyword.
        * A property whose schema specifies ``minimum``, ``maximum``,
          ``minLength``, ``maxLength``, ``minItems``, or ``maxItems``
          (numeric / string / array bounds).

        Deduplication within a single ``properties`` block is *not* performed –
        a field that is both required *and* has an ``enum`` counts twice,
        which provides a more accurate measure of overall constraint density.
        """
        if not isinstance(node, dict):
            return 0

        total = 0

        # --- required array -------------------------------------------------
        required = node.get("required")
        if isinstance(required, list):
            total += sum(1 for item in required if isinstance(item, str))

        # --- properties: inspect each child schema for constraint keywords ---
        properties = node.get("properties")
        if isinstance(properties, dict):
            for child_schema in properties.values():
                if not isinstance(child_schema, dict):
                    continue
                # Enum-typed field
                if "enum" in child_schema:
                    total += 1
                # Pattern-constrained string field
                if "pattern" in child_schema:
                    total += 1
                # Const-constrained field
                if "const" in child_schema:
                    total += 1
                # Numeric / string / array bounds
                for bound_kw in (
                    "minimum",
                    "maximum",
                    "exclusiveMinimum",
                    "exclusiveMaximum",
                    "minLength",
                    "maxLength",
                    "minItems",
                    "maxItems",
                    "minProperties",
                    "maxProperties",
                ):
                    if bound_kw in child_schema:
                        total += 1
                        break  # count at most once per property for bounds

                # Recurse into child schema
                total += self._count_constraints(child_schema)

        # --- items ----------------------------------------------------------
        items = node.get("items")
        if isinstance(items, dict):
            total += self._count_constraints(items)
        elif isinstance(items, list):
            for item_schema in items:
                total += self._count_constraints(item_schema)

        prefix_items = node.get("prefixItems")
        if isinstance(prefix_items, list):
            for item_schema in prefix_items:
                total += self._count_constraints(item_schema)

        # --- combinators ----------------------------------------------------
        for keyword in ("anyOf", "oneOf", "allOf"):
            branches = node.get(keyword)
            if isinstance(branches, list):
                for branch in branches:
                    total += self._count_constraints(branch)

        # --- definition registries ------------------------------------------
        for registry_key in ("$defs", "definitions"):
            registry = node.get(registry_key)
            if isinstance(registry, dict):
                for def_schema in registry.values():
                    total += self._count_constraints(def_schema)

        # --- single-schema keywords -----------------------------------------
        for keyword in (
            "if",
            "then",
            "else",
            "not",
            "contains",
            "propertyNames",
            "additionalProperties",
        ):
            sub = node.get(keyword)
            if isinstance(sub, dict):
                total += self._count_constraints(sub)

        return total
