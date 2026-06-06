"""Generate a STRONG, typed, deeply-nested fixture from a ClinicalTrials.gov study.

Unlike the all-string factbook fixtures, this exercises the full type system:
nested objects, integers, booleans, enums (with constraints), and arrays of
scalars — so it stresses tau/difficulty per type, constraint validation, and deep
nesting. The document is rendered from the SAME data, so every value is genuinely
present (a fair test), and a ground-truth map allows real accuracy scoring.

Big arrays (locations, references, many arms) are capped to the first few elements
and turned into indexed nested objects, so the schema is rich and deep (~250-400
leaves) without the source's tens of thousands of repeated entries.

    uv run python scripts/gen_clinicaltrial_fixture.py

Reads:  tests/fixtures/documents/_cache/clinicaltrial_NCT04368728.json
Writes: tests/fixtures/schemas/clinicaltrial.json          (typed nested schema)
        tests/fixtures/documents/_cache/clinicaltrial.txt  (readable document)
        tests/fixtures/schemas/clinicaltrial_truth.json    ({path: value})
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "clinicaltrial_NCT04368728.json"
_SCHEMA_OUT = _ROOT / "tests" / "fixtures" / "schemas" / "clinicaltrial.json"
_DOC_OUT = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "clinicaltrial.txt"
_TRUTH_OUT = _ROOT / "tests" / "fixtures" / "schemas" / "clinicaltrial_truth.json"

# Cap arrays so a 5 MB study with thousands of sites becomes a rich ~300-field
# schema, not a 97k-leaf explosion.
_ARRAY_CAP = 12
# Categorical fields → enum constraint (only applied when the real value is in the
# set, so the document value stays valid). Exercises enum + constraint difficulty.
_ENUMS: dict[str, list[str]] = {
    "studyType": ["INTERVENTIONAL", "OBSERVATIONAL", "EXPANDED_ACCESS"],
    "allocation": ["RANDOMIZED", "NON_RANDOMIZED", "NA"],
    "primaryPurpose": [
        "TREATMENT",
        "PREVENTION",
        "DIAGNOSTIC",
        "SUPPORTIVE_CARE",
        "SCREENING",
        "HEALTH_SERVICES_RESEARCH",
        "BASIC_SCIENCE",
        "OTHER",
    ],
    "masking": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "QUADRUPLE"],
    "sex": ["ALL", "FEMALE", "MALE"],
    "overallStatus": [
        "RECRUITING",
        "COMPLETED",
        "ACTIVE_NOT_RECRUITING",
        "TERMINATED",
        "NOT_YET_RECRUITING",
        "WITHDRAWN",
        "SUSPENDED",
        "ENROLLING_BY_INVITATION",
    ],
}

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _label(key: str) -> str:
    """camelCase / snake key → human words for descriptions and the document."""
    return _CAMEL.sub(" ", key).replace("_", " ").strip().capitalize()


def _scalar_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _walk(node: Any, key_path: list[str], label_path: list[str], out: list) -> dict | None:
    """Return the JSON-Schema node for *node*, appending leaves to *out*.

    Returns ``None`` for empty containers (nothing to extract).
    """
    if isinstance(node, dict):
        props: dict[str, Any] = {}
        for k, v in node.items():
            child = _walk(v, [*key_path, k], [*label_path, _label(k)], out)
            if child is not None:
                props[k] = child
        return {"type": "object", "properties": props} if props else None

    if isinstance(node, list):
        if not node:
            return None
        if all(not isinstance(x, (dict, list)) for x in node):
            # Array of scalars → one array field.
            out.append((".".join(key_path), list(label_path), list(node), "array"))
            return {
                "type": "array",
                "items": {"type": _scalar_type(node[0])},
                "description": " - ".join(label_path),
            }
        # Array of objects → cap and expand as indexed nested objects.
        props = {}
        for i, elem in enumerate(node[:_ARRAY_CAP]):
            child = _walk(elem, [*key_path, f"item_{i}"], [*label_path, f"#{i + 1}"], out)
            if child is not None:
                props[f"item_{i}"] = child
        return {"type": "object", "properties": props} if props else None

    # Scalar leaf.
    key = key_path[-1]
    node_schema: dict[str, Any] = {
        "type": _scalar_type(node),
        "description": " - ".join(label_path),
    }
    if key in _ENUMS and node in _ENUMS[key]:
        node_schema["enum"] = _ENUMS[key]
    out.append((".".join(key_path), list(label_path), node, node_schema["type"]))
    return node_schema


def _render_doc(leaves: list) -> str:
    lines = ["CLINICAL TRIAL RECORD (ClinicalTrials.gov NCT04368728)", ""]
    current_section = None
    for _, label_path, value, vtype in leaves:
        section = label_path[0]
        if section != current_section:
            lines += ["", f"== {section.upper()} ==", ""]
            current_section = section
        label = " - ".join(label_path[1:]) or section
        # Arrays comma-joined (no literal brackets) so the model reads a real list.
        rendered = ", ".join(str(v) for v in value) if vtype == "array" else str(value)
        lines.append(f"{label}: {rendered}")
    return "\n".join(lines)


def main() -> None:
    data = json.loads(_SRC.read_text(encoding="utf-8"))
    section = data["protocolSection"]

    schema: dict[str, Any] = {"type": "object", "properties": {}}
    leaves: list = []
    for mod, node in section.items():
        child = _walk(node, [mod], [_label(mod)], leaves)
        if child is not None:
            schema["properties"][mod] = child

    truth = {path: value for path, _, value, _ in leaves}
    document = _render_doc(leaves)

    _SCHEMA_OUT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    _DOC_OUT.write_text(document, encoding="utf-8")
    _TRUTH_OUT.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")

    types = {}
    for _, _, _, t in leaves:
        types[t] = types.get(t, 0) + 1
    print(f"leaf fields: {len(leaves)}")
    print(f"by type: {types}")
    print(f"document chars: {len(document):,}")
    print(f"schema -> {_SCHEMA_OUT}")


if __name__ == "__main__":
    main()
