"""Generate a >200-field extraction fixture from a CIA World Factbook profile.

The World Factbook is public-domain and fact-dense (~335 fields per country). This
turns one country's ``factbook.json`` into three aligned artifacts:

  * a JSON Schema (one string field per fact, nested by category),
  * a readable profile *document* rendered from the SAME data (so every field's
    value genuinely appears in the text — a fair extraction test, not a mismatch),
  * a ground-truth ``{path: value}`` map for accuracy scoring.

Usage:
    uv run python scripts/gen_factbook_fixture.py

Reads:  tests/fixtures/documents/_cache/us_factbook.json
Writes: tests/fixtures/schemas/factbook_us.json
        tests/fixtures/documents/_cache/factbook_us.txt
        tests/fixtures/schemas/factbook_us_truth.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent
_SRC = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "us_factbook.json"
_SCHEMA_OUT = _ROOT / "tests" / "fixtures" / "schemas" / "factbook_us.json"
_DOC_OUT = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "factbook_us.txt"
_TRUTH_OUT = _ROOT / "tests" / "fixtures" / "schemas" / "factbook_us_truth.json"

_TAG_RE = re.compile(r"<[^>]+>")


def _snake(label: str) -> str:
    """Normalise a Factbook label to a snake_case path segment."""
    s = _TAG_RE.sub("", label).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "field"


def _clean(text: str) -> str:
    """Strip HTML tags and collapse whitespace from a Factbook value."""
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


def _flatten(node: Any, label_path: list[str], key_path: list[str], out: list[tuple]) -> None:
    """Collect (key_path, label_path, value) for every ``{text: ...}`` leaf."""
    if not isinstance(node, dict):
        return
    if "text" in node and isinstance(node["text"], str):
        value = _clean(node["text"])
        if value:
            out.append((list(key_path), list(label_path), value))
        return
    for label, child in node.items():
        if label == "note" or not isinstance(child, dict):
            continue
        _flatten(child, [*label_path, label], [*key_path, _snake(label)], out)


def _nest_schema(leaves: list[tuple]) -> dict:
    """Build a nested JSON Schema (string leaves) from flattened paths."""
    root: dict[str, Any] = {"type": "object", "properties": {}}
    for key_path, label_path, _ in leaves:
        node = root
        for seg in key_path[:-1]:
            props = node["properties"]
            props.setdefault(seg, {"type": "object", "properties": {}})
            node = props[seg]
        node["properties"][key_path[-1]] = {
            "type": "string",
            "description": " - ".join(_clean(label) for label in label_path),
        }
    return root


def _render_doc(leaves: list[tuple]) -> str:
    """Render the leaves as a readable category-grouped profile document."""
    lines = ["UNITED STATES — COUNTRY PROFILE (CIA World Factbook)", ""]
    current_cat = None
    for _, label_path, value in leaves:
        cat = _clean(label_path[0])
        if cat != current_cat:
            lines += ["", f"== {cat.upper()} ==", ""]
            current_cat = cat
        label = ": ".join(_clean(label) for label in label_path[1:]) or cat
        lines.append(f"{label}: {value}")
    return "\n".join(lines)


def main() -> None:
    data = json.loads(_SRC.read_text(encoding="utf-8"))
    leaves: list[tuple] = []
    for cat, node in data.items():
        _flatten(node, [cat], [_snake(cat)], leaves)

    schema = _nest_schema(leaves)
    document = _render_doc(leaves)
    truth = {".".join(kp): val for kp, _, val in leaves}

    _SCHEMA_OUT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    _DOC_OUT.write_text(document, encoding="utf-8")
    _TRUTH_OUT.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"fields: {len(leaves)}")
    print(f"document chars: {len(document):,}")
    print(f"schema -> {_SCHEMA_OUT}")
    print(f"document -> {_DOC_OUT}")
    print(f"truth -> {_TRUTH_OUT}")


if __name__ == "__main__":
    main()
