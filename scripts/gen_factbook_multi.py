"""Generate a >800-field fixture by merging several CIA World Factbook profiles.

Same fair-test idea as the single-country generator, scaled up: merge N public-
domain country profiles into one schema (namespaced by country), one large readable
document, and one ground-truth ``{path: value}`` map. Every field's value appears
in the document, so the honest extraction number is meaningful and value accuracy
can be scored against truth.

Usage:
    uv run python scripts/gen_factbook_multi.py

Reads:  tests/fixtures/documents/_cache/factbook_<country>.json  (us, china, india, ...)
Writes: tests/fixtures/schemas/factbook_multi.json
        tests/fixtures/documents/_cache/factbook_multi.txt
        tests/fixtures/schemas/factbook_multi_truth.json
"""

from __future__ import annotations

import json
from pathlib import Path

from gen_factbook_fixture import _flatten, _snake  # reuse the proven flatten + clean

_ROOT = Path(__file__).parent.parent
_CACHE = _ROOT / "tests" / "fixtures" / "documents" / "_cache"
_SCHEMA_OUT = _ROOT / "tests" / "fixtures" / "schemas" / "factbook_multi.json"
_DOC_OUT = _CACHE / "factbook_multi.txt"
_TRUTH_OUT = _ROOT / "tests" / "fixtures" / "schemas" / "factbook_multi_truth.json"

# Country code -> source JSON file and display name. Enough to exceed 800 fields.
_COUNTRIES = [
    ("usa", "United States", _CACHE / "us_factbook.json"),
    ("china", "China", _CACHE / "factbook_china.json"),
    ("india", "India", _CACHE / "factbook_india.json"),
]


def main() -> None:
    schema: dict = {"type": "object", "properties": {}}
    truth: dict[str, str] = {}
    doc_parts: list[str] = []

    for code, name, src in _COUNTRIES:
        if not src.exists():
            print(f"skip {code}: {src.name} missing")
            continue
        data = json.loads(src.read_text(encoding="utf-8"))
        leaves: list[tuple] = []
        for cat, node in data.items():
            _flatten(node, [cat], [_snake(cat)], leaves)

        # Per-country schema sub-tree (namespaced by country code).
        country_node: dict = {"type": "object", "properties": {}}
        for key_path, label_path, value in leaves:
            node = country_node
            for seg in key_path[:-1]:
                props = node["properties"]
                props.setdefault(seg, {"type": "object", "properties": {}})
                node = props[seg]
            node["properties"][key_path[-1]] = {
                "type": "string",
                "description": f"{name} - " + " - ".join(label_path),
            }
            truth[f"{code}." + ".".join(key_path)] = value
        schema["properties"][code] = country_node

        # Per-country readable section of the merged document.
        doc_parts.append(f"\n\n######## {name.upper()} ########\n")
        current_cat = None
        for _, label_path, value in leaves:
            cat = label_path[0]
            if cat != current_cat:
                doc_parts.append(f"\n== {cat.upper()} ==\n")
                current_cat = cat
            label = ": ".join(label_path[1:]) or cat
            doc_parts.append(f"{name} {label}: {value}")

    document = "\n".join(doc_parts)

    def count(node: dict) -> int:
        n = 0
        for v in node.get("properties", {}).values():
            n += count(v) if v.get("type") == "object" else 1
        return n

    _SCHEMA_OUT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    _DOC_OUT.write_text(document, encoding="utf-8")
    _TRUTH_OUT.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"countries: {len([c for c in _COUNTRIES if c[2].exists()])}")
    print(f"total fields: {count(schema)}")
    print(f"document chars: {len(document):,}")
    print(f"schema -> {_SCHEMA_OUT}")


if __name__ == "__main__":
    main()
