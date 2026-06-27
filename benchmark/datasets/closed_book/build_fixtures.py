"""Generate closed-book benchmark fixtures from downloaded reference data.

Run once to materialise dataset folders under ``datasets/closed_book/<name>/`` — each with
``schema.json``, ``document.txt`` (empty: closed book), and ``gold.json`` — in the same
on-disk shape as the document-extraction fixtures, so the standard dataset loader reads
them unchanged and ``runner4`` stays a thin orchestrator.

Four domains, increasing size, each with a knowledge gradient (famous entities the model
recalls, obscure ones it should abstain on). The raw source dumps are not committed (large,
downloadable); re-fetch them into this directory before running:

    periodic_table.json  https://raw.githubusercontent.com/Bowserinator/Periodic-Table-JSON/master/PeriodicTableJSON.json
    countries.json       https://raw.githubusercontent.com/mledoze/countries/master/countries.json
    pokemon.json         https://raw.githubusercontent.com/Purukitto/pokemon-data.json/master/pokedex.json
    airports.dat         https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat

Usage:  uv run python benchmark/datasets/closed_book/build_fixtures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_NULL = (None, "", "\\N")


def _write(name: str, schema: dict[str, Any], gold: dict[str, Any]) -> None:
    out = _HERE / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (out / "document.txt").write_text("", encoding="utf-8")
    (out / "gold.json").write_text(
        json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{name}: {len(gold)} fields")


def _entity(fields: dict[str, str]) -> dict[str, Any]:
    return {"type": "object", "properties": {k: {"type": t} for k, t in fields.items()}}


def _accumulate(
    entities: list[tuple[str, dict[str, Any]]], fields: dict[str, str], target: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add whole entities until at least ``target`` gold fields accrue."""
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    gold: dict[str, Any] = {}
    for name, values in entities:
        if len(gold) >= target:
            break
        schema["properties"][name] = _entity(fields)
        for key in fields:
            value = values.get(key)
            if value not in _NULL:
                gold[f"{name}.{key}"] = value
    return schema, gold


def build_elements(target: int) -> None:
    fields = {
        "number": "integer",
        "symbol": "string",
        "period": "integer",
        "group": "integer",
        "phase": "string",
        "category": "string",
        "block": "string",
        "atomic_mass": "number",
        "electronegativity_pauling": "number",
        "density": "number",
        "melt": "number",
        "boil": "number",
    }
    elements = json.loads((_HERE / "periodic_table.json").read_text(encoding="utf-8"))["elements"]
    entities = [(e["name"], e) for e in elements]
    schema, gold = _accumulate(entities, fields, target)
    _write(f"elements_{len(gold)}", schema, gold)


def build_countries(target: int) -> None:
    fields = {
        "cca2": "string",
        "cca3": "string",
        "region": "string",
        "subregion": "string",
        "capital": "string",
        "landlocked": "boolean",
        "unMember": "boolean",
        "independent": "boolean",
    }
    raw = json.loads((_HERE / "countries.json").read_text(encoding="utf-8"))
    entities = []
    for c in raw:
        name = c.get("name", {}).get("common")
        if not name:
            continue
        cap = c.get("capital") or []
        entities.append((name, {**c, "capital": cap[0] if cap else None}))
    schema, gold = _accumulate(entities, fields, target)
    _write(f"countries_{len(gold)}", schema, gold)


def build_pokemon(target: int) -> None:
    fields = {
        "type": "string",
        "species": "string",
        "HP": "integer",
        "Attack": "integer",
        "Defense": "integer",
        "Speed": "integer",
    }
    raw = json.loads((_HERE / "pokemon.json").read_text(encoding="utf-8"))
    entities = []
    for p in raw:
        name = p.get("name", {}).get("english")
        base = p.get("base", {})
        if not name:
            continue
        entities.append(
            (
                name,
                {
                    "type": ", ".join(p.get("type", [])),
                    "species": p.get("species"),
                    "HP": base.get("HP"),
                    "Attack": base.get("Attack"),
                    "Defense": base.get("Defense"),
                    "Speed": base.get("Speed"),
                },
            )
        )
    schema, gold = _accumulate(entities, fields, target)
    _write(f"pokemon_{len(gold)}", schema, gold)


def build_airports(target: int) -> None:
    fields = {"city": "string", "country": "string", "name": "string"}
    rows = list(csv.reader((_HERE / "airports.dat").open(encoding="utf-8")))
    entities = []
    for row in rows:  # id,name,city,country,IATA,ICAO,...
        iata = row[4]
        if iata in _NULL:
            continue
        entities.append((iata, {"city": row[2], "country": row[3], "name": row[1]}))
    schema, gold = _accumulate(entities, fields, target)
    _write(f"airports_{len(gold)}", schema, gold)


def main() -> None:
    build_elements(50)
    build_countries(200)
    build_pokemon(600)
    build_airports(1000)


if __name__ == "__main__":
    main()
