"""Offline excerpt instrumentation for the War & Peace 200-field run (no API).

Runs Stages 1 -> 2A -> 2.5 -> 2C -> 3 exactly as the real pipeline does, but
with a hand-set calibration (chars_per_token, C_eff, M_O) instead of a Stage 0
API call, so it is completely free. It then reports, per character/location/event
group, how much evidence actually reaches the leaf that owns it:

  * matched      — segments BM25 retrieved for the group (Stage 2.5)
  * in_excerpt   — how many of those survived CFCS trimming into the leaf excerpt
  * name_hits    — times the entity's own name appears in that leaf's excerpt
  * status       — whether the group's leaf excerpt likely supports extraction

This pinpoints why minor characters come back empty: a group whose leaf excerpt
contains its name only once (or zero times) cannot yield seven attributes.

Run:  uv run python scripts/inspect_wp_excerpts.py
"""

from __future__ import annotations

import json
from pathlib import Path

from formatshield.config import ExtractionConfig
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s1_schema import run_stage_1
from formatshield.pipeline.s2a_structure import run_stage_2a
from formatshield.pipeline.s2b_prepass import run_stage_2b
from formatshield.pipeline.s2c_packing import run_stage_2c
from formatshield.pipeline.s3_excerpt import run_stage_3
from formatshield.retrieval._bm25 import _fold_diacritics

# Same model limits as the live test; chars_per_token is the English average
# (the live run measures it via Stage 0, ~3.5-4.5 for llama — 4.0 is faithful
# enough for a structural diagnosis that makes no API call).
_CHARS_PER_TOKEN = 4.0
_CONTEXT_WINDOW = 20_000
_MAX_OUTPUT = 5_000

_ROOT = Path(__file__).parent.parent
_SCHEMA_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "war_and_peace_200fields.json"
_DOC_PATH = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "war_and_peace.txt"

# Entity name probes: the surname/word that should appear in the prose if the
# group's evidence is present. Keyed by group parent_path.
_NAME_PROBES: dict[str, str] = {
    "characters.pierre": "Pierre",
    "characters.napoleon": "Napoleon",
    "characters.denisov": "Denisov",
    "characters.old_bolkonsky": "Bolkonski",
    "characters.vasili": "Vasili",
    "characters.kutuzov": "Kutuzov",
    "characters.karataev": "Karataev",
}


def _build_calibrated_state() -> PipelineState:
    return PipelineState(
        chars_per_token=_CHARS_PER_TOKEN,
        C_eff=_CONTEXT_WINDOW,
        M_O=_MAX_OUTPUT,
        C_usable=_CONTEXT_WINDOW * ExtractionConfig().context_utilization_ratio,
    )


def main() -> None:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    document = _DOC_PATH.read_text(encoding="utf-8")
    config = ExtractionConfig()

    state = _build_calibrated_state()
    state = run_stage_1(state, schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, document, config)
    state = run_stage_2c(state, config)
    state = run_stage_3(state)

    print("=" * 78)
    print(f"document_chars   : {len(document):,}")
    print(f"segments (chunks): {len(state.segments):,}")
    print(f"groups           : {len(state.groups)}")
    print(f"leaves (K)       : {len(state.leaves)}   K_min={state.K_min}")
    print(f"C_usable tokens  : {state.C_usable:.0f}")
    print("=" * 78)

    # Map each group's parent_path -> (leaf, group) it landed in.
    group_leaf: dict[str, tuple[int, object]] = {}
    for leaf in state.leaves:
        for g in leaf.groups:
            group_leaf[g.parent_path] = (leaf.leaf_id, g)

    print("\nPer-leaf excerpt sizes:")
    for leaf in state.leaves:
        budget = max(0.0, state.C_usable - leaf.overhead - leaf.safe_output)
        budget_chars = int(budget * _CHARS_PER_TOKEN)
        print(
            f"  leaf {leaf.leaf_id}: groups={len(leaf.groups):<3} "
            f"fields={len(leaf.fields):<4} excerpt={len(leaf.document_excerpt):>7,} chars "
            f"(budget ~{budget_chars:,})"
        )

    print("\nEntity evidence reaching its leaf excerpt:")
    header = f"  {'group':28} {'leaf':>4} {'matched':>8} {'in_excerpt':>11} {'name_hits':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for parent_path, probe in _NAME_PROBES.items():
        entry = group_leaf.get(parent_path)
        if entry is None:
            print(f"  {parent_path:28} {'--':>4}  (group not packed?)")
            continue
        leaf_id, group = entry
        leaf = next(lf for lf in state.leaves if lf.leaf_id == leaf_id)
        # How many of the group's matched segments survived into the leaf excerpt.
        excerpt = leaf.document_excerpt
        # Fold diacritics on both sides so an accented prose spelling (Denísov)
        # counts as a hit for the unaccented probe (Denisov), matching how BM25
        # now indexes the text.
        folded_excerpt = _fold_diacritics(excerpt.lower())
        folded_probe = _fold_diacritics(probe.lower())
        matched = len(group.matched_segments)  # type: ignore[attr-defined]
        in_excerpt = sum(1 for seg in group.matched_segments if seg.text in excerpt)  # type: ignore[attr-defined]
        name_hits = folded_excerpt.count(folded_probe)
        print(f"  {parent_path:28} {leaf_id:>4} {matched:>8} {in_excerpt:>11} {name_hits:>10}")

    print(
        "\nReading: name_hits is how many times the character's name appears in the\n"
        "WHOLE leaf excerpt the model sees. A 7-attribute character with name_hits of\n"
        "0-1 cannot be fully extracted from the document — confirming the coverage gap."
    )


if __name__ == "__main__":
    main()
