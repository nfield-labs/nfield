"""FULL pipeline trace -> a complete .txt, using the War & Peace example.

Runs the REAL FormatShield pipeline (the same stage functions the engine calls)
one stage at a time and writes EVERYTHING to a text file with NO truncation:

    input (document + schema)
      S0  resource calibration         (chars/token, C_eff, M_O, C_usable)
      S1  schema flatten + difficulty   (EVERY field: D(f) parts, tau, deps)
      S2A structural grouping           (every group + all its fields)
      S2B document pre-pass             (REAL chunking: segments + BMX retrieval)
      S2C capacity packing              (K_min, every leaf, every field in it)
      S3  excerpt build                 (each leaf's document budget)
      S4  extraction                    (per leaf: the EXACT FULL PROMPT it sends
                                         = document D + fields, the raw reply, and
                                         the parsed path=value results)
      S5  validation + retry
      S5b recovery
      S6  assembly / merge              (flat blackboard -> nested JSON, full)

Usage:
    uv run python scripts/trace_pipeline.py

Writes: test-results/wp_pipeline_trace.txt   (needs GROQ_API_KEY + cached book).
The file is large on purpose — it contains the whole document each leaf sees.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from formatshield.schema._types import CapacityLeaf, Field

_ROOT = Path(__file__).parent.parent
_SCHEMA_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "war_and_peace_200fields.json"
_DOC_PATH = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "war_and_peace.txt"
_OUT_PATH = _ROOT / "test-results" / "wp_pipeline_trace.txt"

_MODEL = "groq/llama-3.3-70b-versatile"
_CONTEXT_WINDOW = 20_000
_MAX_OUTPUT = 5_000


def _load_env() -> None:
    env = _ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


class Trace:
    """Buffered writer: collects lines, prints progress, flushes to file once."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def __call__(self, line: str = "") -> None:
        self._lines.append(line)

    def block(self, text: str) -> None:
        """Append a multi-line text block verbatim (no per-line processing)."""
        self._lines.append(text)

    def head(self, title: str) -> None:
        self._lines += ["", "=" * 80, title, "=" * 80]
        print(f"  {title}")

    def sub(self, title: str) -> None:
        self._lines += ["", f"--- {title} " + "-" * max(0, 75 - len(title))]

    def flush(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True)
        path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")
        return None


def _toks(chars: int, cpt: float) -> int:
    return round(chars / cpt) if cpt else 0


def _leaf_load(leaf: CapacityLeaf) -> float:
    return sum(1.0 + f.difficulty for f in leaf.fields)


def _field_row(f: Field) -> str:
    desc = ""
    if isinstance(f.schema_node, dict):
        desc = (f.schema_node.get("description") or "")[:60]
    cons = f" constraints={f.constraints}" if f.constraints else ""
    deps = f" dep_in={sorted(f.dep_in)}" if f.dep_in else ""
    return (
        f"  D={f.difficulty:.3f} tau={f.tau:>4.0f} [{f.type:<7}] {f.path:<46} {desc!r}{cons}{deps}"
    )


def _write_extracted(extracted: dict[str, Any], state: Any, needs_reval: Any) -> None:
    """Mirror s4._write_extracted_to_blackboard so the manual S4 loop matches."""
    bb = state.blackboard
    for path, value in extracted.items():
        if value is needs_reval:
            bb.mark_needs_revalidation(path)
        elif value is None:
            bb.mark_failed(path, "field not found in document (LLM output NULL)")
        else:
            bb.write(path, value)


async def main() -> None:
    _load_env()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY not set (need it for the live S4/S5 stages)")
    if not _DOC_PATH.exists():
        raise SystemExit(f"cached book missing: {_DOC_PATH}")

    from formatshield.config import ExtractionConfig
    from formatshield.extraction._papt import select_template
    from formatshield.extraction._prompt import build_extraction_prompt
    from formatshield.extraction._sfep import NEEDS_REVALIDATION, parse_sfep
    from formatshield.pipeline.s0_resources import run_stage_0
    from formatshield.pipeline.s1_schema import run_stage_1
    from formatshield.pipeline.s2a_structure import run_stage_2a
    from formatshield.pipeline.s2b_prepass import run_stage_2b
    from formatshield.pipeline.s2c_packing import run_stage_2c
    from formatshield.pipeline.s3_excerpt import run_stage_3
    from formatshield.pipeline.s5_validate import run_stage_5
    from formatshield.pipeline.s5b_recover import run_recovery_pass
    from formatshield.pipeline.s6_assemble import run_stage_6
    from formatshield.providers import from_model

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    document = _DOC_PATH.read_text(encoding="utf-8")
    config = ExtractionConfig(max_retry_rounds=1)
    provider = from_model(_MODEL, context_window=_CONTEXT_WINDOW, max_output_tokens=_MAX_OUTPUT)

    t = Trace()
    t("FORMATSHIELD FULL PIPELINE TRACE  -  War & Peace, 200 fields (no truncation)")
    t(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    t(f"model: {_MODEL}   context_window={_CONTEXT_WINDOW}   max_output={_MAX_OUTPUT}")

    # ---------------------------------------------------------------- INPUT
    t.head("INPUT")
    t(f"document (D): {len(document):,} chars  (~{len(document.split()):,} words)")
    t("  full document is NOT printed here (3.3 MB) — but each leaf below prints the")
    t("  EXACT slice of D it receives, inside its prompt. That is the document the")
    t("  model actually sees per call.")
    t(f"  preview: {document[:200].strip()!r}")
    t(f"schema: {_SCHEMA_PATH.name}   top-level keys: {list(schema.get('properties', {}))}")

    # ---------------------------------------------------------------- S0
    t.head("STAGE 0  -  Resource calibration")
    t("WHAT: measure the model's tokenizer + read its real limits, so every later")
    t("budget is in true tokens (one provider round-trip to count tokens).")
    state = await run_stage_0(provider, config)
    t.sub("0.1 calibrated numbers")
    t(f"  chars_per_token (NSL)   = {state.chars_per_token:.3f}")
    t(f"  C_eff  (context window) = {state.C_eff:,} tokens")
    t(f"  M_O    (max output)     = {state.M_O:,} tokens")
    t(
        f"  C_usable (input budget) = {state.C_usable:,.0f} tokens   <- per-leaf document+schema budget"
    )
    need = _toks(len(document), state.chars_per_token)
    t(
        f"  document needs ~{need:,} tokens => {need / max(1, state.C_usable):.0f}x C_usable => MUST chunk + retrieve"
    )

    state.system_prompt = ""
    state.user_prompt = ""
    state.inject_dependencies = config.inject_dependencies
    state.knowledge_fallback = config.knowledge_fallback
    state.max_concurrent_calls = config.max_concurrent_calls

    # ---------------------------------------------------------------- S1
    t.head("STAGE 1  -  Schema flatten + difficulty + dependencies (ALL 200 fields)")
    t("WHAT: nested schema -> flat dot-paths; estimate output size tau; score")
    t("D(f) = 0.5*D_type + 0.3*D_constraint + 0.2*D_dep; build dependency DAG.")
    state = run_stage_1(state, schema)
    t.sub("1.1 totals")
    t(
        f"  total leaf fields = {len(state.fields)}   by type: {dict(Counter(f.type for f in state.fields))}"
    )
    ds = [f.difficulty for f in state.fields]
    t(f"  D: min={min(ds):.3f} mean={statistics.mean(ds):.3f} max={max(ds):.3f}")
    t.sub("1.2 EVERY field  (D, tau, type, path, description, constraints, deps)")
    for f in state.fields:
        t(_field_row(f))
    t.sub("1.3 dependency DAG")
    n_deps = sum(1 for v in state.dep_dag.values() if v)
    t(f"  fields with dependencies = {n_deps}  (0 => all leaves run in one parallel round)")
    for path, deps in state.dep_dag.items():
        if deps:
            t(f"  {path} depends on {sorted(deps)}")

    # ---------------------------------------------------------------- S2A
    t.head("STAGE 2A  -  Structural grouping (ALL groups, ALL fields)")
    t("WHAT: group fields by shared parent path; a group is the retrieval unit.")
    state = run_stage_2a(state)
    t(f"  total groups = {len(state.groups)}")
    for g in sorted(state.groups, key=lambda x: len(x.fields), reverse=True):
        t(
            f"  [{len(g.fields):>2} fields] {g.parent_path or '<root>'}: "
            f"{[f.path.split('.')[-1] for f in g.fields]}"
        )

    # ---------------------------------------------------------------- S2B
    t.head("STAGE 2B  -  Document pre-pass (REAL chunking + BMX retrieval)")
    t("WHAT: split D into boundary-aware segments, build a BMX (entropy-weighted")
    t("lexical) index, score each group's segments. Here the 3.3 MB book becomes")
    t("retrievable chunks — the real segmentation.")
    s = time.time()
    state = run_stage_2b(state, document, config)
    seg_lens = [len(seg.text) for seg in state.segments]
    t.sub("2B.1 segmentation")
    t(f"  segments produced = {len(state.segments):,}")
    if seg_lens:
        t(
            f"  size(chars): min={min(seg_lens)} median={statistics.median(seg_lens):.0f} "
            f"max={max(seg_lens)} mean={statistics.mean(seg_lens):.0f}"
        )
        t(
            f"  ~tokens/segment(median) = {_toks(int(statistics.median(seg_lens)), state.chars_per_token)}"
            f"   types: {dict(Counter(seg.segment_type for seg in state.segments))}"
        )
    t(f"  BMX index built: {state.bm25_index is not None}   [{time.time() - s:.1f}s]")
    t.sub("2B.2 retrieval per group  (matched count, top score, segment ids kept)")
    for g in sorted(state.groups, key=lambda x: len(x.fields), reverse=True):
        ids = [seg.segment_id for seg in g.matched_segments]
        top = max(g.segment_scores) if g.segment_scores else 0.0
        t(
            f"  {g.parent_path or '<root>':<34} matched={len(ids):>3} top={top:6.2f} "
            f"D_cost={g.D_cost} tok  seg_ids={ids}"
        )

    # ---------------------------------------------------------------- S2C
    t.head("STAGE 2C  -  Capacity packing (K, and EVERY field in EVERY leaf)")
    t("WHAT: decide K (LLM calls) and which fields go in each, under TWO limits:")
    t("output-token budget AND reliability load = sum(1 + D(f)) <= max_fields_per_call.")
    state = run_stage_2c(state, config)
    t.sub("2C.1 bounds")
    t(
        f"  max_fields_per_call = {config.max_fields_per_call}   K_min = {state.K_min}   "
        f"K = {len(state.leaves)}   total load = {sum(_leaf_load(lf) for lf in state.leaves):.1f}"
    )
    t.sub("2C.2 every leaf, every field")
    for leaf in state.leaves:
        t("")
        t(
            f"  LEAF #{leaf.leaf_id}: {len(leaf.fields)} fields  load={_leaf_load(leaf):.1f}  "
            f"overhead={leaf.overhead} tok  safe_output={leaf.safe_output} tok"
        )
        t(f"     groups: {sorted({g.parent_path or '<root>' for g in leaf.groups})}")
        for f in leaf.fields:
            t(f"       - D={f.difficulty:.3f} [{f.type:<7}] {f.path}")
    t.sub("2C.3 execution order")
    t(f"  rounds = {len(state.execution_order)}")
    for i, rnd in enumerate(state.execution_order):
        t(
            f"  round {i}: leaves {[lf.leaf_id for lf in rnd]}  (concurrent, cap={state.max_concurrent_calls})"
        )

    # ---------------------------------------------------------------- S3
    t.head("STAGE 3  -  Excerpt build (size of the document D each leaf gets)")
    t("WHAT: per leaf, gather its groups' retrieved segments, dedup, trim to the")
    t("leftover input budget. Full text is printed inside each leaf's prompt in S4.")
    state = run_stage_3(state)
    for leaf in state.leaves:
        ex = leaf.document_excerpt
        t(
            f"  LEAF #{leaf.leaf_id}: excerpt = {len(ex):,} chars (~{_toks(len(ex), state.chars_per_token):,} tok)"
        )

    # ---------------------------------------------------------------- S4
    t.head("STAGE 4  -  Extraction: the FULL prompt + reply for EVERY leaf")
    t("WHAT: each leaf is ONE LLM call. Below is the EXACT prompt sent (system +")
    t("user). The user message contains the field list AND the document D (the")
    t("real text the model reads), followed by the raw reply and the parsed values.")
    assert state.blackboard is not None
    s = time.time()
    for leaf in state.leaves:
        for f in leaf.fields:
            state.blackboard.mark_pending(f.path)
        template = select_template(leaf.fields, budget_tokens=leaf.safe_output)
        messages = build_extraction_prompt(
            leaf.fields,
            leaf.document_excerpt,
            template,
            system_prompt=state.system_prompt,
            user_prompt=state.user_prompt,
            knowledge_fallback=state.knowledge_fallback,
        )
        t("")
        t("#" * 80)
        t(f"# LEAF #{leaf.leaf_id} — extraction call  (max_tokens={leaf.safe_output})")
        t("#" * 80)
        t.sub(f"4.{leaf.leaf_id}.A  SYSTEM message")
        t.block(messages[0]["content"])
        t.sub(f"4.{leaf.leaf_id}.B  USER message  (fields + the document D this leaf sees)")
        t.block(messages[1]["content"])
        raw = await provider.complete(messages, max_tokens=leaf.safe_output)
        t.sub(f"4.{leaf.leaf_id}.C  RAW model reply")
        t.block(raw)
        extracted = parse_sfep(raw, leaf.fields)
        _write_extracted(extracted, state, NEEDS_REVALIDATION)
        state.K += 1
        t.sub(f"4.{leaf.leaf_id}.D  parsed path = value  ({len(extracted)} fields)")
        for path, value in extracted.items():
            t(f"     {path} = {value!r}")
    t.sub("4.summary  blackboard after extraction")
    t(f"  K (calls) = {state.K}   states: {state.blackboard.summary()}")
    miss = state.blackboard.get_missing() + state.blackboard.get_failed()
    t(f"  not-yet-extracted: {sorted(set(miss))}")
    t(f"  [{time.time() - s:.1f}s]")

    # ---------------------------------------------------------------- S5
    t.head("STAGE 5  -  Validation + retry")
    t("WHAT: type/constraint-check; retry failed fields (max_retry_rounds); cross-")
    t("leaf disagreements become CONFLICT.")
    s = time.time()
    before5 = state.blackboard.summary()
    state = await run_stage_5(state, provider, config)
    t(f"  before: {before5}")
    t(
        f"  after : {state.blackboard.summary()}   retry_rounds={state.retry_rounds}   [{time.time() - s:.1f}s]"
    )

    # ---------------------------------------------------------------- S5b
    t.head("STAGE 5b  -  Recovery pass (missing-field re-pack)")
    t("WHAT: re-pack still-missing fields into finer leaves and try once more.")
    t("Confirmed-absent fields are marked filled=None (NOT counted as extracted).")
    s = time.time()
    before = state.blackboard.summary()
    state = await run_recovery_pass(state, provider, config)
    t(f"  before: {before}")
    t(f"  after : {state.blackboard.summary()}   [{time.time() - s:.1f}s]")
    t(
        f"  still missing/failed: {sorted(set(state.blackboard.get_missing() + state.blackboard.get_failed()))}"
    )

    # ---------------------------------------------------------------- S6
    t.head("STAGE 6  -  Assembly / merge (full nested JSON)")
    t("WHAT: flat blackboard {path: value} -> nested JSON matching the schema.")
    result = run_stage_6(state)
    m = result.metadata
    t.sub("6.1 final metrics")
    t(
        f"  fields_total={m.fields_total}  fields_extracted={m.fields_extracted}  "
        f"fields_missing={m.fields_missing}"
    )
    t(f"  K/K_min={m.K}/{m.K_min}  quality={m.quality_score:.3f}  status={result.status.value}")
    t.sub("6.2 full result JSON (merged)")
    t.block(json.dumps(result.data, ensure_ascii=False, indent=2))

    t.head("END OF TRACE")
    t.flush(_OUT_PATH)
    print(
        f"\n  full trace written -> {_OUT_PATH}  ({_OUT_PATH.stat().st_size if _OUT_PATH.exists() else 0} bytes pending flush)"
    )


if __name__ == "__main__":
    asyncio.run(main())
