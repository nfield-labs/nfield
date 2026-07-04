"""ExtractBench-specific scoring on top of the generic gold-diff scorer.

The ExtractBench dataset family carries conventions of its own: gold files use
the literal string "NOT_FOUND" as their absent-field sentinel, and each schema
annotates per-field judging tiers (``evaluation_config``), two of which the
official harness scores with an LLM judge (``string_semantic``, ``array_llm``).
This module keeps those conventions out of ``score.py`` so other benchmark
runners consume the generic scorer untouched.
"""

from __future__ import annotations

from typing import Any

from .score import (
    _ITEM_SEGMENT,
    FieldScore,
    Outcome,
    ScoreReport,
    _aggregate,
    _flatten,
    _resolve,
    score,
)

__all__ = ["llm_rejudge", "score_extractbench"]


def score_extractbench(
    extracted: dict[str, Any],
    gold_document: dict[str, Any],
    schema: dict[str, Any],
    *,
    call_failed: int = 0,
) -> tuple[ScoreReport, dict[str, Any]]:
    """Score one extraction under ExtractBench conventions.

    Flattens the gold document and maps its "NOT_FOUND" sentinel to ``None``
    (absent), so the generic scorer treats abstention on such fields as correct
    and a produced value as a hallucination - the official semantics.

    Args:
        extracted: The run's nested result data.
        gold_document: The nested gold document as read from the ``.gold.json``.
        schema: The dataset's JSON Schema (carries ``evaluation_config``).
        call_failed: Fields lost to API errors, passed through to the scorer.

    Returns:
        The deterministic :class:`ScoreReport` and the flattened gold key
        (needed by :func:`llm_rejudge`).
    """
    gold = {
        path: (None if value == "NOT_FOUND" else value)
        for path, value in _flatten(gold_document).items()
    }
    return score(extracted, gold, schema, call_failed=call_failed), gold


_JUDGE_STRING_PROMPT = """You are comparing a gold string against a predicted string for semantic \
equivalence. Consider normalization differences such as casing, punctuation, whitespace, \
abbreviations, formatting, and common synonyms. Penalize only meaningfully incorrect or \
incomplete predictions.

Field path: {path}

Gold string (JSON):
{gold}

Predicted string (JSON):
{predicted}

Respond with JSON only: {{"passed": true}} or {{"passed": false}}"""

_JUDGE_ARRAY_PROMPT = """You are evaluating how well a predicted array matches the gold array. \
Treat arrays as unordered collections when counting matches. A pair is a TRUE MATCH only if the \
items name or state the same thing; normalization differences (casing, punctuation, whitespace, \
abbreviations, name forms) do not break a match, but a different entity, entry, or value does. \
Each gold item matches at most one predicted item.

Field path: {path}

Gold array (JSON):
{gold}

Predicted array (JSON):
{predicted}

Respond with JSON only: {{"matched": <int>, "missed_gold": <int>, "spurious_pred": <int>}}"""


async def llm_rejudge(
    report: ScoreReport,
    gold: dict[str, Any],
    schema: dict[str, Any],
    complete: Any,
    *,
    max_prompt_chars: int = 200_000,
) -> ScoreReport:
    """Re-judge deterministic misses under the benchmark's own LLM tiers.

    The official harness scores ``string_semantic`` fields and ``array_llm``
    arrays with an LLM judge; the deterministic rules here are strictly harsher
    approximations of both. Fields those rules failed are re-asked of the judge
    and flip to CORRECT only on a pass - the judge can never take a correct
    field away, so this pass is monotone.

    Args:
        report: The deterministic :func:`score` result.
        gold: The flattened gold answer key.
        schema: The dataset's JSON Schema (carries ``evaluation_config``).
        complete: Async callable ``(prompt: str) -> str`` running the judge model.
        max_prompt_chars: Judgements whose prompt exceeds this are skipped (the
            deterministic outcome stands) so a huge array cannot overflow the
            judge's context window.

    Returns:
        A new :class:`ScoreReport` with judge-passed fields marked CORRECT.
    """
    import json as _json

    import json_repair as _json_repair

    def _parse(raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        if "```" in text:
            text = text.split("```")[1].removeprefix("json").strip()
        try:
            parsed = _json.loads(text)
        except _json.JSONDecodeError:
            parsed = _json_repair.loads(text)
        return parsed if isinstance(parsed, dict) else None

    fields = list(report.fields)

    # string_semantic: one pairwise meaning judgement per failed field.
    for i, f in enumerate(fields):
        if f.outcome is not Outcome.ACCURACY:
            continue
        node = _resolve(schema, f.path)
        if not (isinstance(node, dict) and node.get("evaluation_config") == "string_semantic"):
            continue
        prompt = _JUDGE_STRING_PROMPT.format(
            path=f.path, gold=_json.dumps(f.gold), predicted=_json.dumps(f.predicted)
        )
        if len(prompt) > max_prompt_chars:
            continue
        parsed = _parse(await complete(prompt))
        if parsed and parsed.get("passed") is True:
            fields[i] = FieldScore(f.path, f.field_type, f.gold, f.predicted, Outcome.CORRECT)

    # array_llm: one whole-array judgement per array with failed items; the judge
    # reports matched/missed and that many failed items flip to CORRECT.
    bases: dict[str, list[int]] = {}
    for i, f in enumerate(fields):
        m = _ITEM_SEGMENT.search(f.path)
        if not m:
            continue
        base = f.path[: m.start()].rstrip(".")
        node = _resolve(schema, base)
        if isinstance(node, dict) and node.get("evaluation_config") == "array_llm":
            bases.setdefault(base, []).append(i)
    for base, idxs in bases.items():
        failed = [i for i in idxs if fields[i].outcome is not Outcome.CORRECT]
        if not failed:
            continue
        gold_items = [fields[i].gold for i in idxs]
        pred_items = [fields[i].predicted for i in idxs if fields[i].predicted is not None]
        prompt = _JUDGE_ARRAY_PROMPT.format(
            path=base, gold=_json.dumps(gold_items), predicted=_json.dumps(pred_items)
        )
        if len(prompt) > max_prompt_chars:
            continue
        parsed = _parse(await complete(prompt))
        if not parsed:
            continue
        matched = parsed.get("matched")
        if not isinstance(matched, int):
            continue
        already = len(idxs) - len(failed)
        flips = max(0, min(matched - already, len(failed)))
        for i in failed[:flips]:
            f = fields[i]
            fields[i] = FieldScore(f.path, f.field_type, f.gold, f.predicted, Outcome.CORRECT)

    return _aggregate(tuple(fields), report.call_failed)
