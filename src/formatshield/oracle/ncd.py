"""
Normalized Compression Distance (NCD) between a prompt and a JSON schema.

NCD is a parameter-free similarity measure based on Kolmogorov complexity,
approximated here via ``zlib`` compression:

    NCD(p, σ) = (C(p ∥ σ_flat) − min(C(p), C(σ_flat))) / max(C(p), C(σ_flat))

where C(x) = len(zlib.compress(x)) and ∥ denotes concatenation.

σ_flat is the schema serialized as human-readable ``"field_name: type"`` lines
(no JSON syntax) so the compressor can find genuine repetition between the
prompt's vocabulary and the schema's field names / types.

NCD = 0: prompt and schema are near-identical (maximally aligned)
NCD = 1: prompt and schema share no compressible structure (misaligned)

High ΔK → prompt and schema are semantically distant → harder to format
correctly in one pass → higher Φ → prefer TTF.

Guards
------
If ``min(len(p_bytes), len(σ_bytes)) < 32`` the signal is too noisy
(zlib overhead dominates) and 0.5 is returned (neutral/uncertain).

Deps: ``zlib`` (stdlib only).
"""

from __future__ import annotations

import zlib

#: Minimum byte length below which NCD is unreliable (zlib header dominates).
_MIN_BYTES: int = 32

#: zlib compression level for fast approximation (level 6 = default).
_COMPRESS_LEVEL: int = 6


def _flatten_schema(schema: object, prefix: str = "", _depth: int = 0) -> str:
    """Serialize a JSON schema dict as ``"field_name: type"`` lines.

    Recurses into ``properties``, ``items``, ``allOf``, ``anyOf``, ``oneOf``,
    ``if``, ``then``, ``else`` so that nested field names are included.

    ``_depth`` guards against infinite recursion in pathological schemas.
    """
    if not isinstance(schema, dict) or _depth > 12:
        return ""

    lines: list[str] = []
    props = schema.get("properties", {})
    for name, sub in props.items():
        full = f"{prefix}.{name}" if prefix else name
        ftype = sub.get("type", "any") if isinstance(sub, dict) else "any"
        lines.append(f"{full}: {ftype}")
        if isinstance(sub, dict):
            if sub.get("properties"):
                nested = _flatten_schema(sub, full, _depth + 1)
                if nested:
                    lines.append(nested)
            # Also recurse into array sub-schemas that have item properties
            sub_items = sub.get("items", {})
            if isinstance(sub_items, dict) and sub_items.get("properties"):
                nested = _flatten_schema(sub_items, full + "[]", _depth + 1)
                if nested:
                    lines.append(nested)

    items = schema.get("items", {})
    if isinstance(items, dict) and items.get("properties"):
        # Only recurse into items when they have their own properties
        nested = _flatten_schema(items, prefix + "[]" if prefix else "[]", _depth + 1)
        if nested:
            lines.append(nested)

    for kw in ("allOf", "anyOf", "oneOf"):
        for clause in schema.get(kw, []):
            if isinstance(clause, dict):
                nested = _flatten_schema(clause, prefix, _depth + 1)
                if nested:
                    lines.append(nested)

    for kw in ("if", "then", "else"):
        clause = schema.get(kw)
        if isinstance(clause, dict):
            nested = _flatten_schema(clause, prefix, _depth + 1)
            if nested:
                lines.append(nested)

    return "\n".join(lines)


def _compress(data: bytes) -> int:
    """Return compressed byte length."""
    return len(zlib.compress(data, _COMPRESS_LEVEL))


def prompt_schema_ncd(prompt: str, schema: object) -> float:
    """Return NCD(prompt, schema_flat) ∈ [0, 1].

    Parameters
    ----------
    prompt:
        The raw user prompt string.
    schema:
        JSON Schema dict (or any object — non-dicts return 0.5).
    """
    if not isinstance(schema, dict):
        return 0.5

    sigma_flat = _flatten_schema(schema)
    if not sigma_flat:
        return 0.5

    p_bytes = prompt.encode("utf-8", errors="replace")
    s_bytes = sigma_flat.encode("utf-8", errors="replace")

    # Guard: too short for reliable NCD
    if min(len(p_bytes), len(s_bytes)) < _MIN_BYTES:
        return 0.5

    c_p = _compress(p_bytes)
    c_s = _compress(s_bytes)
    c_ps = _compress(p_bytes + b"\n" + s_bytes)

    denom = max(c_p, c_s)
    if denom == 0:
        return 0.5

    ncd = (c_ps - min(c_p, c_s)) / denom
    return max(0.0, min(1.0, ncd))
