"""Render an extraction's grounded spans as a standalone HTML page.

Turns a result's provenance offsets into a reviewable artifact: the source document
with every located value highlighted in place, plus a table of field paths and the
exact text each value came from. One self-contained file - inline styles, no
scripts, no dependencies - so it can be attached to an audit or opened anywhere.

Requires provenance: run the extraction with ``ExtractionConfig(provenance=True)``.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nfield.types import ExtractionResult

__all__ = ["save_html"]

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>nfield extraction</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
       color: #1f2430; line-height: 1.6; }}
h1 {{ font-size: 1.3rem; }}
.summary {{ color: #55606e; margin-bottom: 1.5rem; }}
pre {{ white-space: pre-wrap; word-wrap: break-word; background: #f6f8fa;
      border: 1px solid #d8dee6; border-radius: 6px; padding: 1rem; }}
mark {{ background: #b7e3c0; border-radius: 3px; padding: 0 2px; }}
table {{ border-collapse: collapse; margin-top: 1.5rem; width: 100%; }}
th, td {{ border: 1px solid #d8dee6; padding: 6px 10px; text-align: left;
         font-size: 0.9rem; }}
th {{ background: #f6f8fa; }}
</style>
</head>
<body>
<h1>nfield extraction</h1>
<p class="summary">{summary}</p>
<pre>{document}</pre>
<table>
<tr><th>Field</th><th>Source text</th><th>Offsets</th></tr>
{rows}
</table>
</body>
</html>
"""


def save_html(result: ExtractionResult, document: str, path: str | Path | None = None) -> str:
    """Render *result*'s grounded spans over *document* as a standalone HTML page.

    Every provenance span is highlighted in the document, with the field path shown
    on hover, and listed in a table underneath with the exact source text. Spans
    that overlap an earlier one are listed in the table but not double-highlighted.

    Args:
        result: An extraction result carrying ``provenance`` (run with
            ``ExtractionConfig(provenance=True)``).
        document: The source document the extraction ran on.
        path: When given, the HTML is also written to this file (UTF-8).

    Returns:
        The complete HTML page as a string.

    Raises:
        ValueError: If the result carries no provenance.

    Example:
        >>> # html_page = save_html(result, document, "extraction.html")
    """
    if result.provenance is None:
        raise ValueError("result has no provenance; run with ExtractionConfig(provenance=True)")

    # Highlight in document order; a span starting inside an already-marked one is
    # skipped in the text (it still appears in the table below). Offsets from a
    # hand-edited results file may be malformed; a non-integer pair is dropped.
    spans = sorted(
        (
            (offsets[0], offsets[1], field)
            for field, offsets in result.provenance.items()
            if len(offsets) >= 2 and isinstance(offsets[0], int) and isinstance(offsets[1], int)
        ),
        key=lambda s: (s[0], s[1]),
    )
    parts: list[str] = []
    cursor = 0
    for start, end, field in spans:
        if start < cursor or end > len(document) or start >= end:
            continue
        parts.append(html.escape(document[cursor:start]))
        parts.append(
            f'<mark title="{html.escape(field, quote=True)}">'
            f"{html.escape(document[start:end])}</mark>"
        )
        cursor = end
    parts.append(html.escape(document[cursor:]))

    rows = "\n".join(
        f"<tr><td>{html.escape(field)}</td>"
        f"<td>{html.escape(document[start:end])}</td>"
        f"<td>[{start}, {end})</td></tr>"
        for start, end, field in spans
        if 0 <= start < end <= len(document)
    )

    meta = result.metadata
    summary_bits = [
        f"{meta.fields_extracted} of {meta.fields_total} fields extracted",
        f"{len(result.provenance)} located in the source",
    ]
    if meta.hallucination_rate is not None:
        summary_bits.append(f"hallucination rate {meta.hallucination_rate:.1%}")
    page = _PAGE_TEMPLATE.format(
        summary=html.escape(" · ".join(summary_bits)),
        document="".join(parts),
        rows=rows,
    )

    if path is not None:
        Path(path).write_text(page, encoding="utf-8")
    return page
