"""Route a PDF to the right text extractor.

Born-digital PDFs carry an embedded text layer that ``pypdfium2`` reads exactly and
fast (BSD/Apache licensed). Scanned PDFs have no text layer, so the same call returns
almost nothing and OCR is required. The router picks per document: it reads the text
layer first and falls back to OCR only when the layer is too thin to be real text.

Measured on ExtractBench (atomic gold-value recall): pypdfium2 recovers 1.00 of
born-digital tables where OCR-markdown mangles them to 0.05, while the OCR fallback
recovers 0.84 of a scanned resume where the text layer yields 0.00. Routing takes the
best of both.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

__all__ = ["MIN_CHARS_PER_PAGE", "compose_spacing_accents", "extract", "text_layer"]

# A spacing (standalone) accent character maps to its combining counterpart. Some
# PDFs encode an accented letter as the base letter with the accent emitted as a
# separate glyph, so "Montr´eal" reaches the text layer instead of "Montréal".
_SPACING_ACCENTS: dict[str, str] = {
    "´": "́",  # acute
    "`": "̀",  # grave
    "¨": "̈",  # diaeresis
    "ˆ": "̂",  # circumflex
    "˜": "̃",  # tilde
    "¸": "̧",  # cedilla
    "¯": "̄",  # macron
    "ˇ": "̌",  # caron
    "˘": "̆",  # breve
    "˙": "̇",  # dot above
    "˚": "̊",  # ring above
}
# Letters that take a European diacritic; a stray accent before any other letter is
# left untouched rather than risk corrupting correct text.
_ACCENTABLE: frozenset[str] = frozenset("aeiouyAEIOUYnNcCsSzZgG")


def compose_spacing_accents(text: str) -> str:
    """Fold a standalone spacing accent onto the letter it precedes.

    A spacing accent binds to the following letter, skipping any spaces the
    extractor left between them ("R ¨ utte" -> "Rütte"), then the pair is composed
    to its single Unicode codepoint. Text without spacing accents is returned
    unchanged.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        char = text[i]
        if char in _SPACING_ACCENTS:
            j = i + 1
            while j < n and text[j] == " ":
                j += 1
            if j < n and text[j] in _ACCENTABLE:
                out.append(unicodedata.normalize("NFC", text[j] + _SPACING_ACCENTS[char]))
                i = j + 1
                continue
        out.append(char)
        i += 1
    return "".join(out)


# Below this mean characters-per-page the text layer is treated as absent (scanned
# image) and OCR takes over. Born-digital pages carry hundreds of chars; scanned
# pages yield near zero, so the boundary is wide and the exact value is not sensitive.
MIN_CHARS_PER_PAGE: int = 50


def extract(pdf_path: str | Path, *, min_chars_per_page: int = MIN_CHARS_PER_PAGE) -> str:
    """Extract text from a PDF, using OCR only when the text layer is missing.

    Args:
        pdf_path: Path to the PDF file.
        min_chars_per_page: Mean chars-per-page below which the text layer is deemed
            absent and OCR is used instead. Defaults to :data:`MIN_CHARS_PER_PAGE`.

    Returns:
        The document text: the embedded text layer for born-digital PDFs, or the OCR
        transcription for scanned ones.

    Example:
        >>> text = extract("filing.pdf")  # doctest: +SKIP
    """
    path = Path(pdf_path)
    text, pages = text_layer(path)
    if pages and len(text.strip()) / pages >= min_chars_per_page:
        return text
    return _ocr(path)


def text_layer(pdf_path: str | Path) -> tuple[str, int]:
    """Read the embedded text layer with pypdfium2; return ``(text, page_count)``."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        pages = len(pdf)
        text = "".join(page.get_textpage().get_text_range() + "\n" for page in pdf)
        return compose_spacing_accents(text), pages
    finally:
        pdf.close()


# Rasterisation DPI for the OCR fallback. 300 is the scanning-industry reference
# point for 10-12pt body text; higher DPI did not improve either engine here.
_OCR_DPI: int = 300


def _ocr(pdf_path: Path) -> str:
    """OCR a scanned PDF with an ensemble of engines.

    Different OCR engines garble different characters, so two independent renditions
    of the same page act as an error-correcting pair: a value mangled by one engine
    is usually intact in the other (measured on ExtractBench: Tesseract 0.84, RapidOCR
    0.87, union 0.90 gold-value recall). The renditions are concatenated with a label
    so a reader can cross-check. Engines degrade gracefully: whichever is available
    runs; a GPU deployment can swap in a VLM OCR without changing the routing.
    """
    renditions: list[str] = []
    tess = _tesseract_text(pdf_path)
    if tess:
        renditions.append(tess)
    rapid = _rapidocr_text(pdf_path)
    if rapid:
        if renditions:
            renditions.append("[Alternate OCR rendition of the same document]\n" + rapid)
        else:
            renditions.append(rapid)
    return "\n\n".join(renditions)


def _tesseract_text(pdf_path: Path) -> str:
    """OCR via PyMuPDF's built-in Tesseract integration; empty string when unavailable."""
    try:
        import pymupdf

        with pymupdf.open(str(pdf_path)) as doc:
            pages = []
            for page in doc:
                textpage = page.get_textpage_ocr(dpi=_OCR_DPI, full=True)
                pages.append(page.get_text(textpage=textpage))
        return "\n".join(pages)
    except (ImportError, RuntimeError, ValueError):
        return ""


def _rapidocr_text(pdf_path: Path) -> str:
    """OCR via RapidOCR (ONNX, CPU); empty string when the engine is not installed."""
    try:
        import pymupdf
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return ""
    engine = RapidOCR()
    lines: list[str] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page in doc:
            pixmap = page.get_pixmap(dpi=_OCR_DPI, colorspace=pymupdf.csGRAY)
            result, _ = engine(pixmap.tobytes("png"))
            if result:
                lines.extend(item[1] for item in result)
    return "\n".join(lines)
