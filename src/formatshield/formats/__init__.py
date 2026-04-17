"""Multi-format output support for FormatShield."""

from __future__ import annotations

from formatshield.formats.latex import LaTeXFormat, extract_latex_answer, wrap_latex_prompt
from formatshield.formats.markdown import (
    MarkdownFormat,
    extract_markdown_answer,
    wrap_markdown_prompt,
)
from formatshield.formats.router import FormatRouter, OutputFormat
from formatshield.formats.xml import XMLFormat, extract_xml_answer, wrap_xml_prompt

__all__ = [
    "FormatRouter",
    "LaTeXFormat",
    "MarkdownFormat",
    "OutputFormat",
    "XMLFormat",
    "extract_latex_answer",
    "extract_markdown_answer",
    "extract_xml_answer",
    "wrap_latex_prompt",
    "wrap_markdown_prompt",
    "wrap_xml_prompt",
]
