"""Document retrieval for the Stage 2.5 pre-pass.

Chunking (`chunk_document`), a diacritic-folding lexical tokenizer, and the BMX
entropy-weighted scorer used to rank chunks against each field group.
"""

from __future__ import annotations

from formatshield.retrieval._bmx import BMXIndex, bmx_rescore, build_bmx_index
from formatshield.retrieval._chunker import chunk_document
from formatshield.retrieval._tokenize import fold_diacritics, tokenize

__all__ = [
    "BMXIndex",
    "bmx_rescore",
    "build_bmx_index",
    "chunk_document",
    "fold_diacritics",
    "tokenize",
]
