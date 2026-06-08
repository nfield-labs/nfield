"""Document retrieval for the Stage 2.5 pre-pass.

Chunking (`chunk_document`), a diacritic-folding lexical tokenizer, the BMX
entropy-weighted scorer, and GLEAN — a schema-typed fusion retriever that augments
BMX with a morphological evidence signal (literal enum hits, format/pattern
matches, and label↔typed-value proximity) at CPU/lexical speed, no model.
"""

from __future__ import annotations

from formatshield.retrieval._bmx import BMXIndex, bmx_rescore, bmx_scores, build_bmx_index
from formatshield.retrieval._chunker import chunk_document
from formatshield.retrieval._glean import (
    GleanIndex,
    build_glean_index,
    field_best_segments,
    glean_rescore,
)
from formatshield.retrieval._morphology import (
    MorphologyIndex,
    build_morphology_index,
    field_type_classes,
)
from formatshield.retrieval._tokenize import fold_diacritics, tokenize

__all__ = [
    "BMXIndex",
    "GleanIndex",
    "MorphologyIndex",
    "bmx_rescore",
    "bmx_scores",
    "build_bmx_index",
    "build_glean_index",
    "build_morphology_index",
    "chunk_document",
    "field_best_segments",
    "field_type_classes",
    "fold_diacritics",
    "glean_rescore",
    "tokenize",
]
