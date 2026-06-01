"""Document retrieval module for Stage 2.5 pre-pass.

Exports public retrieval functions for document chunking and BM25 scoring.
"""

from __future__ import annotations

from formatshield.retrieval._bm25 import (
    BM25Index,
    bm25_rescore,
    bm25_rescore_single,
    build_bm25_index,
)
from formatshield.retrieval._chunker import chunk_document

__all__ = [
    "BM25Index",
    "bm25_rescore",
    "bm25_rescore_single",
    "build_bm25_index",
    "chunk_document",
]
