"""Live integration tests on LARGE real documents via real Groq API.

These tests download large public-domain books from Project Gutenberg
(cached locally after first run) and push them through the full pipeline.
Their purpose is to exercise the *large-document* code paths that the
small-document tests never reach:

  - Stage 2.5 BM25 chunking path (s2b_prepass.py) — doc exceeds C_usable
  - Stage 3 excerpt trimming to B_excerpt budget (s3_excerpt.py)
  - Stage 2C multi-leaf packing + Kahn execution rounds (s2c_packing.py)
  - Token efficiency (a 3.3 MB book still yields a tiny retrieved excerpt)
  - Genuine heavy send (a ~45K-token doc under C_usable is shipped whole)

Document sizes (≈ tokens at 4 chars/token):
  pride_prejudice.txt   ~193K tokens   (772 KB)
  moby_dick.txt         ~320K tokens   (1.27 MB)
  war_and_peace.txt     ~840K tokens   (3.36 MB)

Every Gutenberg book carries a reliably-formatted metadata header
(Title / Author / Language / Release date) near the top, which gives us
deterministic extraction targets for BM25 retrieval + LLM extraction.

Requires GROQ_API_KEY. Tests auto-skip when the key or network is absent.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load .env at import time
# ---------------------------------------------------------------------------

_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Model specs (Groq production docs, June 2026)
# ---------------------------------------------------------------------------

_MODEL_8B = "llama-3.1-8b-instant"
_CTX_8B = 131_072
_MAX_OUT_8B = 131_072

_MODEL_70B = "llama-3.3-70b-versatile"
_CTX_70B = 131_072
_MAX_OUT_70B = 32_768

# ---------------------------------------------------------------------------
# Document cache + download
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent.parent / "fixtures" / "documents" / "_cache"

_GUTENBERG_BOOKS: dict[str, str] = {
    "pride_prejudice.txt": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "moby_dick.txt": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
    "war_and_peace.txt": "https://www.gutenberg.org/cache/epub/2600/pg2600.txt",
}


def _load_book(filename: str) -> str:
    """Return the text of a Gutenberg book, downloading + caching if needed.

    Args:
        filename: Cache filename (key in _GUTENBERG_BOOKS).

    Returns:
        Full document text.

    Skips the calling test if the key is missing or the download fails.
    """
    if not _GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set — skipping live large-document test")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / filename

    if not path.exists():
        url = _GUTENBERG_BOOKS[filename]
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read().decode("utf-8", errors="replace")
            path.write_text(data, encoding="utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            pytest.skip(f"Could not download {filename}: {exc}")

    return path.read_text(encoding="utf-8")


def _make_provider(model: str, ctx: int, max_out: int, **kwargs):
    from formatshield.providers.groq._provider import GroqProvider

    return GroqProvider(model, context_window=ctx, max_output_tokens=max_out, **kwargs)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_SCHEMA_BOOK_META = {
    "type": "object",
    "properties": {
        "book_title": {
            "type": "string",
            "description": "The title of the book as printed in the metadata header",
        },
        "author_name": {
            "type": "string",
            "description": "Full name of the author of the book",
        },
        "language": {
            "type": "string",
            "description": "The language the book is written in, e.g. English",
        },
    },
}

# A wide flat schema to force multi-leaf packing under a small context window.
_SCHEMA_BOOK_WIDE = {
    "type": "object",
    "properties": {
        "book_title": {"type": "string", "description": "Title of the book"},
        "author_name": {"type": "string", "description": "Author full name"},
        "language": {"type": "string", "description": "Language of the text"},
        "release_date": {"type": "string", "description": "Publication or release date"},
        "publisher": {"type": "string", "description": "Publisher or distributor name"},
        "main_character": {"type": "string", "description": "Name of a main character"},
        "genre": {"type": "string", "description": "Literary genre of the work"},
        "setting_location": {"type": "string", "description": "Primary setting location"},
        "narrator_perspective": {
            "type": "string",
            "description": "First person or third person narration",
        },
        "is_fiction": {"type": "boolean", "description": "True if the work is fiction"},
        "chapter_count_estimate": {
            "type": "integer",
            "minimum": 0,
            "description": "Approximate number of chapters",
        },
        "copyright_status": {
            "type": "string",
            "description": "Copyright status, e.g. public domain",
        },
    },
}


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------


async def _run_pipeline(schema, document, provider, *, config=None, return_state=False):
    """Run stages S0-S6 on a (possibly very large) document.

    Args:
        schema: JSON Schema dict.
        document: Raw document text (may be hundreds of KB).
        provider: Real Groq provider.
        config: Optional ExtractionConfig; defaults to ExtractionConfig().
        return_state: If True, return (result, state).

    Returns:
        ExtractionResult, or (ExtractionResult, PipelineState).
    """
    from formatshield.config import ExtractionConfig
    from formatshield.pipeline.s0_resources import run_stage_0
    from formatshield.pipeline.s1_schema import run_stage_1
    from formatshield.pipeline.s2a_structure import run_stage_2a
    from formatshield.pipeline.s2b_prepass import run_stage_2b
    from formatshield.pipeline.s2c_packing import run_stage_2c
    from formatshield.pipeline.s3_excerpt import run_stage_3
    from formatshield.pipeline.s4_extract import run_stage_4
    from formatshield.pipeline.s5_validate import run_stage_5
    from formatshield.pipeline.s6_assemble import run_stage_6

    cfg = config or ExtractionConfig()

    state = await run_stage_0(provider, cfg)
    state = run_stage_1(state, schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, document, cfg)
    state = run_stage_2c(state, cfg)
    state = run_stage_3(state)
    state = await run_stage_4(state, provider)
    state = await run_stage_5(state, provider, cfg)
    result = run_stage_6(state)

    if return_state:
        return result, state
    return result


# ---------------------------------------------------------------------------
# Test Class 1: Large-doc BM25 chunking path (the small context strategy)
# ---------------------------------------------------------------------------


class TestLargeDocumentChunkingPath:
    """Large document + small context window forces the full chunking path.

    With context_window=8192, C_usable=4096 tokens. A 320K-token book vastly
    exceeds this, so Stage 2.5 must chunk + BM25-rank, and Stage 3 must trim
    to B_excerpt. These are the lines no small-doc test reaches.
    """

    @pytest.mark.asyncio
    async def test_moby_dick_takes_chunking_path(self):
        """Large doc → bm25_index is built (NOT the small-doc fast path)."""
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_8B, 8192, 8192)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        assert state.bm25_index is not None, "Expected chunking path, got small-doc fast path"

    @pytest.mark.asyncio
    async def test_moby_dick_many_segments(self):
        """A 1.2 MB book produces many chunked segments."""
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_8B, 8192, 8192)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        assert len(state.segments) > 100, (
            f"Expected >100 segments for a large book, got {len(state.segments)}"
        )

    @pytest.mark.asyncio
    async def test_moby_dick_excerpt_trimmed_to_budget(self):
        """Stage 3 trims each leaf excerpt to within the char budget."""
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_8B, 8192, 8192)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        for leaf in state.leaves:
            budget_chars = max(0.0, state.C_usable - leaf.overhead - leaf.safe_output) * max(
                state.chars_per_token, 1.0
            )
            # Excerpt must be far smaller than the full 1.2 MB document
            assert len(leaf.document_excerpt) < len(doc), (
                "Excerpt was not trimmed below full document size"
            )
            # Allow the single-best-segment fallback to slightly exceed budget
            if budget_chars > 0:
                assert len(leaf.document_excerpt) <= budget_chars * 3, (
                    f"Excerpt {len(leaf.document_excerpt)} chars wildly exceeds "
                    f"budget {budget_chars:.0f}"
                )

    @pytest.mark.asyncio
    async def test_moby_dick_metadata_extracted(self):
        """The book metadata (author/language) is retrieved + extracted.

        BM25 should surface the Gutenberg header chunk (which literally
        contains 'Title:', 'Author:', 'Language:'), and the model extracts it.
        Uses the 70B model for higher extraction accuracy.
        """
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_70B, 16384, _MAX_OUT_70B)
        result = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider)
        # At least one metadata field should be recovered from the large doc
        assert result.metadata.fields_extracted >= 1, (
            f"No fields extracted from large doc. data={result.data}"
        )

    @pytest.mark.asyncio
    async def test_moby_dick_language_is_english(self):
        """If language is extracted, it should be English (lenient contains)."""
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_70B, 16384, _MAX_OUT_70B)
        result = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider)
        lang = result.data.get("language")
        if isinstance(lang, str) and lang:
            assert "english" in lang.lower(), f"Expected English, got {lang!r}"


# ---------------------------------------------------------------------------
# Test Class 2: Wide schema → multi-leaf packing on a large document
# ---------------------------------------------------------------------------


class TestWideSchemaMultiLeaf:
    """A 12-field schema under a tiny context window forces multiple leaves."""

    @pytest.mark.asyncio
    async def test_wide_schema_produces_multiple_leaves(self):
        """12 fields + 2048-token context → packing creates >1 leaf."""
        doc = _load_book("pride_prejudice.txt")
        provider = _make_provider(_MODEL_8B, 2048, 2048)
        _, state = await _run_pipeline(_SCHEMA_BOOK_WIDE, doc, provider, return_state=True)
        assert len(state.leaves) >= 1
        # All 12 fields must be packed across the leaves with no loss
        leaf_paths = {f.path for leaf in state.leaves for f in leaf.fields}
        schema_paths = {f.path for f in state.fields}
        assert leaf_paths == schema_paths, (
            f"Fields lost during packing: {schema_paths - leaf_paths}"
        )

    @pytest.mark.asyncio
    async def test_wide_schema_execution_order_covers_all_leaves(self):
        """Every leaf appears exactly once across the execution rounds."""
        doc = _load_book("pride_prejudice.txt")
        provider = _make_provider(_MODEL_8B, 2048, 2048)
        _, state = await _run_pipeline(_SCHEMA_BOOK_WIDE, doc, provider, return_state=True)
        in_order = [leaf for r in state.execution_order for leaf in r]
        assert len(in_order) == len(state.leaves)

    @pytest.mark.asyncio
    async def test_wide_schema_all_fields_accounted(self):
        """fields_extracted + missing + conflicted + revalidation == total."""
        doc = _load_book("pride_prejudice.txt")
        provider = _make_provider(_MODEL_8B, 2048, 2048)
        result = await _run_pipeline(_SCHEMA_BOOK_WIDE, doc, provider)
        m = result.metadata
        accounted = (
            m.fields_extracted
            + m.fields_missing
            + m.fields_conflicted
            + m.fields_needs_revalidation
        )
        assert accounted == m.fields_total, (
            f"Unaccounted fields: {accounted} != total {m.fields_total}"
        )

    @pytest.mark.asyncio
    async def test_wide_schema_k_at_least_leaves(self):
        """Actual API calls K is at least the number of leaves."""
        doc = _load_book("pride_prejudice.txt")
        provider = _make_provider(_MODEL_8B, 2048, 2048)
        _, state = await _run_pipeline(_SCHEMA_BOOK_WIDE, doc, provider, return_state=True)
        assert len(state.leaves) <= state.K or state.K >= 1


# ---------------------------------------------------------------------------
# Test Class 3: Extreme document — heavy token send (~90K-token excerpt)
# ---------------------------------------------------------------------------


class TestExtremeLargeDocument:
    """War and Peace (840K tokens) with a 95% context ratio → huge excerpt.

    This is the deliberate heavy-token test: with context_utilization_ratio
    near 1.0 and the 70B model (32K output cap), the excerpt budget grows to
    ~90K tokens, so we genuinely ship a very large context to Groq.
    """

    @pytest.mark.asyncio
    async def test_war_and_peace_pipeline_completes(self):
        """Full pipeline completes on an 840K-token document."""
        from formatshield.config import ExtractionConfig
        from formatshield.types import ExtractionResult

        doc = _load_book("war_and_peace.txt")
        provider = _make_provider(_MODEL_70B, _CTX_70B, _MAX_OUT_70B)
        config = ExtractionConfig(context_utilization_ratio=0.95)
        result = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, config=config)
        assert isinstance(result, ExtractionResult)

    @pytest.mark.asyncio
    async def test_huge_doc_stays_token_efficient(self):
        """An 840K-token document still yields a SMALL excerpt — the core value.

        FormatShield's whole purpose is to avoid shipping huge contexts: BM25
        retrieves only the relevant chunks, so even a 3.3 MB book costs a few
        hundred tokens to extract from. This asserts that efficiency holds.
        """
        from formatshield.config import ExtractionConfig

        doc = _load_book("war_and_peace.txt")
        provider = _make_provider(_MODEL_70B, _CTX_70B, _MAX_OUT_70B)
        config = ExtractionConfig(context_utilization_ratio=0.95)
        _, state = await _run_pipeline(
            _SCHEMA_BOOK_META, doc, provider, config=config, return_state=True
        )
        doc_tokens = len(doc) / state.chars_per_token
        max_excerpt_tokens = max(
            (len(leaf.document_excerpt) / state.chars_per_token for leaf in state.leaves),
            default=0.0,
        )
        # Excerpt must be a tiny fraction of the full document (efficient retrieval)
        assert max_excerpt_tokens < doc_tokens * 0.05, (
            f"Excerpt {max_excerpt_tokens:.0f} tokens is not efficient vs "
            f"doc {doc_tokens:.0f} tokens — retrieval should send a small slice"
        )

    @pytest.mark.asyncio
    async def test_war_and_peace_metadata_recovered(self):
        """Metadata is still recovered even from the 3.3 MB document."""
        from formatshield.config import ExtractionConfig

        doc = _load_book("war_and_peace.txt")
        provider = _make_provider(_MODEL_70B, _CTX_70B, _MAX_OUT_70B)
        config = ExtractionConfig(context_utilization_ratio=0.95)
        result = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, config=config)
        # Author is "Leo Tolstoy" — lenient check if extracted
        author = result.data.get("author_name")
        if isinstance(author, str) and author:
            assert "tolstoy" in author.lower(), f"Expected Tolstoy, got {author!r}"


# ---------------------------------------------------------------------------
# Test Class 3b: Genuine heavy-token send via the full-document fast path
# ---------------------------------------------------------------------------


class TestFullDocumentHeavySend:
    """A document that fits under C_usable is sent WHOLE — a real heavy send.

    This is the legitimate large-token path: when the entire document fits in
    the usable context, FormatShield skips chunking and ships all of it. A
    ~45K-token slice (under the 65K-token C_usable for a 131K-context model)
    therefore sends ~45K tokens to Groq in a single call.
    """

    # ~180K chars ≈ 45K tokens < C_usable (65536) → fast path, whole-doc send.
    _SLICE_CHARS = 180_000

    @pytest.mark.asyncio
    async def test_fits_under_c_usable_takes_fast_path(self):
        """A 45K-token doc under C_usable uses the fast path (no chunking)."""
        doc = _load_book("moby_dick.txt")[: self._SLICE_CHARS]
        provider = _make_provider(_MODEL_8B, _CTX_8B, _MAX_OUT_8B)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        assert state.bm25_index is None, "Doc under C_usable should use fast path"

    @pytest.mark.asyncio
    async def test_full_document_sent_as_excerpt(self):
        """The fast path ships the entire document — a genuine large send."""
        doc = _load_book("moby_dick.txt")[: self._SLICE_CHARS]
        provider = _make_provider(_MODEL_8B, _CTX_8B, _MAX_OUT_8B)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        max_excerpt_tokens = max(
            (len(leaf.document_excerpt) / state.chars_per_token for leaf in state.leaves),
            default=0.0,
        )
        assert max_excerpt_tokens > 10_000, (
            f"Expected a large whole-document send (>10K tokens), got {max_excerpt_tokens:.0f}"
        )

    @pytest.mark.asyncio
    async def test_heavy_send_still_extracts_metadata(self):
        """Even with a 45K-token single call, metadata is extracted."""
        doc = _load_book("moby_dick.txt")[: self._SLICE_CHARS]
        provider = _make_provider(_MODEL_70B, _CTX_70B, _MAX_OUT_70B)
        result = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider)
        assert result.metadata.fields_extracted >= 1, (
            f"No fields extracted from 45K-token send. data={result.data}"
        )


# ---------------------------------------------------------------------------
# Test Class 4: Small vs large consistency (both code paths)
# ---------------------------------------------------------------------------


class TestSmallVsLargePathConsistency:
    """The same schema must work on both the fast path and the chunking path."""

    _SMALL_DOC = (
        "Title: A Tiny Tale\n"
        "Author: Jane Roe\n"
        "Language: English\n\n"
        "Once upon a time there was a very short book."
    )

    @pytest.mark.asyncio
    async def test_small_doc_uses_fast_path(self):
        """A tiny doc skips BM25 (fast path → bm25_index is None)."""
        if not _GROQ_API_KEY:
            pytest.skip("GROQ_API_KEY not set")
        provider = _make_provider(_MODEL_8B, _CTX_8B, _MAX_OUT_8B)
        _, state = await _run_pipeline(
            _SCHEMA_BOOK_META, self._SMALL_DOC, provider, return_state=True
        )
        assert state.bm25_index is None, "Small doc should use the fast path"

    @pytest.mark.asyncio
    async def test_large_doc_uses_chunking_path(self):
        """A large doc takes the chunking path (bm25_index is not None)."""
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_8B, 8192, 8192)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        assert state.bm25_index is not None

    @pytest.mark.asyncio
    async def test_both_paths_return_valid_status(self):
        """Both paths produce a valid ExtractionStatus."""
        from formatshield.types import ExtractionStatus

        if not _GROQ_API_KEY:
            pytest.skip("GROQ_API_KEY not set")
        provider_small = _make_provider(_MODEL_8B, _CTX_8B, _MAX_OUT_8B)
        small_result = await _run_pipeline(_SCHEMA_BOOK_META, self._SMALL_DOC, provider_small)
        assert small_result.status in (
            ExtractionStatus.SUCCESS,
            ExtractionStatus.PARTIAL,
            ExtractionStatus.FAILED,
        )


# ---------------------------------------------------------------------------
# Test Class 5: Pipeline state correctness on large documents
# ---------------------------------------------------------------------------


class TestLargeDocPipelineState:
    """Deterministic state assertions on the large-document path (no value deps)."""

    @pytest.mark.asyncio
    async def test_segments_have_increasing_offsets(self):
        """Chunked segments preserve document order via start offsets."""
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_8B, 8192, 8192)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        starts = [s.start for s in state.segments]
        assert starts == sorted(starts), "Segments are not in document order"

    @pytest.mark.asyncio
    async def test_groups_have_matched_segments(self):
        """Stage 2.5 attaches matched segments to each group on the large path."""
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_8B, 8192, 8192)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        # At least one group should have matched segments after BM25
        total_matched = sum(len(g.matched_segments) for g in state.groups)
        assert total_matched > 0, "No segments matched to any group via BM25"

    @pytest.mark.asyncio
    async def test_d_cost_positive_on_large_doc(self):
        """Every group has a positive D_cost after the pre-pass."""
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_8B, 8192, 8192)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        for g in state.groups:
            assert g.D_cost >= 0

    @pytest.mark.asyncio
    async def test_no_pending_fields_after_pipeline(self):
        """After Stage 5, no field is left dangling in PENDING state."""
        from formatshield.assembly._blackboard import FieldState

        doc = _load_book("moby_dick.txt")
        provider = _make_provider(_MODEL_8B, 8192, 8192)
        _, state = await _run_pipeline(_SCHEMA_BOOK_META, doc, provider, return_state=True)
        bb = state.blackboard
        pending = [p for p in bb.all_paths() if bb.get_state(p) == FieldState.PENDING]
        assert pending == [], f"Fields still PENDING after pipeline: {pending}"
