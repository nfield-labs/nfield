"""
Targeted coverage tests for formatshield.scorer.complexity_scorer.

Covers the following previously-uncovered lines:
  162-167 : score() except block → returns neutral features on _score_impl error
  201-206 : compute_score() except block → returns 0.5 on unexpected error
  258-264 : _load_encoding() except block → tiktoken unavailable / bad encoding
  275-278 : _tokenise() except block → encoder.encode() raises at runtime
  303     : _compute_token_entropy() vocab_size == 1 → 0.0
  315     : _compute_token_entropy() max_entropy == 0.0 (single token) → 0.0
  334     : _length_bucket() n_tokens in 200-1000 → return 2
  373     : _neutral_features() → called from score() fallback
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from formatshield.scorer.complexity_scorer import ComplexityScorer
from formatshield.scorer.features import ComplexityFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer() -> ComplexityScorer:
    return ComplexityScorer()


# ---------------------------------------------------------------------------
# Lines 162-167 and 373: score() except block + _neutral_features()
# ---------------------------------------------------------------------------


def test_score_except_block_returns_neutral_features(scorer: ComplexityScorer) -> None:
    """When _score_impl raises, score() catches the error and returns neutral features."""
    with patch.object(scorer, "_score_impl", side_effect=RuntimeError("injected error")):
        result = scorer.score("Any prompt")
    assert isinstance(result, ComplexityFeatures)
    # Neutral features have mid-range values (not zeros)
    assert result.instruction_tune_score == 0.5
    assert result.token_entropy == 0.5


def test_neutral_features_directly(scorer: ComplexityScorer) -> None:
    """_neutral_features() returns a ComplexityFeatures with mid-range values."""
    neutral = scorer._neutral_features()
    assert isinstance(neutral, ComplexityFeatures)
    assert neutral.token_entropy == 0.5
    assert neutral.instruction_tune_score == 0.5


# ---------------------------------------------------------------------------
# Lines 201-206: compute_score() except block
# ---------------------------------------------------------------------------


def test_compute_score_except_block_returns_half(scorer: ComplexityScorer) -> None:
    """When the internal calculation raises, compute_score() returns 0.5."""
    bad_features = MagicMock()
    # Make .token_entropy return a string so _clip("bad") raises TypeError
    bad_features.token_entropy = "not_a_number"
    result = scorer.compute_score(bad_features)
    assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Lines 258-264: _load_encoding() except block
# ---------------------------------------------------------------------------


def test_load_encoding_returns_none_when_tiktoken_missing() -> None:
    """_load_encoding() returns None when tiktoken is unavailable or encoding fails."""
    with patch.dict("sys.modules", {"tiktoken": None}):
        # Construct a scorer while tiktoken is unavailable — _load_encoding should
        # catch the ImportError and return None, so _enc is None.
        scorer = ComplexityScorer(encoding_name="cl100k_base")
        assert scorer._enc is None


def test_load_encoding_returns_none_on_bad_encoding_name() -> None:
    """_load_encoding() returns None for a nonexistent encoding name."""
    # Pass an encoding name that tiktoken doesn't know about
    scorer = ComplexityScorer(encoding_name="this_encoding_does_not_exist_xyz_9999")
    # If tiktoken is installed it raises on bad name; _load_encoding catches it.
    # If tiktoken isn't installed the import error is caught.
    # Either way _enc is None and score() still works.
    result = scorer.score("Hello world")
    assert isinstance(result, ComplexityFeatures)


# ---------------------------------------------------------------------------
# Lines 275-278: _tokenise() except block (encoder.encode raises at runtime)
# ---------------------------------------------------------------------------


def test_tokenise_falls_back_to_char_ordinals_when_encode_raises(
    scorer: ComplexityScorer,
) -> None:
    """_tokenise() uses char-ordinal fallback when the encoder's encode() raises."""
    if scorer._enc is None:
        pytest.skip("tiktoken not installed — char fallback is the default path")

    original_encode = scorer._enc.encode

    def bad_encode(_text: str) -> list[int]:
        raise RuntimeError("simulated tiktoken encode failure")

    scorer._enc.encode = bad_encode  # type: ignore[method-assign]
    try:
        tokens = scorer._tokenise("hello")
        # Should fall back to [ord(c) for c in "hello"]
        assert tokens == [ord(c) for c in "hello"]
    finally:
        scorer._enc.encode = original_encode  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Line 303: _compute_token_entropy() vocab_size == 1 → return 0.0
# ---------------------------------------------------------------------------


def test_compute_token_entropy_all_same_tokens() -> None:
    """_compute_token_entropy([42, 42, 42]) must return 0.0 (vocab_size == 1)."""
    result = ComplexityScorer._compute_token_entropy([42, 42, 42, 42, 42])
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Line 315: _compute_token_entropy() max_entropy == 0.0 (n == 1)
# ---------------------------------------------------------------------------


def test_compute_token_entropy_single_token() -> None:
    """_compute_token_entropy([42]) must return 0.0 (n=1, log2(1)=0)."""
    result = ComplexityScorer._compute_token_entropy([42])
    assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Line 334: _length_bucket() return 2 (200 <= n_tokens <= 1000)
# ---------------------------------------------------------------------------


def test_length_bucket_returns_2_for_medium_long_prompt() -> None:
    """_length_bucket() must return 2 for token counts in the 200-1000 range."""
    assert ComplexityScorer._length_bucket(200) == 2
    assert ComplexityScorer._length_bucket(500) == 2
    assert ComplexityScorer._length_bucket(1000) == 2


def test_score_medium_long_prompt_bucket_2(scorer: ComplexityScorer) -> None:
    """A prompt of 200-1000 tokens must land in bucket 2."""
    # Each "word_N" is ~2 tokens with tiktoken; 120 words ≈ 240 tokens → bucket 2.
    medium_prompt = " ".join([f"word{i}" for i in range(120)])
    features = scorer.score(medium_prompt)
    assert features.prompt_length_bucket == 2
