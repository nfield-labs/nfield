"""Tests for providers._token_budget - script-aware chars-per-token estimate."""

from __future__ import annotations

from nfield.providers._token_budget import chars_per_token_for_language


class TestCharsPerTokenForLanguage:
    """Script/language → characters-per-token ratio."""

    def test_english(self) -> None:
        assert chars_per_token_for_language("en") == 4.0
        assert chars_per_token_for_language("en-US") == 4.0

    def test_cjk(self) -> None:
        for tag in ("ja", "zh", "ko", "zh-Hans", "cjk"):
            assert chars_per_token_for_language(tag) == 1.5

    def test_other(self) -> None:
        for tag in ("fr", "de", "ar", "ru", "hi", "mixed", "other"):
            assert chars_per_token_for_language(tag) == 3.0

    def test_unknown_tag_is_other(self) -> None:
        assert chars_per_token_for_language("xx-YY") == 3.0

    def test_whitespace_and_case_normalized(self) -> None:
        assert chars_per_token_for_language("  JA  ") == 1.5
        assert chars_per_token_for_language("EN") == 4.0

    def test_always_positive(self) -> None:
        for tag in ("en", "ja", "fr", "zz"):
            assert chars_per_token_for_language(tag) > 0.0
