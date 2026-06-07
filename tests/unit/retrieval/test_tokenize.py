"""Unit tests for the shared lexical tokenizer (diacritic folding + tokenize)."""

from __future__ import annotations

from formatshield.retrieval._tokenize import fold_diacritics, tokenize


class TestFoldDiacritics:
    def test_strips_accents(self) -> None:
        assert fold_diacritics("Denísov") == "Denisov"
        assert fold_diacritics("café résumé") == "cafe resume"

    def test_plain_ascii_unchanged(self) -> None:
        assert fold_diacritics("Napoleon") == "Napoleon"

    def test_empty(self) -> None:
        assert fold_diacritics("") == ""


class TestTokenize:
    def test_lowercases_and_splits(self) -> None:
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_folds_then_tokenizes(self) -> None:
        # Accented document spelling matches the plain query token.
        assert tokenize("Denísov rode past Kutúzov") == ["denisov", "rode", "past", "kutuzov"]

    def test_empty_returns_empty_list(self) -> None:
        assert tokenize("") == []

    def test_punctuation_only_returns_empty(self) -> None:
        assert tokenize("!!! ??? ...") == []

    def test_accented_query_matches_plain_corpus_token(self) -> None:
        # The whole point of folding: both sides reduce to the same token.
        assert tokenize("café")[0] == tokenize("cafe")[0]
