"""Lexical tokenizer shared by the retrieval layer.

A generic word tokenizer with Unicode diacritic folding, so an accented spelling
("Denísov") matches its plain form ("Denisov"). Used by the BMX index and by
targeted re-retrieval; kept here (not in a scorer module) because it is a
retriever-agnostic utility.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = ["fold_diacritics", "tokenize"]

# Split on runs of non-word characters.
_TOKENIZATION_PATTERN: str = r"\W+"
# Unicode category for accent marks; dropped so "Denísov" matches "Denisov".
_COMBINING_MARK_CATEGORY: str = "Mn"


def fold_diacritics(text: str) -> str:
    """Strip accents so an accented spelling matches its plain form.

    Args:
        text: Text to fold.

    Returns:
        The text with accent marks removed; unchanged for plain ASCII.

    Example:
        >>> fold_diacritics("Denísov")
        'Denisov'
        >>> fold_diacritics("café résumé")
        'cafe resume'
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != _COMBINING_MARK_CATEGORY)


def tokenize(text: str) -> list[str]:
    """Split text into lowercase, accent-folded words.

    Folding both the document and the query lets "Denísov" and "Denisov" match.

    Args:
        text: Text to tokenize.

    Returns:
        Lowercase, accent-folded tokens (non-empty only).

    Example:
        >>> tokenize("Hello, World!")
        ['hello', 'world']
        >>> tokenize("Denísov rode past Kutúzov")
        ['denisov', 'rode', 'past', 'kutuzov']
        >>> tokenize("")
        []
    """
    folded = fold_diacritics(text.lower())
    tokens = re.split(_TOKENIZATION_PATTERN, folded)
    return [t for t in tokens if t]
