"""
FormatShield complexity scorer.

Computes a numeric complexity signal for an LLM inference request by combining
several independent features into a single :class:`ComplexityFeatures` object
and a scalar score in [0, 1].

The scalar score is consumed by :class:`~formatshield.oracle.ThresholdOracle`
to decide whether to apply the Think-Then-Format (TTF) strategy or to fall
back to direct structured generation.

Dependencies
------------
* ``tiktoken`` (pip install tiktoken) – tokenisation for entropy computation
* :class:`~formatshield.scorer.SchemaAnalyzer` – JSON-schema metrics
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

from formatshield.scorer.features import ComplexityFeatures
from formatshield.scorer.schema_analyzer import SchemaAnalyzer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CoT keyword list
# ---------------------------------------------------------------------------

_COT_KEYWORDS: frozenset[str] = frozenset(
    {
        "because",
        "therefore",
        "step",
        "analyze",
        "analyse",  # British spelling variant
        "calculate",
        "reason",
        "prove",
        "derive",
        "solve",
        "compare",
        "evaluate",
        "explain",
    }
)

# ---------------------------------------------------------------------------
# Instruction-tune score lookup (by model name prefix, longest-match first)
# ---------------------------------------------------------------------------

_INSTRUCTION_TUNE_PREFIXES: list[tuple[str, float]] = [
    # Native thinkers / heavy RLHF
    ("o1-mini", 1.0),
    ("o1-preview", 1.0),
    ("o3-mini", 1.0),
    ("o1", 1.0),
    ("o3", 1.0),
    # DeepSeek R1
    ("deepseek-r1", 0.9),
    # GPT-4 family
    ("gpt-4", 0.8),
    # Claude-3 family
    ("claude-3", 0.8),
    # Open-source instruction-tuned
    ("llama-3", 0.5),
    ("mistral", 0.5),
]

_DEFAULT_INSTRUCTION_TUNE_SCORE: float = 0.4

# ---------------------------------------------------------------------------
# Feature weights for the composite scalar score
# ---------------------------------------------------------------------------

# Weights must sum to 1.0 for the score to remain in [0, 1].
_WEIGHTS: dict[str, float] = {
    "token_entropy": 0.20,
    "schema_depth": 0.25,  # normalised 0–1
    "required_reasoning_ops": 0.20,  # normalised 0–1
    "instruction_tune_score": 0.15,
    "prompt_length_bucket": 0.10,  # normalised 0–1
    "schema_constraint_count": 0.10,  # normalised 0–1
}

# Normalisation caps (values above these are clipped to 1.0)
_SCHEMA_DEPTH_CAP: float = 10.0
_REASONING_OPS_CAP: float = 20.0
_SCHEMA_CONSTRAINT_CAP: float = 30.0
_PROMPT_LENGTH_BUCKET_MAX: float = 3.0

# Neutral feature values returned on any error
_NEUTRAL_FEATURES = {
    "token_entropy": 0.5,
    "schema_depth": 1,
    "required_reasoning_ops": 0,
    "instruction_tune_score": 0.5,
    "prompt_length_bucket": 1,
    "schema_constraint_count": 1,
}


class ComplexityScorer:
    """Compute :class:`ComplexityFeatures` and a scalar complexity score for
    an inference request.

    Parameters
    ----------
    encoding_name:
        Name of the tiktoken encoding to use.  Defaults to ``"cl100k_base"``
        which covers GPT-3.5 / GPT-4 / text-embedding-ada-002 vocabularies and
        is a reasonable proxy for most modern LLMs.

    Example::

        scorer = ComplexityScorer()
        features = scorer.score(prompt, schema=schema_dict, model_id="gpt-4o")
        scalar  = scorer.compute_score(features)
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._schema_analyzer = SchemaAnalyzer()
        self._enc = self._load_encoding(encoding_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        prompt: str,
        *,
        schema: dict | None = None,  # type: ignore[type-arg]
        model_id: str = "",
    ) -> ComplexityFeatures:
        """Compute and return a :class:`ComplexityFeatures` for *prompt*.

        Parameters
        ----------
        prompt:
            The full prompt string (system + user messages concatenated, or
            just the user message – any reasonable string works).
        schema:
            Optional target JSON Schema dict.  When ``None`` the schema-based
            features default to minimal values (depth=0, constraints=0).
        model_id:
            Model identifier string used to look up the instruction-tune score
            (e.g. ``"gpt-4o"``, ``"claude-3-sonnet-20240229"``).

        Returns
        -------
        ComplexityFeatures
            All six features populated.  Returns neutral features on any error.
        """
        try:
            return self._score_impl(prompt, schema=schema or {}, model_id=model_id)
        except Exception:
            logger.warning(
                "ComplexityScorer.score: unexpected error – returning neutral features",
                exc_info=True,
            )
            return self._neutral_features()

    def compute_score(self, features: ComplexityFeatures) -> float:
        """Convert a :class:`ComplexityFeatures` object to a single float in [0, 1].

        Higher values indicate higher complexity / more likely to benefit from TTF.

        Parameters
        ----------
        features:
            Previously computed feature object.

        Returns
        -------
        float
            Weighted linear combination of normalised features, clipped to [0, 1].
        """
        try:
            normalised = {
                "token_entropy": _clip(features.token_entropy),
                "schema_depth": _clip(features.schema_depth / _SCHEMA_DEPTH_CAP),
                "required_reasoning_ops": _clip(
                    features.required_reasoning_ops / _REASONING_OPS_CAP
                ),
                "instruction_tune_score": _clip(features.instruction_tune_score),
                "prompt_length_bucket": _clip(
                    features.prompt_length_bucket / _PROMPT_LENGTH_BUCKET_MAX
                ),
                "schema_constraint_count": _clip(
                    features.schema_constraint_count / _SCHEMA_CONSTRAINT_CAP
                ),
            }
            raw = sum(normalised[k] * _WEIGHTS[k] for k in _WEIGHTS)
            return _clip(raw)
        except Exception:
            logger.warning(
                "ComplexityScorer.compute_score: unexpected error – returning 0.5",
                exc_info=True,
            )
            return 0.5

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _score_impl(
        self,
        prompt: str,
        schema: dict,  # type: ignore[type-arg]
        model_id: str,
    ) -> ComplexityFeatures:
        """Core implementation (called inside a try/except in :meth:`score`)."""

        # 1. Tokenise the prompt
        tokens: list[int] = self._tokenise(prompt)

        # 2. Token entropy
        token_entropy = self._compute_token_entropy(tokens)

        # 3. Prompt length bucket
        prompt_length_bucket = self._length_bucket(len(tokens))

        # 4. Schema features
        schema_depth, schema_constraint_count = self._schema_analyzer.analyze(schema)

        # 5. Reasoning ops (CoT keyword count)
        required_reasoning_ops = self._count_reasoning_ops(prompt)

        # 6. Instruction-tune score
        instruction_tune_score = self._instruction_tune_score(model_id)

        return ComplexityFeatures(
            token_entropy=token_entropy,
            schema_depth=schema_depth,
            required_reasoning_ops=required_reasoning_ops,
            instruction_tune_score=instruction_tune_score,
            prompt_length_bucket=prompt_length_bucket,
            schema_constraint_count=schema_constraint_count,
        )

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_encoding(name: str) -> Any:
        """Load the tiktoken encoding, returning ``None`` on import failure."""
        try:
            import tiktoken  # type: ignore[import]

            return tiktoken.get_encoding(name)
        except Exception:
            logger.warning(
                "tiktoken not available or encoding '%s' not found – "
                "token entropy will use character-level fallback.",
                name,
            )
            return None

    def _tokenise(self, text: str) -> list[int]:
        """Return a list of token IDs for *text*.

        Falls back to a simple character-ordinal representation when tiktoken
        is unavailable, which still produces a meaningful entropy estimate.
        """
        if self._enc is not None:
            try:
                return self._enc.encode(text)
            except Exception as exc:  # tiktoken may raise on unusual inputs
                logger.debug("tiktoken encoding failed, using char fallback: %s", exc)
        # Character-level fallback
        return [ord(c) for c in text]

    # ------------------------------------------------------------------
    # Feature computation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_token_entropy(tokens: list[int]) -> float:
        """Compute normalised Shannon entropy over the token-ID distribution.

        Returns 0.0 for a single unique token, up to 1.0 for a perfectly
        uniform distribution (all tokens distinct).

        The normalisation uses ``log2(n_tokens)`` as the maximum possible
        entropy (achieved when every token is unique), so the result is
        always in [0, 1].
        """
        if not tokens:
            return 0.0

        n = len(tokens)
        counts = Counter(tokens)
        vocab_size = len(counts)

        if vocab_size == 1:
            return 0.0

        # Shannon entropy in bits
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0.0:
                entropy -= p * math.log2(p)

        # Normalise: max entropy for n tokens is log2(n)
        max_entropy = math.log2(n)
        if max_entropy == 0.0:
            return 0.0

        return _clip(entropy / max_entropy)

    @staticmethod
    def _length_bucket(n_tokens: int) -> int:
        """Map token count to a coarse length bucket.

        Bucket definitions:
            0 – short     (< 50 tokens)
            1 – medium    (50–200 tokens)
            2 – long      (200–1 000 tokens)
            3 – very long (> 1 000 tokens)
        """
        if n_tokens < 50:
            return 0
        if n_tokens < 200:
            return 1
        if n_tokens <= 1000:
            return 2
        return 3

    @staticmethod
    def _count_reasoning_ops(prompt: str) -> int:
        """Count the number of CoT reasoning keyword occurrences in *prompt*.

        The count is case-insensitive and matches whole-word occurrences to
        avoid false positives (e.g. "steps" should not match "step" multiple
        times; however, simple ``split()`` gives a close-enough approximation
        without a regex penalty for very long prompts).
        """
        prompt_lower = prompt.lower()
        words = prompt_lower.split()
        count = 0
        for word in words:
            # Strip common punctuation attached to the word
            stripped = word.strip(".,;:!?\"'()")
            if stripped in _COT_KEYWORDS:
                count += 1
        return count

    @staticmethod
    def _instruction_tune_score(model_id: str) -> float:
        """Return the instruction-tune score for *model_id* using prefix matching.

        Tries each prefix in order from most-specific to least-specific and
        returns the first match.  Defaults to :data:`_DEFAULT_INSTRUCTION_TUNE_SCORE`
        when no prefix matches.
        """
        lower = model_id.lower()
        for prefix, score in _INSTRUCTION_TUNE_PREFIXES:
            if lower.startswith(prefix):
                return score
        return _DEFAULT_INSTRUCTION_TUNE_SCORE

    @staticmethod
    def _neutral_features() -> ComplexityFeatures:
        """Return neutral / mid-range features used as a safe fallback."""
        return ComplexityFeatures(**_NEUTRAL_FEATURES)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clip *value* to the closed interval [*lo*, *hi*]."""
    return max(lo, min(hi, value))
