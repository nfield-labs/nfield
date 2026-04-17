"""LLM Judge module for benchmark evaluation.

Provides SHA256-cached LLM-based evaluation of model responses against gold
answers, with task-specific rubrics for FormatShield benchmark tasks.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from typing import Any

# ---------------------------------------------------------------------------
# Task-specific rubrics
# ---------------------------------------------------------------------------

_TASK_RUBRICS: dict[str, str] = {
    "gsm": (
        "This is a math word problem. The response is CORRECT if the final "
        "numerical answer matches the gold answer (ignore units, formatting, "
        "and intermediate steps). The response is INCORRECT if the final "
        "number differs or is missing."
    ),
    "medical_ner": (
        "This is a medical named-entity recognition task. The response is "
        "CORRECT if all required entities (medications, dosages, conditions) "
        "match the gold answer exactly (case-insensitive). Partial matches "
        "or missing entities make it INCORRECT."
    ),
    "legal_extract": (
        "This is a legal information extraction task. The response is "
        "CORRECT if all extracted fields (parties, dates, clauses) match "
        "the gold answer semantically. Minor paraphrasing is acceptable. "
        "Missing required fields make it INCORRECT."
    ),
    "financial": (
        "This is a financial data extraction or reasoning task. The response "
        "is CORRECT if numerical values and entity names match the gold "
        "answer (allow rounding to 2 decimal places). The response is "
        "INCORRECT if key figures are wrong or missing."
    ),
    "classification": (
        "This is a classification task. The response is CORRECT only if the "
        "predicted class label matches the gold label exactly "
        "(case-insensitive). Any other label is INCORRECT."
    ),
    "gpqa": (
        "This is a graduate-level science question. The response is CORRECT "
        "if the answer matches the gold answer (a single letter A/B/C/D or "
        "the full answer text). Reasoning steps do not affect correctness."
    ),
    "zebralogic": (
        "This is a logic puzzle. The response is CORRECT if every assignment "
        "in the solution matches the gold answer exactly. A single wrong "
        "assignment makes it INCORRECT."
    ),
    "math500": (
        "This is a mathematics problem from the MATH benchmark. The response "
        "is CORRECT if the final answer is mathematically equivalent to the "
        "gold answer (allow equivalent forms, e.g. 1/2 == 0.5). The response "
        "is INCORRECT if the final answer differs."
    ),
}

_GENERIC_RUBRIC = (
    "Evaluate whether the model response is semantically equivalent to the "
    "gold answer. The response is CORRECT if it conveys the same information "
    "or reaches the same conclusion as the gold answer. The response is "
    "INCORRECT if it contradicts the gold answer, is missing key information, "
    "or gives a substantially different result."
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def build_judge_prompt(
    task: str,
    question: str,
    gold: str,
    response: str,
) -> str:
    """Build a task-specific LLM judge prompt.

    Constructs a prompt that instructs an LLM judge to evaluate a model
    response against the gold answer for a given task, using task-specific
    rubrics where available.

    Args:
        task: Benchmark task name (e.g. "gsm", "medical_ner"). Used to
            select the appropriate rubric. Unknown tasks use a generic rubric.
        question: The original question posed to the model.
        gold: The gold/reference answer for the question.
        response: The model's response to evaluate.

    Returns:
        A complete judge prompt string. The expected output from the LLM
        contains a line ``VERDICT: CORRECT`` or ``VERDICT: INCORRECT``.

    Example:
        >>> prompt = build_judge_prompt(
        ...     task="gsm",
        ...     question="What is 2+2?",
        ...     gold="4",
        ...     response='{"answer": 4}',
        ... )
        >>> "VERDICT" in prompt
        True
    """
    rubric = _TASK_RUBRICS.get(task.lower(), _GENERIC_RUBRIC)

    return (
        "You are an expert evaluator. Your task is to judge whether a model "
        "response is correct given the gold answer.\n\n"
        f"TASK TYPE: {task}\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        "---\n\n"
        f"QUESTION:\n{question}\n\n"
        f"GOLD ANSWER:\n{gold}\n\n"
        f"MODEL RESPONSE:\n{response}\n\n"
        "---\n\n"
        "Instructions:\n"
        "1. Compare the MODEL RESPONSE to the GOLD ANSWER using the RUBRIC.\n"
        "2. Think step by step, then on the final line write exactly:\n"
        "   VERDICT: CORRECT\n"
        "   or\n"
        "   VERDICT: INCORRECT\n\n"
        "Your evaluation:"
    )


def parse_verdict(text: str) -> bool:
    """Parse an LLM judge response to extract a boolean verdict.

    Looks for a ``VERDICT: CORRECT`` or ``VERDICT: INCORRECT`` line in the
    judge's response. If no explicit VERDICT line is found, falls back to
    scanning for the keywords "correct" or "incorrect" in the text.

    Handles reversal detection: if surrounding text contains negations like
    "not correct" or "is incorrect" near a CORRECT verdict, returns False.

    Args:
        text: Raw text output from the LLM judge.

    Returns:
        True if the verdict is CORRECT, False if INCORRECT or ambiguous.

    Example:
        >>> parse_verdict("After analysis... VERDICT: CORRECT")
        True
        >>> parse_verdict("The answer is wrong. VERDICT: INCORRECT")
        False
        >>> parse_verdict("Looks right to me.")
        True
        >>> parse_verdict("This is not correct at all.")
        False
    """
    # Primary: look for explicit VERDICT line
    verdict_match = re.search(
        r"VERDICT\s*:\s*(CORRECT|INCORRECT)",
        text,
        re.IGNORECASE,
    )
    if verdict_match:
        verdict_word = verdict_match.group(1).upper()
        if verdict_word == "INCORRECT":
            return False

        # Reversal detection: check the 60 chars before the VERDICT line
        start = max(0, verdict_match.start() - 60)
        context = text[start : verdict_match.start()].lower()
        reversal_patterns = [
            r"\bnot\s+correct\b",
            r"\bincorrect\b",
            r"\bwrong\b",
            r"\bis\s+not\b",
        ]
        for pattern in reversal_patterns:
            if re.search(pattern, context):
                return False

        return True

    # Fallback: keyword scan
    lower = text.lower()
    incorrect_pos = lower.rfind("incorrect")
    correct_pos = lower.rfind("correct")

    if incorrect_pos == -1 and correct_pos == -1:
        # No keywords found — default to False (conservative)
        return False

    if incorrect_pos > correct_pos:
        return False

    # "correct" appears after "incorrect" (or only "correct" present)
    # Still check for negation immediately before the last "correct"
    if correct_pos > 0:
        pre = lower[max(0, correct_pos - 20) : correct_pos]
        if re.search(r"\bnot\s+$|\bnot\s*$", pre.rstrip()):
            return False

    return True


# ---------------------------------------------------------------------------
# LLMJudge class
# ---------------------------------------------------------------------------


class LLMJudge:
    """SHA256-cached LLM judge for benchmark evaluation.

    Wraps any FormatShield-compatible backend to evaluate model responses
    against gold answers. Results are cached by a SHA256 key derived from
    (task, question, gold, response), so repeated calls are free.

    Optionally persists the cache to disk as JSON files, one per entry.

    Args:
        backend: Any object with an async ``generate(prompt: str) -> str``
            method. Use ``DryRunBackend`` in unit tests.
        cache_dir: Optional directory path for disk-based cache persistence.
            If None, only in-memory caching is used.

    Example:
        >>> from formatshield.backends.dryrun_backend import DryRunBackend
        >>> judge = LLMJudge(backend=DryRunBackend())
        >>> result = judge.judge("gsm", "2+2?", "4", '{"answer": 4}')
        >>> isinstance(result, bool)
        True
    """

    def __init__(
        self,
        backend: Any,
        cache_dir: str | None = None,
    ) -> None:
        self._backend = backend
        self._cache_dir = cache_dir
        self._cache: dict[str, bool] = {}

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def judge(
        self,
        task: str,
        question: str,
        gold: str,
        response: str,
    ) -> bool:
        """Judge synchronously, using SHA256 cache.

        Runs the async ``ajudge`` method in a new event loop, or falls back
        to the current running loop if one is already active.

        Args:
            task: Benchmark task name (e.g. "gsm", "medical_ner").
            question: The original question posed to the model.
            gold: The gold/reference answer.
            response: The model's response to evaluate.

        Returns:
            True if the response is judged CORRECT, False otherwise.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an event loop — create a new thread-safe future
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.ajudge(task, question, gold, response),
                )
                return future.result()

        return asyncio.run(self.ajudge(task, question, gold, response))

    async def ajudge(
        self,
        task: str,
        question: str,
        gold: str,
        response: str,
    ) -> bool:
        """Judge asynchronously, using SHA256 cache.

        Checks memory cache first, then disk cache (if ``cache_dir`` is set),
        then calls the backend and caches the result.

        Args:
            task: Benchmark task name (e.g. "gsm", "medical_ner").
            question: The original question posed to the model.
            gold: The gold/reference answer.
            response: The model's response to evaluate.

        Returns:
            True if the response is judged CORRECT, False otherwise.
        """
        key = self.cache_key(task, question, gold, response)

        # 1. Memory cache hit
        if key in self._cache:
            return self._cache[key]

        # 2. Disk cache hit
        disk_result = self._load_from_disk(key)
        if disk_result is not None:
            self._cache[key] = disk_result
            return disk_result

        # 3. Call the backend
        prompt = build_judge_prompt(task, question, gold, response)
        raw = await self._backend.generate(prompt)
        verdict = parse_verdict(raw)

        # 4. Store in memory and disk
        self._cache[key] = verdict
        self._save_to_disk(key, verdict)

        return verdict

    @staticmethod
    def cache_key(task: str, question: str, gold: str, response: str) -> str:
        """Compute SHA256 cache key for a judge request.

        The key is derived from the concatenation of all four inputs separated
        by pipe characters, ensuring uniqueness across (task, question, gold,
        response) tuples.

        Args:
            task: Benchmark task name.
            question: The original question.
            gold: The gold/reference answer.
            response: The model response.

        Returns:
            Hex string of the full SHA256 digest.

        Example:
            >>> key = LLMJudge.cache_key("gsm", "2+2?", "4", "4")
            >>> len(key) == 64
            True
        """
        raw = f"{task}|{question}|{gold}|{response}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def clear_cache(self) -> None:
        """Clear the judge's in-memory result cache.

        Does not remove any on-disk cache files. To remove disk files,
        delete the ``cache_dir`` directory manually.

        Example:
            >>> judge = LLMJudge(backend=None)
            >>> judge.clear_cache()
        """
        self._cache.clear()

    # ------------------------------------------------------------------
    # Private disk-cache helpers
    # ------------------------------------------------------------------

    def _cache_filename(self, key: str) -> str:
        """Return the full path for a cache file given a key.

        Args:
            key: SHA256 hex digest cache key.

        Returns:
            Absolute file path string.
        """
        short = key[:16]
        return os.path.join(self._cache_dir, f"judge_{short}.json")  # type: ignore[arg-type]

    def _load_from_disk(self, key: str) -> bool | None:
        """Load a cached verdict from disk.

        Args:
            key: SHA256 hex digest cache key.

        Returns:
            Cached bool verdict, or None if not found or cache_dir is unset.
        """
        if self._cache_dir is None:
            return None
        path = self._cache_filename(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            # Verify the full key to guard against SHA prefix collisions
            if data.get("key") == key:
                return bool(data["verdict"])
        except (json.JSONDecodeError, KeyError, OSError):
            pass
        return None

    def _save_to_disk(self, key: str, verdict: bool) -> None:
        """Persist a verdict to disk.

        Args:
            key: SHA256 hex digest cache key.
            verdict: The boolean verdict to store.
        """
        if self._cache_dir is None:
            return
        path = self._cache_filename(key)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"key": key, "verdict": verdict}, fh)
        except OSError:
            pass
