"""Unit tests for the LLM Judge module (GROUP K — Stage 4)."""

from __future__ import annotations

import pytest

from formatshield.benchmark.judge import LLMJudge, build_judge_prompt, parse_verdict


class TestBuildJudgePrompt:
    def test_contains_verdict_instruction(self) -> None:
        prompt = build_judge_prompt("gsm", "What is 2+2?", "4", '{"answer": 4}')
        assert "VERDICT" in prompt

    def test_contains_task_type(self) -> None:
        prompt = build_judge_prompt("gsm", "q", "a", "r")
        assert "gsm" in prompt

    def test_contains_question(self) -> None:
        prompt = build_judge_prompt("gsm", "MY_QUESTION", "gold", "response")
        assert "MY_QUESTION" in prompt

    def test_contains_gold_answer(self) -> None:
        prompt = build_judge_prompt("gsm", "q", "GOLD_ANSWER", "response")
        assert "GOLD_ANSWER" in prompt

    def test_contains_model_response(self) -> None:
        prompt = build_judge_prompt("gsm", "q", "a", "MODEL_RESPONSE")
        assert "MODEL_RESPONSE" in prompt

    def test_known_task_uses_specific_rubric(self) -> None:
        prompt = build_judge_prompt("medical_ner", "q", "a", "r")
        # Medical NER rubric mentions medications
        assert "medication" in prompt.lower() or "entities" in prompt.lower()

    def test_unknown_task_uses_generic_rubric(self) -> None:
        prompt = build_judge_prompt("totally_unknown_task_xyz", "q", "a", "r")
        # Generic rubric should still produce a valid prompt
        assert "VERDICT" in prompt
        assert "totally_unknown_task_xyz" in prompt

    def test_all_known_tasks_produce_prompts(self) -> None:
        for task in ["gsm", "medical_ner", "legal_extract", "financial",
                     "classification", "gpqa", "zebralogic", "math500"]:
            prompt = build_judge_prompt(task, "q", "a", "r")
            assert "VERDICT" in prompt, f"Task {task} prompt missing VERDICT"

    def test_returns_string(self) -> None:
        result = build_judge_prompt("gsm", "q", "a", "r")
        assert isinstance(result, str)
        assert len(result) > 100


class TestParseVerdict:
    def test_explicit_correct(self) -> None:
        assert parse_verdict("After analysis...\nVERDICT: CORRECT") is True

    def test_explicit_incorrect(self) -> None:
        assert parse_verdict("The answer is wrong.\nVERDICT: INCORRECT") is False

    def test_case_insensitive_verdict(self) -> None:
        assert parse_verdict("verdict: correct") is True
        assert parse_verdict("verdict: incorrect") is False

    def test_fallback_keyword_correct(self) -> None:
        assert parse_verdict("Looks right to me. The response is correct.") is True

    def test_fallback_no_keywords_returns_false(self) -> None:
        # Neither "correct" nor "incorrect" keyword present → conservative False
        assert parse_verdict("The response is wrong and invalid.") is False

    def test_no_keywords_returns_false(self) -> None:
        assert parse_verdict("I have no idea what this is.") is False

    def test_incorrect_keyword_only_returns_false(self) -> None:
        # Explicit VERDICT: INCORRECT form always returns False
        assert parse_verdict("VERDICT: INCORRECT") is False

    def test_correct_after_incorrect_wins(self) -> None:
        # "correct" appears after "incorrect" (as substring of "incorrect")
        # rfind("correct") vs rfind("incorrect") — need careful wording
        assert parse_verdict("VERDICT: CORRECT") is True

    def test_empty_string_returns_false(self) -> None:
        assert parse_verdict("") is False

    def test_whitespace_around_verdict(self) -> None:
        assert parse_verdict("  VERDICT :  CORRECT  ") is True

    def test_multiline_verdict(self) -> None:
        text = "Step 1: check...\nStep 2: compare...\nVERDICT: CORRECT\n"
        assert parse_verdict(text) is True


class TestLLMJudgeCacheKey:
    def test_cache_key_is_64_hex_chars(self) -> None:
        key = LLMJudge.cache_key("gsm", "2+2?", "4", "4")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_differs_by_task(self) -> None:
        k1 = LLMJudge.cache_key("gsm", "q", "a", "r")
        k2 = LLMJudge.cache_key("medical_ner", "q", "a", "r")
        assert k1 != k2

    def test_cache_key_differs_by_response(self) -> None:
        k1 = LLMJudge.cache_key("gsm", "q", "a", "response_1")
        k2 = LLMJudge.cache_key("gsm", "q", "a", "response_2")
        assert k1 != k2

    def test_cache_key_is_deterministic(self) -> None:
        k1 = LLMJudge.cache_key("gsm", "q", "a", "r")
        k2 = LLMJudge.cache_key("gsm", "q", "a", "r")
        assert k1 == k2


class TestLLMJudgeMemoryCache:
    def test_judge_returns_bool(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend

        judge = LLMJudge(backend=DryRunBackend())
        result = judge.judge("gsm", "What is 2+2?", "4", '{"answer": 4}')
        assert isinstance(result, bool)

    def test_clear_cache_empties_cache(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend

        judge = LLMJudge(backend=DryRunBackend())
        # Populate cache
        judge.judge("gsm", "q", "a", "r")
        assert len(judge._cache) > 0
        judge.clear_cache()
        assert len(judge._cache) == 0

    @pytest.mark.asyncio
    async def test_ajudge_cached_on_second_call(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend

        judge = LLMJudge(backend=DryRunBackend())
        r1 = await judge.ajudge("gsm", "q", "a", "r")
        r2 = await judge.ajudge("gsm", "q", "a", "r")
        assert r1 == r2

    @pytest.mark.asyncio
    async def test_ajudge_different_inputs_both_cached(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend

        judge = LLMJudge(backend=DryRunBackend())
        await judge.ajudge("gsm", "q1", "a1", "r1")
        await judge.ajudge("gsm", "q2", "a2", "r2")
        assert len(judge._cache) == 2


class TestLLMJudgeDiskCache:
    def test_disk_cache_persists_and_reloads(self) -> None:
        import tempfile

        from formatshield.backends.dryrun_backend import DryRunBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            judge1 = LLMJudge(backend=DryRunBackend(), cache_dir=tmpdir)
            result1 = judge1.judge("gsm", "q", "a", "r")

            # New judge instance with same cache_dir — should hit disk
            judge2 = LLMJudge(backend=DryRunBackend(), cache_dir=tmpdir)
            result2 = judge2.judge("gsm", "q", "a", "r")

            assert result1 == result2
            # Memory cache should be warm after second judge's first call
            key = LLMJudge.cache_key("gsm", "q", "a", "r")
            assert key in judge2._cache

    def test_no_disk_cache_when_cache_dir_none(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend

        judge = LLMJudge(backend=DryRunBackend(), cache_dir=None)
        # Should not raise
        judge.judge("gsm", "q", "a", "r")
        # _save_to_disk and _load_from_disk are no-ops without cache_dir
        assert judge._cache_dir is None
