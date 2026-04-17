"""Unit tests for BatchProcessor (GROUP M — Stage 4)."""

from __future__ import annotations

import pytest

from formatshield.batch import (
    BatchError,
    BatchJobInfo,
    BatchProcessor,
    BatchStatus,
    BatchSuccess,
)


class TestBatchStatus:
    def test_all_statuses_exist(self) -> None:
        assert BatchStatus.PENDING == "pending"
        assert BatchStatus.PROCESSING == "processing"
        assert BatchStatus.COMPLETED == "completed"
        assert BatchStatus.FAILED == "failed"
        assert BatchStatus.CANCELLED == "cancelled"

    def test_is_str_enum(self) -> None:
        assert isinstance(BatchStatus.COMPLETED, str)


class TestBatchJobInfo:
    def test_fields(self) -> None:
        from datetime import datetime

        info = BatchJobInfo(
            job_id="fs_batch_abc",
            status=BatchStatus.PENDING,
            created_at=datetime.now(),
            model="dryrun/test",
            request_count=3,
        )
        assert info.job_id == "fs_batch_abc"
        assert info.request_count == 3
        assert info.completed_count == 0
        assert info.failed_count == 0
        assert info.completed_at is None
        assert info.error_message is None


class TestBatchSuccessAndError:
    def test_batch_success_fields(self) -> None:
        s: BatchSuccess[str] = BatchSuccess(
            custom_id="req_0", result="hello", usage_tokens=42
        )
        assert s.custom_id == "req_0"
        assert s.result == "hello"
        assert s.usage_tokens == 42

    def test_batch_error_fields(self) -> None:
        e = BatchError(
            custom_id="req_1",
            error_type="TimeoutError",
            error_message="timed out",
            raw_response=None,
        )
        assert e.custom_id == "req_1"
        assert e.error_type == "TimeoutError"
        assert e.raw_response is None


class TestBatchProcessorMakeJobId:
    def test_job_id_prefix(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        jid = proc._make_job_id(["hello", "world"])
        assert jid.startswith("fs_batch_")

    def test_job_id_deterministic(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        jid1 = proc._make_job_id(["a", "b"])
        jid2 = proc._make_job_id(["a", "b"])
        assert jid1 == jid2

    def test_job_id_differs_for_different_prompts(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        jid1 = proc._make_job_id(["prompt_a"])
        jid2 = proc._make_job_id(["prompt_b"])
        assert jid1 != jid2


class TestBatchProcessorResolveSchema:
    def test_none_response_model(self) -> None:
        proc = BatchProcessor(model="dryrun/test", response_model=None)
        assert proc._resolve_schema() is None

    def test_dict_response_model(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        proc = BatchProcessor(model="dryrun/test", response_model=schema)
        assert proc._resolve_schema() == schema

    def test_pydantic_model_response_model(self) -> None:
        from pydantic import BaseModel

        class MyModel(BaseModel):
            answer: str

        proc = BatchProcessor(model="dryrun/test", response_model=MyModel)
        schema = proc._resolve_schema()
        assert schema is not None
        assert "properties" in schema or "answer" in str(schema)


class TestBatchProcessorSubmit:
    @pytest.mark.asyncio
    async def test_submit_returns_job_info(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["Hello", "World"])
        assert isinstance(job, BatchJobInfo)
        assert job.request_count == 2

    @pytest.mark.asyncio
    async def test_submit_status_completed(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["What is 2+2?"])
        assert job.status == BatchStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_submit_completed_at_set(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["prompt"])
        assert job.completed_at is not None

    @pytest.mark.asyncio
    async def test_submit_custom_ids(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(
            ["p1", "p2"], custom_ids=["my_id_1", "my_id_2"]
        )
        results = await proc.results(job.job_id)
        ids = {r.custom_id for r in results}
        assert "my_id_1" in ids
        assert "my_id_2" in ids

    @pytest.mark.asyncio
    async def test_submit_default_custom_ids(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["p1", "p2"])
        results = await proc.results(job.job_id)
        ids = {r.custom_id for r in results}
        assert "req_0" in ids
        assert "req_1" in ids

    @pytest.mark.asyncio
    async def test_submit_raises_on_mismatched_ids(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        with pytest.raises(ValueError, match="custom_ids length"):
            await proc.submit(["p1", "p2"], custom_ids=["only_one"])

    @pytest.mark.asyncio
    async def test_submit_empty_prompts(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit([])
        assert job.request_count == 0
        assert job.status == BatchStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_submit_job_id_override(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["hello"], _job_id="custom_job_id")
        assert job.job_id == "custom_job_id"


class TestBatchProcessorStatus:
    @pytest.mark.asyncio
    async def test_status_returns_job_info(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        submitted = await proc.submit(["p"])
        info = await proc.status(submitted.job_id)
        assert info.job_id == submitted.job_id

    @pytest.mark.asyncio
    async def test_status_raises_for_unknown_job(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        with pytest.raises(KeyError, match="no_such_job"):
            await proc.status("no_such_job")


class TestBatchProcessorResults:
    @pytest.mark.asyncio
    async def test_results_length_matches_prompts(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["p1", "p2", "p3"])
        results = await proc.results(job.job_id)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_results_are_success_or_error(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["hello", "world"])
        results = await proc.results(job.job_id)
        for r in results:
            assert isinstance(r, (BatchSuccess, BatchError))

    @pytest.mark.asyncio
    async def test_results_raises_for_unknown_job(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        with pytest.raises(KeyError, match="no_such_job"):
            await proc.results("no_such_job")

    @pytest.mark.asyncio
    async def test_dryrun_all_succeed(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["What is 2+2?", "What is 3+3?"])
        results = await proc.results(job.job_id)
        successes = [r for r in results if isinstance(r, BatchSuccess)]
        assert len(successes) == 2

    @pytest.mark.asyncio
    async def test_completed_count_matches_successes(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["a", "b", "c"])
        results = await proc.results(job.job_id)
        success_count = sum(1 for r in results if isinstance(r, BatchSuccess))
        assert job.completed_count == success_count


class TestBatchProcessorCancel:
    @pytest.mark.asyncio
    async def test_cancel_sets_status_cancelled(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["p"])
        cancelled = await proc.cancel(job.job_id)
        assert cancelled.status == BatchStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_raises_for_unknown_job(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        with pytest.raises(KeyError, match="unknown_job"):
            await proc.cancel("unknown_job")

    @pytest.mark.asyncio
    async def test_cancel_sets_completed_at(self) -> None:
        proc = BatchProcessor(model="dryrun/test")
        job = await proc.submit(["p"])
        cancelled = await proc.cancel(job.job_id)
        assert cancelled.completed_at is not None
