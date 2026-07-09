"""CLI tests: `extract`, `batch`, and `inspect` via Typer's CliRunner.

End-to-end command behaviour (formats, metadata, config wiring) runs against a
mock provider so no network call is made; pure config-assembly and helper logic is
tested directly.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from nfield.cli._app import (
    OutputFormat,
    _build_config,
    _collect_documents,
    _parse_confidence,
    app,
)
from nfield.config import ExtractionConfig
from nfield.types import (
    ExtractionResult,
    ExtractionStatus,
    FieldResult,
    Metadata,
)

runner = CliRunner()

_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "paid": {"type": "boolean"},
    },
    "required": ["vendor"],
}


class _MockProvider:
    """Records the last prompt and echoes a canned SFEP body."""

    context_window = 8192
    max_output_tokens = 8192
    model_name = "mock/echo"

    def __init__(self, sfep_text: str) -> None:
        self._sfep = sfep_text
        self.last_messages: list[dict[str, str]] = []

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.last_messages = messages
        return self._sfep


def _write(path, data: str) -> str:
    path.write_text(data, encoding="utf-8")
    return str(path)


def _install_provider(monkeypatch, sfep_text: str) -> _MockProvider:
    """Point the engine's provider factory at a mock provider."""
    provider = _MockProvider(sfep_text)
    monkeypatch.setattr("nfield.engine._async.from_model", lambda _m, **_kw: provider)
    return provider


def _make_result(data: dict, **meta_overrides) -> ExtractionResult:
    """Build a minimal ExtractionResult for helpers that consume one directly."""
    base = {
        "K": 3,
        "K_min": 2,
        "optimality_gap": 0.1,
        "quality_score": 0.95,
        "confidence_level": "HIGH",
        "fields_extracted": len(data),
        "fields_total": len(data),
        "fields_missing": 0,
        "fields_conflicted": 0,
        "fields_needs_revalidation": 0,
        "per_field_confidence": dict.fromkeys(data, 0.9),
        "retry_rounds": 0,
    }
    base.update(meta_overrides)
    fields = tuple(FieldResult(path=k, value=v, confidence=0.9) for k, v in data.items())
    return ExtractionResult(
        data=data,
        metadata=Metadata(**base),
        status=ExtractionStatus.SUCCESS,
        fields=fields,
    )


class _SpyNField:
    """Stand-in for NField that records constructor args and returns canned results.

    Installed in place of ``nfield.cli._app.NField`` to assert how the CLI wires flags
    into the engine without running the pipeline.
    """

    last_instance: _SpyNField | None = None

    def __init__(self, model, schema, **kwargs) -> None:
        self.model_arg = model
        self.schema_arg = schema
        self.kwargs = kwargs
        self.extract_calls: list[str] = []
        self.batch_calls: list[list[str]] = []
        type(self).last_instance = self

    call_failed = 0

    def extract(self, document: str) -> ExtractionResult:
        self.extract_calls.append(document)
        return _make_result({"vendor": "Acme"}, fields_call_failed=type(self).call_failed)

    def extract_batch(self, documents, *, max_concurrent=None):
        self.batch_calls.append(list(documents))
        self.max_concurrent = max_concurrent
        return [
            _make_result({"vendor": f"doc{i}"}, fields_call_failed=type(self).call_failed)
            for i in range(len(documents))
        ]


class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "nfield" in result.stdout


class TestInspect:
    def test_inspect_reports_fields(self, tmp_path):
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(app, ["inspect", schema_file])
        assert result.exit_code == 0
        assert "Total leaf fields : 3" in result.stdout
        assert "K_min estimate" in result.stdout
        assert "vendor" in result.stdout

    def test_inspect_missing_file(self, tmp_path):
        result = runner.invoke(app, ["inspect", str(tmp_path / "nope.json")])
        assert result.exit_code != 0

    def test_inspect_invalid_json(self, tmp_path):
        bad = _write(tmp_path / "bad.json", "{not json")
        result = runner.invoke(app, ["inspect", bad])
        assert result.exit_code != 0

    def test_inspect_empty_schema_exits(self, tmp_path):
        empty = _write(tmp_path / "s.json", json.dumps({"type": "object", "properties": {}}))
        result = runner.invoke(app, ["inspect", empty])
        assert result.exit_code != 0


class TestExtract:
    def test_extract_writes_json_file(self, tmp_path, monkeypatch):
        _install_provider(monkeypatch, "vendor = Acme\ntotal = 42.5\npaid = true")
        doc = _write(tmp_path / "doc.txt", "Acme invoice total 42.50 paid")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        out = tmp_path / "out.json"

        result = runner.invoke(
            app,
            [
                "extract",
                doc,
                "--schema",
                schema_file,
                "--model",
                "mock/echo",
                "--output",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.stdout
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["vendor"] == "Acme"
        assert data["total"] == 42.5
        assert data["paid"] is True

    def test_extract_to_stdout(self, tmp_path, monkeypatch):
        _install_provider(monkeypatch, "vendor = Globex")
        doc = _write(tmp_path / "doc.txt", "Globex")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app, ["extract", doc, "--schema", schema_file, "--model", "mock/echo"]
        )
        assert result.exit_code == 0
        assert "Globex" in result.stdout

    def test_extract_jsonl_format_emits_full_result(self, tmp_path, monkeypatch):
        _install_provider(monkeypatch, "vendor = Acme")
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app,
            ["extract", doc, "--schema", schema_file, "--model", "mock/echo", "--format", "jsonl"],
        )
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout.strip())
        # JSONL carries the whole result envelope, not just the data object.
        assert payload["data"]["vendor"] == "Acme"
        assert payload["status"] in {"success", "partial", "failed"}
        assert "metadata" in payload
        assert "fields" in payload

    def test_extract_csv_format(self, tmp_path, monkeypatch):
        pytest.importorskip("pandas")
        _install_provider(monkeypatch, "vendor = Acme")
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app,
            ["extract", doc, "--schema", schema_file, "--model", "mock/echo", "-f", "csv"],
        )
        assert result.exit_code == 0, result.stdout
        assert "vendor" in result.stdout
        assert "Acme" in result.stdout

    def test_extract_show_metadata_goes_to_stderr(self, tmp_path, monkeypatch):
        _install_provider(monkeypatch, "vendor = Acme")
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app,
            ["extract", doc, "--schema", schema_file, "--model", "mock/echo", "--show-metadata"],
        )
        assert result.exit_code == 0, result.stdout
        assert "status=" in result.stderr
        assert "quality=" in result.stderr
        # Metadata never contaminates the stdout data payload.
        assert "status=" not in result.stdout

    def test_extract_with_instructions_flag(self, tmp_path, monkeypatch):
        provider = _install_provider(monkeypatch, "vendor = Acme")
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app,
            [
                "extract",
                doc,
                "--schema",
                schema_file,
                "--model",
                "mock/echo",
                "--instructions",
                "DOMAIN: invoices. Be exact.",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "DOMAIN: invoices. Be exact." in provider.last_messages[1]["content"]

    def test_extract_model_from_env(self, tmp_path, monkeypatch):
        _install_provider(monkeypatch, "vendor = Acme")
        monkeypatch.setenv("NFIELD_MODEL", "mock/echo")
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(app, ["extract", doc, "--schema", schema_file])
        assert result.exit_code == 0, result.stdout
        assert "Acme" in result.stdout

    def test_extract_missing_model_is_clean_error(self, tmp_path, monkeypatch):
        monkeypatch.delenv("NFIELD_MODEL", raising=False)
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(app, ["extract", doc, "--schema", schema_file])
        assert result.exit_code != 0
        assert "Extraction failed" in result.stdout or "Extraction failed" in (result.stderr or "")

    def test_extract_missing_document(self, tmp_path):
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app,
            [
                "extract",
                str(tmp_path / "nope.txt"),
                "--schema",
                schema_file,
                "--model",
                "mock/echo",
            ],
        )
        assert result.exit_code != 0


class TestExtractConfigWiring:
    """Flags must reach ExtractionConfig / the engine; assert via a spy engine."""

    def _run(self, tmp_path, monkeypatch, extra_args):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        args = ["extract", doc, "--schema", schema_file, "--model", "mock/echo", *extra_args]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.stdout
        return _SpyNField.last_instance

    def test_grounding_and_provenance_flags(self, tmp_path, monkeypatch):
        spy = self._run(tmp_path, monkeypatch, ["--ground-values", "--provenance"])
        cfg = spy.kwargs["config"]
        assert cfg.ground_values is True
        assert cfg.provenance is True

    def test_reasoning_and_fallback_flags(self, tmp_path, monkeypatch):
        spy = self._run(
            tmp_path, monkeypatch, ["--reasoning-model", "--fallback-model", "mock/big"]
        )
        cfg = spy.kwargs["config"]
        assert cfg.reasoning_model is True
        assert cfg.fallback_model == "mock/big"

    def test_connection_flags_reach_engine(self, tmp_path, monkeypatch):
        spy = self._run(
            tmp_path,
            monkeypatch,
            ["--api-key", "sk-x", "--base-url", "http://local", "--context-window", "16000"],
        )
        assert spy.kwargs["api_key"] == "sk-x"
        assert spy.kwargs["base_url"] == "http://local"
        assert spy.kwargs["context_window"] == 16000

    def test_negatable_default_true_flag(self, tmp_path, monkeypatch):
        spy = self._run(tmp_path, monkeypatch, ["--no-validate-schema"])
        assert spy.kwargs["config"].validate_schema is False

    def test_unset_flags_inherit_defaults(self, tmp_path, monkeypatch):
        spy = self._run(tmp_path, monkeypatch, [])
        cfg = spy.kwargs["config"]
        default = ExtractionConfig()
        assert cfg.validate_schema == default.validate_schema
        assert cfg.ground_values == default.ground_values
        assert cfg.inject_dependencies == default.inject_dependencies

    def test_confidence_pairs(self, tmp_path, monkeypatch):
        spy = self._run(
            tmp_path, monkeypatch, ["--confidence", "HIGH=0.95", "--confidence", "LOW=0.4"]
        )
        assert spy.kwargs["config"].confidence_thresholds == {"HIGH": 0.95, "LOW": 0.4}

    def test_closed_book_skips_document_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        # No document file exists; closed-book must not try to read one.
        result = runner.invoke(
            app,
            [
                "extract",
                str(tmp_path / "nope.txt"),
                "--schema",
                schema_file,
                "--model",
                "mock/echo",
                "--closed-book",
            ],
        )
        assert result.exit_code == 0, result.stdout
        spy = _SpyNField.last_instance
        assert spy.kwargs["config"].closed_book is True
        assert spy.extract_calls == [""]

    def test_think_budget_requires_both(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app,
            [
                "extract",
                doc,
                "--schema",
                schema_file,
                "--model",
                "mock/echo",
                "--think-budget-min",
                "50",
            ],
        )
        assert result.exit_code != 0


class TestCallFailureExit:
    """A run left incomplete by API/call failures must exit non-zero for scripts."""

    def test_extract_exits_nonzero_on_call_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        monkeypatch.setattr(_SpyNField, "call_failed", 2)
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        out = tmp_path / "out.json"
        result = runner.invoke(
            app,
            ["extract", doc, "-s", schema_file, "-m", "mock/echo", "-o", str(out)],
        )
        assert result.exit_code == 1
        assert "incomplete" in result.stderr
        # Output is still written before exiting, so no data is lost.
        assert out.exists()

    def test_batch_exits_nonzero_on_call_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        monkeypatch.setattr(_SpyNField, "call_failed", 1)
        a = _write(tmp_path / "a.txt", "one")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(app, ["batch", a, "-s", schema_file, "-m", "mock/echo"])
        assert result.exit_code == 1
        assert "incomplete" in result.stderr


class TestBatch:
    def test_batch_directory_to_jsonl_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        docs = tmp_path / "docs"
        docs.mkdir()
        _write(docs / "a.txt", "first")
        _write(docs / "b.txt", "second")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        out = tmp_path / "results.jsonl"
        result = runner.invoke(
            app,
            ["batch", str(docs), "--schema", schema_file, "--model", "mock/echo", "-o", str(out)],
        )
        assert result.exit_code == 0, result.stdout
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert all("data" in json.loads(line) for line in lines)
        # Directory inputs are read in sorted order.
        assert _SpyNField.last_instance.batch_calls[0] == ["first", "second"]

    def test_batch_explicit_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        a = _write(tmp_path / "a.txt", "one")
        b = _write(tmp_path / "b.txt", "two")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app, ["batch", a, b, "--schema", schema_file, "--model", "mock/echo", "-f", "jsonl"]
        )
        assert result.exit_code == 0, result.stdout
        assert len(result.stdout.strip().splitlines()) == 2

    def test_batch_json_array_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        a = _write(tmp_path / "a.txt", "one")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app, ["batch", a, "--schema", schema_file, "--model", "mock/echo", "-f", "json"]
        )
        assert result.exit_code == 0, result.stdout
        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        assert payload[0]["vendor"] == "doc0"

    def test_batch_show_metadata_per_document(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        a = _write(tmp_path / "a.txt", "one")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app,
            ["batch", a, "--schema", schema_file, "--model", "mock/echo", "--show-metadata"],
        )
        assert result.exit_code == 0, result.stdout
        assert "[a.txt]" in result.stderr

    def test_batch_pattern_no_match_is_clean_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        docs = tmp_path / "docs"
        docs.mkdir()
        _write(docs / "a.md", "markdown")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app, ["batch", str(docs), "--schema", schema_file, "--model", "mock/echo"]
        )
        assert result.exit_code != 0

    def test_batch_custom_pattern(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        docs = tmp_path / "docs"
        docs.mkdir()
        _write(docs / "a.md", "markdown")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app,
            [
                "batch",
                str(docs),
                "--schema",
                schema_file,
                "--model",
                "mock/echo",
                "--pattern",
                "*.md",
            ],
        )
        assert result.exit_code == 0, result.stdout

    def test_batch_missing_input_is_clean_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nfield.cli._app.NField", _SpyNField)
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app,
            ["batch", str(tmp_path / "nope.txt"), "--schema", schema_file, "--model", "mock/echo"],
        )
        assert result.exit_code != 0


class TestBuildConfigHelper:
    def test_only_set_flags_are_applied(self):
        cfg = _build_config(ground_values=True)
        default = ExtractionConfig()
        assert cfg.ground_values is True
        # Everything unset matches the library default.
        assert cfg.max_retry_rounds == default.max_retry_rounds
        assert cfg.provenance == default.provenance

    def test_empty_build_equals_default(self):
        assert _build_config() == ExtractionConfig()

    def test_think_budget_pairing_enforced(self):
        with pytest.raises(Exception, match="think-budget"):
            _build_config(think_budget_min=10)

    def test_think_budget_tuple(self):
        cfg = _build_config(think_budget_min=10, think_budget_max=20)
        assert cfg.think_phase_budget == (10, 20)

    def test_false_bool_is_applied_not_dropped(self):
        # inject_dependencies defaults True; an explicit False must survive.
        cfg = _build_config(inject_dependencies=False)
        assert cfg.inject_dependencies is False


class TestParseConfidence:
    def test_none_returns_none(self):
        assert _parse_confidence(None) is None

    def test_parses_pairs(self):
        assert _parse_confidence(["HIGH=0.9", "MED=0.7"]) == {"HIGH": 0.9, "MED": 0.7}

    def test_malformed_pair_raises(self):
        with pytest.raises(Exception, match="TIER=SCORE"):
            _parse_confidence(["HIGH"])

    def test_non_numeric_score_raises(self):
        with pytest.raises(Exception, match="not a number"):
            _parse_confidence(["HIGH=high"])


class TestCollectDocuments:
    def test_directory_expands_sorted(self, tmp_path):
        d = tmp_path / "docs"
        d.mkdir()
        _write(d / "b.txt", "b")
        _write(d / "a.txt", "a")
        files = _collect_documents([d], "*.txt")
        assert [p.name for p in files] == ["a.txt", "b.txt"]

    def test_files_taken_as_is(self, tmp_path):
        a = tmp_path / "a.txt"
        _write(a, "a")
        files = _collect_documents([a], "*.txt")
        assert files == [a]

    def test_missing_input_raises(self, tmp_path):
        with pytest.raises(Exception, match="not found"):
            _collect_documents([tmp_path / "nope.txt"], "*.txt")

    def test_empty_directory_raises(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        with pytest.raises(Exception, match="No files matching"):
            _collect_documents([d], "*.txt")


class TestOutputFormatEnum:
    def test_values(self):
        assert OutputFormat.JSON.value == "json"
        assert OutputFormat.JSONL.value == "jsonl"
        assert OutputFormat.CSV.value == "csv"


class TestIOErrorHandling:
    """Bad file inputs produce a clean message, never a raw Python traceback."""

    def test_non_utf8_schema_is_clean_error(self, tmp_path):
        bad = tmp_path / "s.json"
        bad.write_bytes(b"\xff\xfe\x00not utf8")
        result = runner.invoke(app, ["inspect", str(bad)])
        assert result.exit_code != 0
        assert not isinstance(result.exception, (UnicodeDecodeError, OSError))

    def test_non_utf8_document_is_clean_error(self, tmp_path):
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        doc = tmp_path / "doc.txt"
        doc.write_bytes(b"\xff\xfe\x00not utf8")
        result = runner.invoke(
            app, ["extract", str(doc), "--schema", schema_file, "--model", "mock/echo"]
        )
        assert result.exit_code != 0
        assert not isinstance(result.exception, (UnicodeDecodeError, OSError))

    def test_schema_path_is_directory_is_clean_error(self, tmp_path):
        d = tmp_path / "a_dir"
        d.mkdir()
        result = runner.invoke(app, ["inspect", str(d)])
        assert result.exit_code != 0
        assert not isinstance(result.exception, (UnicodeDecodeError, OSError))

    def test_schema_not_json_object_is_clean_error(self, tmp_path):
        arr = _write(tmp_path / "s.json", "[1, 2, 3]")
        result = runner.invoke(app, ["inspect", arr])
        assert result.exit_code != 0
        assert not isinstance(result.exception, (UnicodeDecodeError, OSError))

    def test_unwritable_output_is_clean_error(self, tmp_path, monkeypatch):
        _install_provider(monkeypatch, "vendor = Acme")
        doc = _write(tmp_path / "doc.txt", "Acme")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        bad_out = tmp_path / "missing_dir" / "out.json"  # parent dir does not exist
        result = runner.invoke(
            app,
            [
                "extract",
                doc,
                "--schema",
                schema_file,
                "--model",
                "mock/echo",
                "--output",
                str(bad_out),
            ],
        )
        assert result.exit_code != 0
        assert not isinstance(result.exception, OSError)
