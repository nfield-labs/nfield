"""CLI tests: `inspect` (offline) and `extract` (mock provider) via CliRunner."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from formatshield.cli._app import app

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
    context_window = 8192
    max_output_tokens = 8192
    model_name = "mock/echo"

    def __init__(self, sfep_text: str) -> None:
        self._sfep = sfep_text
        self.last_messages: list[dict[str, str]] = []

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.last_messages = messages
        return self._sfep

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _write(path, data: str) -> str:
    path.write_text(data, encoding="utf-8")
    return str(path)


class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "formatshield" in result.stdout


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


class TestExtract:
    def test_extract_writes_json(self, tmp_path, monkeypatch):
        provider = _MockProvider("vendor = Acme\ntotal = 42.5\npaid = true")
        monkeypatch.setattr("formatshield.engine._async.from_model", lambda _m, **_kw: provider)

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
        provider = _MockProvider("vendor = Globex")
        monkeypatch.setattr("formatshield.engine._async.from_model", lambda _m, **_kw: provider)
        doc = _write(tmp_path / "doc.txt", "Globex")
        schema_file = _write(tmp_path / "s.json", json.dumps(_SCHEMA))
        result = runner.invoke(
            app, ["extract", doc, "--schema", schema_file, "--model", "mock/echo"]
        )
        assert result.exit_code == 0
        assert "Globex" in result.stdout

    def test_extract_with_prompt_flags(self, tmp_path, monkeypatch):
        provider = _MockProvider("vendor = Acme")
        monkeypatch.setattr("formatshield.engine._async.from_model", lambda _m, **_kw: provider)
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
                "--system-prompt",
                "DOMAIN: invoices.",
                "--user-prompt",
                "Be exact.",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "DOMAIN: invoices." in provider.last_messages[0]["content"]
        assert "Be exact." in provider.last_messages[1]["content"]

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


class TestIOErrorHandling:
    """Bad file inputs produce a clean message, never a raw Python traceback."""

    def test_non_utf8_schema_is_clean_error(self, tmp_path):
        bad = tmp_path / "s.json"
        bad.write_bytes(b"\xff\xfe\x00not utf8")
        result = runner.invoke(app, ["inspect", str(bad)])
        assert result.exit_code != 0
        # The fix turns the read failure into BadParameter; without it the raw
        # UnicodeDecodeError would propagate as result.exception.
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
        provider = _MockProvider("vendor = Acme")
        monkeypatch.setattr("formatshield.engine._async.from_model", lambda _m, **_kw: provider)
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
