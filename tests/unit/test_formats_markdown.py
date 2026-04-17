"""Unit tests for MarkdownFormat — no API keys required."""

from __future__ import annotations

from formatshield.formats.markdown import (
    MarkdownFormat,
    extract_markdown_answer,
    wrap_markdown_prompt,
)


def test_markdown_format_name() -> None:
    fmt = MarkdownFormat()
    assert fmt.name == "markdown"


def test_markdown_wrap_contains_answer_section() -> None:
    fmt = MarkdownFormat()
    result = fmt.wrap_prompt("Write a summary")
    assert "## Answer" in result


def test_markdown_wrap_preserves_prompt() -> None:
    fmt = MarkdownFormat()
    prompt = "Write a summary of the document"
    result = fmt.wrap_prompt(prompt)
    assert prompt in result


def test_markdown_wrap_with_schema_shows_sections() -> None:
    fmt = MarkdownFormat()
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
        },
    }
    result = fmt.wrap_prompt("Analyze this", schema)
    assert "Title" in result or "title" in result.lower()
    assert "Summary" in result or "summary" in result.lower()


def test_markdown_extract_from_answer_section() -> None:
    fmt = MarkdownFormat()
    text = "Some preamble\n\n## Answer\nThis is the answer text."
    result = fmt.extract_answer(text)
    assert "This is the answer text." in result


def test_markdown_extract_from_fence() -> None:
    fmt = MarkdownFormat()
    text = "Here:\n```answer\nfenced content\n```"
    result = fmt.extract_answer(text)
    assert result == "fenced content"


def test_markdown_extract_no_marker() -> None:
    fmt = MarkdownFormat()
    text = "Plain text response with no markers"
    assert fmt.extract_answer(text) == "Plain text response with no markers"


def test_markdown_module_wrap_function() -> None:
    result = wrap_markdown_prompt("Summarize the results")
    assert "Summarize the results" in result
    assert "## Answer" in result


def test_markdown_module_extract_function() -> None:
    text = "## Answer\nThe result is 42."
    result = extract_markdown_answer(text)
    assert "The result is 42." in result


def test_markdown_load_grammar_returns_string() -> None:
    fmt = MarkdownFormat()
    grammar = fmt.load_grammar()
    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_markdown_load_grammar_contains_answer() -> None:
    fmt = MarkdownFormat()
    grammar = fmt.load_grammar()
    assert "Answer" in grammar or "answer" in grammar.lower()


def test_markdown_extract_fence_preferred_over_section() -> None:
    fmt = MarkdownFormat()
    text = "## Answer\nsection answer\n```answer\nfenced answer\n```"
    result = fmt.extract_answer(text)
    # Fence is tried first
    assert result == "fenced answer"


def test_markdown_wrap_schema_limits_to_five_keys() -> None:
    fmt = MarkdownFormat()
    schema = {
        "type": "object",
        "properties": {f"field_{i}": {"type": "string"} for i in range(10)},
    }
    result = fmt.wrap_prompt("Analyze", schema)
    # Should include at most 5 sections (### headings from schema)
    section_count = result.count("###")
    assert section_count <= 5


def test_markdown_extract_trims_whitespace() -> None:
    fmt = MarkdownFormat()
    text = "## Answer\n   trimmed content   \n"
    result = fmt.extract_answer(text)
    assert result == "trimmed content"
