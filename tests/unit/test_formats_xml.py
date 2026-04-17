"""Unit tests for XMLFormat — no API keys required."""

from __future__ import annotations

from formatshield.formats.xml import XMLFormat, extract_xml_answer, wrap_xml_prompt


def test_xml_format_name() -> None:
    fmt = XMLFormat()
    assert fmt.name == "xml"


def test_xml_wrap_prompt_contains_tags() -> None:
    fmt = XMLFormat()
    result = fmt.wrap_prompt("Tell me something")
    assert "<answer>" in result
    assert "</answer>" in result


def test_xml_wrap_prompt_contains_original() -> None:
    fmt = XMLFormat()
    prompt = "Extract the name from the document"
    result = fmt.wrap_prompt(prompt)
    assert prompt in result


def test_xml_wrap_with_schema_shows_structure() -> None:
    fmt = XMLFormat()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    result = fmt.wrap_prompt("Extract fields", schema)
    assert "<name>" in result
    assert "<age>" in result


def test_xml_extract_with_tags() -> None:
    fmt = XMLFormat()
    text = "Here is my answer: <answer>Paris</answer>"
    assert fmt.extract_answer(text) == "Paris"


def test_xml_extract_without_tags() -> None:
    fmt = XMLFormat()
    text = "The answer is Paris"
    assert fmt.extract_answer(text) == "The answer is Paris"


def test_xml_extract_multiline() -> None:
    fmt = XMLFormat()
    text = "<answer>\nLine one\nLine two\n</answer>"
    result = fmt.extract_answer(text)
    assert "Line one" in result
    assert "Line two" in result


def test_xml_extract_trims_whitespace() -> None:
    fmt = XMLFormat()
    text = "<answer>  spaced content  </answer>"
    assert fmt.extract_answer(text) == "spaced content"


def test_xml_module_wrap_function() -> None:
    result = wrap_xml_prompt("Hello world")
    assert "<answer>" in result
    assert "Hello world" in result


def test_xml_module_extract_function() -> None:
    text = "<answer>42</answer>"
    assert extract_xml_answer(text) == "42"


def test_xml_load_grammar_returns_string() -> None:
    fmt = XMLFormat()
    grammar = fmt.load_grammar()
    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_xml_load_grammar_contains_answer_tag() -> None:
    fmt = XMLFormat()
    grammar = fmt.load_grammar()
    assert "answer" in grammar


def test_xml_wrap_no_schema_no_structure_hint() -> None:
    fmt = XMLFormat()
    result = fmt.wrap_prompt("Simple prompt", schema=None)
    # Should not contain example_tags structure when no schema
    assert "Expected XML structure" not in result


def test_xml_wrap_schema_limits_to_five_keys() -> None:
    fmt = XMLFormat()
    schema = {
        "type": "object",
        "properties": {f"key{i}": {"type": "string"} for i in range(10)},
    }
    result = fmt.wrap_prompt("Extract", schema)
    # Should only show up to 5 keys
    shown = result.count("<key")
    assert shown <= 5
