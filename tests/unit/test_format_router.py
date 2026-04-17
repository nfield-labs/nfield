"""Unit tests for FormatRouter — no API keys required."""

from __future__ import annotations

from formatshield.formats.router import FormatRouter, OutputFormat


def test_format_router_math_returns_latex() -> None:
    router = FormatRouter()
    result = router.predict("Solve: x^2 + 4 = 0")
    assert result == OutputFormat.LATEX


def test_format_router_calc_returns_latex() -> None:
    router = FormatRouter()
    result = router.predict("Calculate the integral of sin(x) dx")
    assert result == OutputFormat.LATEX


def test_format_router_extraction_returns_xml() -> None:
    router = FormatRouter()
    result = router.predict("Extract all named entities from the document")
    assert result == OutputFormat.XML


def test_format_router_ner_returns_xml() -> None:
    router = FormatRouter()
    result = router.predict("Identify the entities in this text")
    assert result == OutputFormat.XML


def test_format_router_report_returns_markdown() -> None:
    router = FormatRouter()
    result = router.predict("Summarize and explain the quarterly results in detail")
    assert result == OutputFormat.MARKDOWN


def test_format_router_schema_with_properties_returns_json() -> None:
    router = FormatRouter()
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    result = router.predict("Get user info", schema)
    assert result == OutputFormat.JSON


def test_format_router_structured_keyword_returns_json() -> None:
    router = FormatRouter()
    result = router.predict("output as structured json schema object")
    assert result == OutputFormat.JSON


def test_format_router_default_returns_json() -> None:
    router = FormatRouter()
    result = router.predict("")
    assert result == OutputFormat.JSON


def test_format_router_output_format_enum_values() -> None:
    values = {f.value for f in OutputFormat}
    assert "json" in values
    assert "xml" in values
    assert "latex" in values
    assert "markdown" in values
    assert len(values) == 4


def test_format_router_predict_with_confidence_returns_tuple() -> None:
    router = FormatRouter()
    result = router.predict_with_confidence("Solve the equation x + 3 = 7")
    assert isinstance(result, tuple)
    assert len(result) == 2
    fmt, conf = result
    assert isinstance(fmt, OutputFormat)
    assert isinstance(conf, float)


def test_format_router_confidence_in_range() -> None:
    router = FormatRouter()
    for prompt in [
        "Solve: x^2 = 4",
        "Extract names",
        "Summarize the report",
        "Return as JSON",
        "",
    ]:
        _, conf = router.predict_with_confidence(prompt)
        assert 0.0 <= conf <= 1.0, f"Confidence out of range for prompt: {prompt!r}"


def test_format_router_math_confidence_high() -> None:
    router = FormatRouter()
    fmt, conf = router.predict_with_confidence(
        "Solve and compute the derivative of sin(x) + cos(x)"
    )
    assert fmt == OutputFormat.LATEX
    assert conf > 0.0


def test_format_router_schema_confidence_high() -> None:
    router = FormatRouter()
    schema = {"type": "object", "properties": {"result": {"type": "string"}}}
    fmt, conf = router.predict_with_confidence("Generate output", schema)
    assert fmt == OutputFormat.JSON
    assert conf >= 0.9


def test_format_router_predict_deterministic() -> None:
    router = FormatRouter()
    prompt = "Solve: 3x + 7 = 22"
    results = [router.predict(prompt) for _ in range(5)]
    assert len({r.value for r in results}) == 1


def test_format_router_keyword_scoring_monotonic() -> None:
    router = FormatRouter()
    # Single math keyword
    score_one = router._score_keywords("solve the problem", frozenset({"solve"}))
    # Two math keywords
    score_two = router._score_keywords(
        "solve and calculate the problem", frozenset({"solve", "calculate"})
    )
    assert score_two > score_one


def test_format_router_array_schema_returns_json() -> None:
    router = FormatRouter()
    schema = {"type": "array", "items": {"type": "string"}}
    result = router.predict("List items", schema)
    assert result == OutputFormat.JSON


def test_format_router_no_schema_math_beats_extraction() -> None:
    router = FormatRouter()
    # Math-heavy prompt with one extraction keyword — math should still win
    result = router.predict("Solve the equation and compute the integral derivative theorem")
    assert result == OutputFormat.LATEX


def test_format_router_all_formats_accessible_via_enum() -> None:
    assert OutputFormat.JSON.value == "json"
    assert OutputFormat.XML.value == "xml"
    assert OutputFormat.LATEX.value == "latex"
    assert OutputFormat.MARKDOWN.value == "markdown"
