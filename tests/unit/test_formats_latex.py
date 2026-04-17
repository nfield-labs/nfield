"""Unit tests for LaTeXFormat — no API keys required."""

from __future__ import annotations

from formatshield.formats.latex import LaTeXFormat, extract_latex_answer, wrap_latex_prompt


def test_latex_format_name() -> None:
    fmt = LaTeXFormat()
    assert fmt.name == "latex"


def test_latex_wrap_contains_boxed_instruction() -> None:
    fmt = LaTeXFormat()
    result = fmt.wrap_prompt("Solve: x + 1 = 5")
    assert "\\boxed{}" in result or "\\boxed" in result


def test_latex_wrap_preserves_prompt() -> None:
    fmt = LaTeXFormat()
    prompt = "Solve: x + 1 = 5"
    result = fmt.wrap_prompt(prompt)
    assert prompt in result


def test_latex_extract_boxed() -> None:
    fmt = LaTeXFormat()
    text = "The answer is \\boxed{42}"
    assert fmt.extract_answer(text) == "42"


def test_latex_extract_boxed_expression() -> None:
    fmt = LaTeXFormat()
    text = "We get \\boxed{x^2 + 1}"
    assert fmt.extract_answer(text) == "x^2 + 1"


def test_latex_extract_answer_env() -> None:
    fmt = LaTeXFormat()
    text = "\\begin{answer}\n-1, -2\n\\end{answer}"
    result = fmt.extract_answer(text)
    assert "-1, -2" in result


def test_latex_extract_no_marker() -> None:
    fmt = LaTeXFormat()
    text = "The answer is 42"
    assert fmt.extract_answer(text) == "The answer is 42"


def test_latex_module_wrap_function() -> None:
    result = wrap_latex_prompt("Compute the integral")
    assert "Compute the integral" in result
    assert "\\boxed" in result


def test_latex_module_extract_function() -> None:
    text = "Therefore \\boxed{\\frac{1}{2}}"
    result = extract_latex_answer(text)
    assert "\\frac{1}{2}" in result


def test_latex_load_grammar_returns_string() -> None:
    fmt = LaTeXFormat()
    grammar = fmt.load_grammar()
    assert isinstance(grammar, str)
    assert len(grammar) > 0


def test_latex_load_grammar_contains_boxed() -> None:
    fmt = LaTeXFormat()
    grammar = fmt.load_grammar()
    assert "boxed" in grammar


def test_latex_wrap_schema_ignored() -> None:
    fmt = LaTeXFormat()
    schema = {"type": "object", "properties": {"answer": {"type": "number"}}}
    result = fmt.wrap_prompt("Solve for x", schema)
    # Schema is ignored in LaTeX mode — just prompt + boxed instruction
    assert "Solve for x" in result
    assert "\\boxed" in result


def test_latex_extract_boxed_with_nested_braces() -> None:
    fmt = LaTeXFormat()
    text = "Result: \\boxed{\\frac{3}{4}}"
    result = fmt.extract_answer(text)
    assert "\\frac{3}{4}" in result


def test_latex_extract_prefers_boxed_over_env() -> None:
    fmt = LaTeXFormat()
    # Both markers present — \\boxed{} takes priority
    text = "\\begin{answer}env answer\\end{answer} and \\boxed{boxed answer}"
    result = fmt.extract_answer(text)
    assert result == "boxed answer"
