"""Unit tests for formatshield.prompting — Template, Chat, few_shot."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from formatshield.prompting import Chat, Template, few_shot

# ---------------------------------------------------------------------------
# Template.from_string
# ---------------------------------------------------------------------------


def test_template_from_string_renders_basic_variable() -> None:
    tmpl = Template.from_string("Hello {{ name }}!")
    result = tmpl(name="Alice")
    assert result == "Hello Alice!"


def test_template_from_string_filter_name() -> None:
    def my_function() -> None:
        """Do something."""

    tmpl = Template.from_string("{{ fn | name }}")
    assert tmpl(fn=my_function) == "my_function"


def test_template_from_string_filter_description() -> None:
    def my_function() -> None:
        """Extracts entities from text."""

    tmpl = Template.from_string("{{ fn | description }}")
    assert tmpl(fn=my_function) == "Extracts entities from text."


def test_template_from_string_filter_signature() -> None:
    def my_function(x: int, y: str = "default") -> bool: ...

    tmpl = Template.from_string("{{ fn | signature }}")
    result = tmpl(fn=my_function)
    # Parameter names must appear; type annotations may be quoted strings on PEP 563
    assert "x" in result
    assert "y" in result
    assert "default" in result


def test_template_from_string_filter_args() -> None:
    def my_function(a: int, b: str) -> None: ...

    tmpl = Template.from_string("{{ fn | args | join(', ') }}")
    result = tmpl(fn=my_function)
    assert "a" in result
    assert "b" in result


def test_template_from_string_filter_schema_with_dict() -> None:
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    tmpl = Template.from_string("{{ s | schema }}")
    result = tmpl(s=schema)
    assert '"type": "object"' in result
    assert '"name"' in result


def test_template_from_string_filter_schema_with_pydantic_model() -> None:
    class Person(BaseModel):
        name: str
        age: int

    tmpl = Template.from_string("{{ model | schema }}")
    result = tmpl(model=Person)
    assert "name" in result
    assert "age" in result


def test_template_from_string_extra_filters() -> None:
    def upper_filter(s: str) -> str:
        return s.upper()

    tmpl = Template.from_string("{{ text | upper }}", filters={"upper": upper_filter})
    result = tmpl(text="hello")
    assert result == "HELLO"


def test_template_from_string_raises_on_undefined_variable() -> None:
    """StrictUndefined should raise rather than silently produce empty string."""
    import jinja2

    tmpl = Template.from_string("Hello {{ missing }}!")
    with pytest.raises(jinja2.UndefinedError):
        tmpl()


def test_template_from_string_filter_source_returns_code() -> None:
    def simple(x: int) -> int:
        return x + 1

    tmpl = Template.from_string("{{ fn | source }}")
    result = tmpl(fn=simple)
    assert "return x + 1" in result


# ---------------------------------------------------------------------------
# Template.from_file
# ---------------------------------------------------------------------------


def test_template_from_file_reads_and_renders(tmp_path: pytest.TempdirFactory) -> None:
    template_file = tmp_path / "prompt.jinja2"  # type: ignore[operator]
    template_file.write_text("Hello {{ name }}!", encoding="utf-8")

    tmpl = Template.from_file(str(template_file))
    assert tmpl(name="World") == "Hello World!"


def test_template_from_file_raises_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        Template.from_file("/nonexistent/path/template.jinja2")


# ---------------------------------------------------------------------------
# Template.with_examples
# ---------------------------------------------------------------------------


def test_template_with_examples_prepends_blocks() -> None:
    content = "Input: {{ text }}"
    tmpl = Template.from_string(content)
    tmpl_with_ex = tmpl.with_examples(
        [{"input": "cat", "output": "animal"}, {"input": "oak", "output": "plant"}],
        content,
    )
    rendered = tmpl_with_ex(text="rose")
    assert "Input: cat" in rendered
    assert "Output: animal" in rendered
    assert "Input: rose" in rendered


# ---------------------------------------------------------------------------
# Chat — direct message methods
# ---------------------------------------------------------------------------


def test_chat_add_system_message() -> None:
    chat = Chat()
    chat.add_system_message("You are an assistant.")
    assert len(chat) == 1
    assert chat.messages[0] == {"role": "system", "content": "You are an assistant."}


def test_chat_add_user_message() -> None:
    chat = Chat()
    chat.add_user_message("Hello!")
    assert chat.messages[0]["role"] == "user"
    assert chat.messages[0]["content"] == "Hello!"


def test_chat_add_assistant_message() -> None:
    chat = Chat()
    chat.add_assistant_message("Hi there!")
    assert chat.messages[0]["role"] == "assistant"


def test_chat_str_representation() -> None:
    chat = Chat()
    chat.add_system_message("System prompt.")
    chat.add_user_message("User question.")
    result = str(chat)
    assert "[SYSTEM]" in result
    assert "[USER]" in result
    assert "System prompt." in result


def test_chat_len() -> None:
    chat = Chat()
    assert len(chat) == 0
    chat.add_user_message("Hi")
    assert len(chat) == 1


def test_chat_clear() -> None:
    chat = Chat()
    chat.add_user_message("Hi")
    chat.clear()
    assert len(chat) == 0


# ---------------------------------------------------------------------------
# Chat — context manager role helpers
# ---------------------------------------------------------------------------


def test_chat_context_manager_system() -> None:
    chat = Chat()
    with chat.system():
        chat.append("You are a helpful assistant.")
    assert len(chat) == 1
    assert chat.messages[0]["role"] == "system"
    assert "helpful" in chat.messages[0]["content"]


def test_chat_context_manager_user() -> None:
    chat = Chat()
    with chat.user():
        chat.append("Line 1.")
        chat.append("Line 2.")
    assert len(chat) == 1
    assert "Line 1." in chat.messages[0]["content"]
    assert "Line 2." in chat.messages[0]["content"]


def test_chat_context_manager_assistant() -> None:
    chat = Chat()
    with chat.assistant():
        chat.append("Here is my answer.")
    assert chat.messages[0]["role"] == "assistant"


def test_chat_context_manager_multiple_turns() -> None:
    chat = Chat()
    with chat.system():
        chat.append("System prompt.")
    with chat.user():
        chat.append("User query.")
    with chat.assistant():
        chat.append("Response.")
    assert len(chat) == 3
    roles = [m["role"] for m in chat.messages]
    assert roles == ["system", "user", "assistant"]


def test_chat_append_outside_context_raises() -> None:
    chat = Chat()
    with pytest.raises(RuntimeError, match="role context manager"):
        chat.append("This should fail")


def test_chat_nested_context_raises() -> None:
    chat = Chat()
    with pytest.raises(RuntimeError, match="Cannot nest"):
        with chat.user():
            with chat.system():
                chat.append("nested")


def test_chat_context_empty_append_produces_no_message() -> None:
    """A context that has no appends should not add an empty message."""
    chat = Chat()
    with chat.user():
        pass  # no append calls
    assert len(chat) == 0


# ---------------------------------------------------------------------------
# Chat.to_prompt
# ---------------------------------------------------------------------------


def test_chat_to_prompt_flattens_to_string() -> None:
    chat = Chat()
    chat.add_system_message("System.")
    chat.add_user_message("User.")
    result = chat.to_prompt()
    assert "System." in result
    assert "User." in result


# ---------------------------------------------------------------------------
# few_shot
# ---------------------------------------------------------------------------


def test_few_shot_renders_examples() -> None:
    examples = [
        {"input": "I love this!", "output": "positive"},
        {"input": "This is awful.", "output": "negative"},
    ]
    result = few_shot(examples)
    assert "I love this!" in result
    assert "positive" in result
    assert "This is awful." in result
    assert "negative" in result


def test_few_shot_custom_separator() -> None:
    examples = [{"input": "a", "output": "b"}]
    result = few_shot(examples, separator="||")
    assert "||" not in result  # Only one example, no separator between examples


def test_few_shot_custom_keys() -> None:
    examples = [{"question": "What?", "answer": "42"}]
    result = few_shot(examples, input_key="question", output_key="answer")
    assert "What?" in result
    assert "42" in result


def test_few_shot_empty_list() -> None:
    result = few_shot([])
    assert result == ""


def test_template_filter_schema_with_non_basemodel_type() -> None:
    """schema filter with a type that is NOT a BaseModel covers lines 92-97."""
    import dataclasses

    @dataclasses.dataclass
    class Coord:
        x: float
        y: float

    tmpl = Template.from_string("{{ t | schema }}")
    result = tmpl(t=Coord)
    # Should produce some JSON-like output
    assert "{" in result


def test_template_filter_schema_fallback_for_plain_object() -> None:
    """schema filter fallback when obj has no schema method — covers lines 98-105."""
    tmpl = Template.from_string("{{ val | schema }}")
    result = tmpl(val=42)
    assert "object" in result
