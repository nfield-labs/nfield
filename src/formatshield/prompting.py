"""
FormatShield prompt templating and chat construction.

Provides a Jinja2-backed ``Template`` class with built-in filters for Python
introspection (function names, signatures, docstrings, JSON schema extraction)
and a ``Chat`` dataclass for building multi-turn conversation histories.

Usage::

    from formatshield.prompting import Template, Chat

    # Jinja2 template with introspection filters
    tmpl = Template.from_string(
        "Extract data matching {{ schema | schema }} from: {{ text }}"
    )
    prompt = tmpl(schema=MyModel, text="Alice is 30 years old.")

    # Chat with role context managers
    chat = Chat()
    with chat.system():
        chat.append("You are a helpful extraction assistant.")
    with chat.user():
        chat.append(prompt)

    # Convert to string for a backend that accepts plain text:
    print(str(chat))

    # Or pass messages directly to backends that accept dicts:
    messages = chat.messages
"""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Jinja2 filters
# ---------------------------------------------------------------------------


def _filter_name(fn: Any) -> str:
    """Return the ``__name__`` of a callable."""
    return fn.__name__


def _filter_description(fn: Any) -> str:
    """Return the first line of a callable's docstring, or empty string."""
    return (inspect.getdoc(fn) or "").split("\n")[0]


def _filter_source(fn: Any) -> str:
    """Return the full source code of a callable."""
    return inspect.getsource(fn)


def _filter_signature(fn: Any) -> str:
    """Return the call signature of a callable as a string."""
    return str(inspect.signature(fn))


def _filter_args(fn: Any) -> list[str]:
    """Return the parameter names of a callable as a list."""
    return list(inspect.signature(fn).parameters.keys())


def _filter_schema(obj: Any) -> str:
    """Return a pretty-printed JSON schema string for *obj*.

    Accepts:
    - A plain ``dict`` — serialized as-is.
    - A Pydantic ``BaseModel`` subclass — extracted via ``model_json_schema()``.
    - Any other type — attempted via ``pydantic.TypeAdapter``.
    - Fallback: ``{"type": "object"}``
    """
    if isinstance(obj, dict):
        return json.dumps(obj, indent=2)
    if isinstance(obj, type):
        # Pydantic BaseModel
        schema_fn = getattr(obj, "model_json_schema", None)
        if schema_fn is not None:
            return json.dumps(schema_fn(), indent=2)
        # Try TypeAdapter for other annotated types
        try:
            import pydantic

            return json.dumps(pydantic.TypeAdapter(obj).json_schema(), indent=2)
        except Exception:
            logger.debug("TypeAdapter schema extraction failed for %r", obj)
    try:
        # Plain instance with a model_json_schema classmethod
        obj_schema_fn = getattr(type(obj), "model_json_schema", None)
        if obj_schema_fn is not None:
            return json.dumps(obj_schema_fn(), indent=2)
    except Exception:
        logger.debug("model_json_schema extraction failed for %r", obj)
    return json.dumps({"type": "object"}, indent=2)


_DEFAULT_FILTERS: dict[str, Any] = {
    "name": _filter_name,
    "description": _filter_description,
    "source": _filter_source,
    "signature": _filter_signature,
    "args": _filter_args,
    "schema": _filter_schema,
}


def _create_jinja_env(extra_filters: dict[str, Any] | None = None) -> Any:
    """Build a Jinja2 ``Environment`` with FormatShield's default filters.

    Args:
        extra_filters: Additional user-supplied Jinja2 filters to register.

    Returns:
        A ``jinja2.Environment`` with ``undefined=StrictUndefined`` to surface
        template typos as errors rather than silent empty strings.

    Raises:
        ImportError: If Jinja2 is not installed.
    """
    try:
        import jinja2
    except ImportError as exc:
        raise ImportError(
            "Jinja2 is required for FormatShield templates. Install with: pip install jinja2"
        ) from exc

    # autoescape=False intentional: FormatShield prompts are plain text, not HTML/XML.
    # XSS is not a concern — outputs go to LLM APIs, never to a browser.
    env = jinja2.Environment(  # nosec B701
        undefined=jinja2.StrictUndefined,
        autoescape=False,
    )
    env.filters.update(_DEFAULT_FILTERS)
    if extra_filters:
        env.filters.update(extra_filters)
    return env


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


class Template:
    """Jinja2-backed prompt template with Python introspection filters.

    Filters available inside template strings:

    - ``{{ fn | name }}``        → function name string
    - ``{{ fn | description }}`` → first line of docstring
    - ``{{ fn | source }}``      → full source code
    - ``{{ fn | signature }}``   → call signature string
    - ``{{ fn | args }}``        → list of parameter names
    - ``{{ schema | schema }}``  → pretty JSON schema (dict, Pydantic model, TypeAdapter)

    Args:
        _template: A compiled Jinja2 template object.

    Example::

        tmpl = Template.from_string("Hello {{ name }}! Extract: {{ schema | schema }}")
        prompt = tmpl(name="Alice", schema=MyModel)
    """

    def __init__(self, _template: Any) -> None:
        self._template = _template

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        """Render the template.

        Args:
            *args: Positional arguments — passed as positional context (not common).
            **kwargs: Template variable bindings.

        Returns:
            Rendered string.

        Raises:
            jinja2.UndefinedError: If a variable used in the template is not provided.
        """
        return self._template.render(*args, **kwargs)

    @classmethod
    def from_string(
        cls,
        content: str,
        filters: dict[str, Any] | None = None,
    ) -> Template:
        """Build a Template from an inline string.

        Args:
            content: Jinja2 template string.
            filters: Extra Jinja2 filters to register in addition to defaults.

        Returns:
            :class:`Template` instance.

        Example::

            tmpl = Template.from_string("Classify: {{ text }} → {{ schema | schema }}")
        """
        env = _create_jinja_env(filters)
        return cls(env.from_string(content))

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        filters: dict[str, Any] | None = None,
        encoding: str = "utf-8",
    ) -> Template:
        """Build a Template from a file on disk.

        Args:
            path: Path to a ``.jinja2`` or ``.txt`` template file.
            filters: Extra Jinja2 filters to register in addition to defaults.
            encoding: File encoding (default UTF-8).

        Returns:
            :class:`Template` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.

        Example::

            tmpl = Template.from_file("prompts/extract.jinja2")
            prompt = tmpl(text="Alice is 30.")
        """
        content = Path(path).read_text(encoding=encoding)
        return cls.from_string(content, filters=filters)

    def with_examples(
        self,
        examples: list[dict[str, str]],
        template_content: str,
    ) -> Template:
        """Prepend few-shot examples to a new template string.

        Each example dict must have ``"input"`` and ``"output"`` keys.
        Because compiled Jinja2 templates do not expose their source string,
        you must pass the original template content alongside the examples.

        Args:
            examples: List of ``{"input": ..., "output": ...}`` dicts.
            template_content: The Jinja2 template string to append after examples.

        Returns:
            A new :class:`Template` that prepends the examples.

        Example::

            content = "Input: {{ text }}"
            tmpl = Template.from_string(content).with_examples(
                [{"input": "foo", "output": "bar"}],
                content,
            )
        """
        examples_block = "\n".join(
            f"Input: {ex['input']}\nOutput: {ex['output']}" for ex in examples
        )
        new_content = f"{examples_block}\n\n{template_content}"
        return Template.from_string(new_content)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@dataclass
class Chat:
    """Multi-turn conversation builder.

    Provides both direct message methods and context manager role helpers.
    Use :meth:`messages` to access the list of role/content dicts for passing
    to backends that accept chat-style message arrays.

    Example::

        chat = Chat()
        chat.add_system_message("You are a helpful assistant.")
        chat.add_user_message("What is the capital of France?")
        print(str(chat))
        # [SYSTEM]: You are a helpful assistant.
        # [USER]: What is the capital of France?
    """

    messages: list[dict[str, str]] = field(default_factory=list)
    _active_role: str | None = field(default=None, init=False, repr=False)
    _active_buffer: list[str] = field(default_factory=list, init=False, repr=False)

    # ------------------------------------------------------------------
    # Direct append methods
    # ------------------------------------------------------------------

    def add_system_message(self, content: str) -> None:
        """Append a system-role message.

        Args:
            content: Message text.
        """
        self.messages.append({"role": "system", "content": content})

    def add_user_message(self, content: str) -> None:
        """Append a user-role message.

        Args:
            content: Message text.
        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant-role message.

        Args:
            content: Message text.
        """
        self.messages.append({"role": "assistant", "content": content})

    # ------------------------------------------------------------------
    # Context manager role helpers
    # ------------------------------------------------------------------

    @contextmanager
    def system(self) -> Generator[None, None, None]:
        """Context manager — subsequent :meth:`append` calls become system messages.

        Example::

            with chat.system():
                chat.append("You are a helpful assistant.")
        """
        yield from self._role_context("system")

    @contextmanager
    def user(self) -> Generator[None, None, None]:
        """Context manager — subsequent :meth:`append` calls become user messages.

        Example::

            with chat.user():
                chat.append("Classify this text.")
                chat.append(prompt)
        """
        yield from self._role_context("user")

    @contextmanager
    def assistant(self) -> Generator[None, None, None]:
        """Context manager — subsequent :meth:`append` calls become assistant messages.

        Example::

            with chat.assistant():
                chat.append("Here is my analysis:")
        """
        yield from self._role_context("assistant")

    def _role_context(self, role: str) -> Generator[None, None, None]:
        """Internal: set active role, collect appends, commit on exit."""
        if self._active_role is not None:
            raise RuntimeError(
                f"Cannot nest role context managers. Already inside '{self._active_role}' block."
            )
        self._active_role = role
        self._active_buffer = []
        try:
            yield
        finally:
            combined = "\n".join(self._active_buffer)
            if combined:
                self.messages.append({"role": role, "content": combined})
            self._active_role = None
            self._active_buffer = []

    def append(self, content: str) -> None:
        """Append text inside the active role context, or raise if called outside one.

        Args:
            content: Text to add to the current role's message buffer.

        Raises:
            RuntimeError: If called outside a role context manager.
        """
        if self._active_role is None:
            raise RuntimeError(
                "Chat.append() must be called inside a role context manager: "
                "with chat.system(): / with chat.user(): / with chat.assistant():"
            )
        self._active_buffer.append(content)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.messages)

    def __str__(self) -> str:
        return "\n".join(f"[{m['role'].upper()}]: {m['content']}" for m in self.messages)

    def __repr__(self) -> str:
        return f"Chat(messages={len(self.messages)})"

    def to_prompt(self, separator: str = "\n\n") -> str:
        """Flatten all messages into a single prompt string.

        Useful for backends that accept plain text rather than message arrays.

        Args:
            separator: String placed between each message block.

        Returns:
            Plain text string with all messages concatenated.
        """
        return separator.join(f"{m['role'].capitalize()}: {m['content']}" for m in self.messages)

    def clear(self) -> None:
        """Remove all messages from the chat history."""
        self.messages.clear()


# ---------------------------------------------------------------------------
# Few-shot helper
# ---------------------------------------------------------------------------


def few_shot(
    examples: list[dict[str, str]],
    input_key: str = "input",
    output_key: str = "output",
    separator: str = "\n---\n",
) -> str:
    """Render a list of few-shot examples as a formatted string block.

    Args:
        examples: List of dicts with ``input_key`` and ``output_key`` fields.
        input_key: Key for the example input (default ``"input"``).
        output_key: Key for the example output (default ``"output"``).
        separator: String placed between examples (default ``"\\n---\\n"``).

    Returns:
        Formatted multi-line string ready to insert into a prompt.

    Example::

        examples = [
            {"input": "I love Paris", "output": "positive"},
            {"input": "This is terrible", "output": "negative"},
        ]
        block = few_shot(examples)
    """
    parts = []
    for ex in examples:
        parts.append(f"Input: {ex[input_key]}\nOutput: {ex[output_key]}")
    return separator.join(parts)
