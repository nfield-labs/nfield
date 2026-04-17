"""Unit tests for FormatShieldNode (LangGraph) — no API keys required."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.integrations.langgraph import FormatShieldNode


def _make_node(**kwargs: Any) -> FormatShieldNode:
    """Return a FormatShieldNode instance backed by DryRunBackend."""
    from formatshield.core import FormatShield

    node = FormatShieldNode.__new__(FormatShieldNode)
    node._shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    node.model = "dryrun/test"
    node._schema = kwargs.get("schema", None)
    node._prompt_key = kwargs.get("prompt_key", "prompt")
    node._output_key = kwargs.get("output_key", "response")
    return node


def test_langgraph_node_model_attribute() -> None:
    """model attribute is stored on the node instance."""
    node = _make_node()
    assert node.model == "dryrun/test"


def test_langgraph_node_schema_stored() -> None:
    """_schema attribute is stored when provided."""
    schema: dict[str, Any] = {"type": "object"}
    node = _make_node(schema=schema)
    assert node._schema is schema


def test_langgraph_node_prompt_key_default() -> None:
    """Default prompt_key is 'prompt'."""
    sig = inspect.signature(FormatShieldNode.__init__)
    assert sig.parameters["prompt_key"].default == "prompt"


def test_langgraph_node_output_key_default() -> None:
    """Default output_key is 'response'."""
    sig = inspect.signature(FormatShieldNode.__init__)
    assert sig.parameters["output_key"].default == "response"


def test_langgraph_node_is_callable() -> None:
    """FormatShieldNode implements __call__."""
    assert callable(FormatShieldNode)
    node = _make_node()
    assert callable(node)


def test_langgraph_node_has_ainvoke() -> None:
    """FormatShieldNode exposes an async ainvoke() method."""
    assert hasattr(FormatShieldNode, "ainvoke")
    assert inspect.iscoroutinefunction(FormatShieldNode.ainvoke)


def test_langgraph_node_has_invoke() -> None:
    """FormatShieldNode exposes a synchronous invoke() method."""
    assert hasattr(FormatShieldNode, "invoke")
    assert callable(FormatShieldNode.invoke)


def test_langgraph_node_call_returns_dict() -> None:
    """__call__ returns a dict containing the output_key."""
    node = _make_node()
    state: dict[str, Any] = {"prompt": "What is 2+2?"}
    result = node(state)
    assert isinstance(result, dict)
    assert "response" in result


def test_langgraph_node_custom_keys() -> None:
    """Node respects custom prompt_key and output_key."""
    node = _make_node(prompt_key="question", output_key="answer")
    state: dict[str, Any] = {"question": "Explain gravity."}
    result = node(state)
    assert "answer" in result
    assert isinstance(result["answer"], str)


def test_langgraph_node_state_preserved() -> None:
    """Other keys in the input state are preserved in the output state."""
    node = _make_node()
    state: dict[str, Any] = {
        "prompt": "What is the capital of France?",
        "session_id": "abc-123",
        "step": 3,
    }
    result = node(state)
    assert result["session_id"] == "abc-123"
    assert result["step"] == 3
    assert "response" in result


def test_langgraph_node_invoke_alias() -> None:
    """invoke() produces the same result as __call__."""
    node = _make_node()
    state: dict[str, Any] = {"prompt": "Hello"}
    assert node.invoke(state) == node(state)


def test_langgraph_node_missing_prompt_key_uses_empty_string() -> None:
    """Node handles missing prompt_key gracefully (empty string prompt)."""
    node = _make_node()
    state: dict[str, Any] = {"other_key": "value"}
    result = node(state)
    assert isinstance(result, dict)
    assert "response" in result


def test_langgraph_node_default_model_string() -> None:
    """Default model string is the expected Groq model."""
    sig = inspect.signature(FormatShieldNode.__init__)
    assert sig.parameters["model"].default == "groq/llama-3.3-70b-versatile"


def test_langgraph_node_init_direct() -> None:
    """Calling __init__ directly covers lines 65-71."""
    backend = DryRunBackend()
    node = FormatShieldNode(model="dryrun/test", backend=backend)
    assert node.model == "dryrun/test"
    assert node._prompt_key == "prompt"
    assert node._output_key == "response"
    assert node._schema is None


def test_langgraph_node_init_custom_keys() -> None:
    """__init__ stores custom prompt_key and output_key."""
    backend = DryRunBackend()
    node = FormatShieldNode(
        model="dryrun/test",
        backend=backend,
        prompt_key="question",
        output_key="answer",
    )
    assert node._prompt_key == "question"
    assert node._output_key == "answer"


def test_langgraph_node_ainvoke_async() -> None:
    """ainvoke() returns a dict with the output key — covers lines 97-99."""
    node = _make_node()

    async def _run() -> dict[str, Any]:
        return await node.ainvoke({"prompt": "What is 2+2?"})

    result = asyncio.run(_run())
    assert isinstance(result, dict)
    assert "response" in result
    assert isinstance(result["response"], str)
