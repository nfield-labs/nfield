"""Native Anthropic (Claude) provider.

Exports the AnthropicProvider class for use with the from_model() factory.
"""

from __future__ import annotations

from nfield.providers.anthropic._provider import AnthropicProvider

__all__ = ["AnthropicProvider"]
