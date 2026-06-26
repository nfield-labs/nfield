"""OpenAI-compatible provider implementation.

Exports the OpenAIProvider class for use with the from_model() factory.
"""

from __future__ import annotations

from nfield.providers.openai._provider import OpenAIProvider

__all__ = ["OpenAIProvider"]
