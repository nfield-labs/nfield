"""Native Google Gemini provider (google-genai SDK).

Exports the GeminiProvider class for use with the from_model() factory.
"""

from __future__ import annotations

from nfield.providers.gemini._provider import GeminiProvider

__all__ = ["GeminiProvider"]
