"""Error types raised by the wake-word TTS core.

I. Design intent

1. Every error carries a short machine-facing ``message`` and an optional
   human-facing ``hint``.
2. ``hint`` is what the plugin shows in the chat window, so it must never
   contain the API key or other secrets.
3. Grouping the errors here keeps ``main.py`` free of stringly-typed branching.

@author zexuan.peng <pengzexuan2001@gmail.com>
"""

from __future__ import annotations


class TtsError(Exception):
    """Base class for every expected failure of this plugin."""

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def user_message(self) -> str:
        """Render the message that is safe to send back to the chat."""
        return self.hint or self.message


class TtsConfigError(TtsError):
    """Raised when the plugin configuration cannot produce a valid request."""


class TtsTextError(TtsError):
    """Raised when the incoming text is empty or unusable after sanitizing."""


class TtsNetworkError(TtsError):
    """Raised when the HTTP layer failed after all retries were exhausted."""


class LlmError(TtsError):
    """Raised when the pre-synthesis LLM stage could not produce a script.

    The pipeline treats this as non-fatal when ``llm_fail_open`` is enabled, so
    the message still reaches the TTS model with its original text.
    """


class TtsApiError(TtsError):
    """Raised when the TTS endpoint answered with a business-level error."""

    def __init__(
        self,
        message: str,
        *,
        hint: str = "",
        status_code: int | None = None,
        code: str = "",
        request_id: str = "",
    ) -> None:
        super().__init__(message, hint=hint)
        self.status_code = status_code
        self.code = code
        self.request_id = request_id
