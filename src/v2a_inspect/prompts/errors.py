from __future__ import annotations


class PromptNotFoundError(KeyError):
    """Raised when a named prompt pair does not exist."""


class PromptRenderError(ValueError):
    """Raised when a prompt pair cannot be rendered."""
