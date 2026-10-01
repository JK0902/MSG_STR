"""Privacy safeguards for public or non-secure execution."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import PrivacyConfig


def require_text_allowed(config: PrivacyConfig) -> None:
    """Refuse message-text processing unless explicitly enabled."""
    if config.safe_mode and not config.allow_text_input:
        raise RuntimeError(
            "Safe mode is enabled and text input is disabled. Enable text processing "
            "only inside an institutionally approved secure environment."
        )


def assert_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    """Raise a clear error when a required input column is absent."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def sanitize_reason(reason: str, maximum_characters: int = 500) -> str:
    """Return a single-line, length-limited model rationale."""
    compact = " ".join(str(reason).split())
    return compact[:maximum_characters]


def aggregate_log(message: str) -> None:
    """Log only caller-supplied aggregate information."""
    print(message, flush=True)
