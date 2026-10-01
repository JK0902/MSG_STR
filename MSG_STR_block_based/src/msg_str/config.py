"""Typed configuration objects used across the analysis blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class PrivacyConfig:
    """Controls whether code may process message text."""

    safe_mode: bool = True
    allow_text_input: bool = False
    verbose: bool = False


@dataclass(frozen=True)
class TopicModelConfig:
    """Configuration for guided BERTopic validation."""

    id_column: str = "message_id"
    text_column: str = "message_text"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 128
    n_neighbors: int = 30
    n_components: int = 10
    min_cluster_size: int = 30
    min_samples: int = 10
    random_state: int = 42


@dataclass(frozen=True)
class ScreeningConfig:
    """Configuration for one screening-window analysis."""

    window_days: int = 3
    assumed_prevalence: float = 0.10
    minimum_specificity: float = 0.90
    test_size: float = 0.30
    random_state: int = 42
    thresholds: Tuple[float, ...] = field(
        default_factory=lambda: tuple(i / 100 for i in range(101))
    )

    def __post_init__(self) -> None:
        if self.window_days <= 0:
            raise ValueError("window_days must be positive.")
        if not 0 < self.assumed_prevalence < 1:
            raise ValueError("assumed_prevalence must be between 0 and 1.")
        if not 0 < self.minimum_specificity <= 1:
            raise ValueError("minimum_specificity must be in (0, 1].")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
