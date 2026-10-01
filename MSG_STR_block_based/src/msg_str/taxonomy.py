"""Manuscript block: iterative LLM-guided symptom taxonomy."""

from __future__ import annotations

import re
from typing import List, Tuple

import pandas as pd

from .privacy import assert_columns


UPDATE_SYSTEM = """
You are updating a 3-level clinical taxonomy for de-identified clinical
portal messages: MAIN -> SUB1 -> SUB2.

You must:
- Keep existing categories whenever possible.
- Add new MAIN/SUB1/SUB2 categories only to capture clinically novel topics.
- Do not merge or remove categories unless they are nearly identical.
- Keep the taxonomy interpretable and moderately granular.
- Use additional granularity for symptom topics.
- In the reasoning line, describe only categories added, merged, or renamed.

Return plain text.
""".strip()

STOP_WORDS = {
    "and",
    "or",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "a",
    "an",
    "with",
    "without",
}


def normalize_text(value: object) -> str:
    """Normalize whitespace without otherwise changing the text."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def phrase_to_keywords(phrase: object) -> List[str]:
    """Convert an ontology phrase to unique lowercase seed keywords."""
    tokens = re.findall(r"[a-z]+", normalize_text(phrase).lower())
    return [token for token in tokens if token not in STOP_WORDS and len(token) > 2]


def build_seed_topic_list(
    seed_frame: pd.DataFrame,
    main_column: str = "main",
    sub1_column: str = "sub1",
) -> Tuple[List[str], List[List[str]], pd.DataFrame]:
    """Build one guided-BERTopic keyword list per MAIN ontology label."""
    if seed_frame.empty:
        raise ValueError("seed_frame is empty.")
    assert_columns(seed_frame, [main_column, sub1_column], "seed_frame")

    deduplicated = seed_frame[[main_column, sub1_column]].copy()
    deduplicated[main_column] = deduplicated[main_column].map(normalize_text)
    deduplicated[sub1_column] = deduplicated[sub1_column].map(normalize_text)
    deduplicated = deduplicated.drop_duplicates([main_column, sub1_column])

    labels: List[str] = []
    seed_topics: List[List[str]] = []
    for main_label, sub1_values in deduplicated.groupby(main_column)[sub1_column]:
        words = phrase_to_keywords(main_label)
        for sub1_value in sub1_values:
            words.extend(phrase_to_keywords(sub1_value))
        labels.append(main_label)
        seed_topics.append(list(dict.fromkeys(words)))

    return labels, seed_topics, deduplicated
