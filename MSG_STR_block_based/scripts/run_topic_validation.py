#!/usr/bin/env python3
"""Run privacy-guarded BERTopic taxonomy validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from msg_str.bertopic_validation import run_bertopic_validation
from msg_str.config import PrivacyConfig, TopicModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages-csv", type=Path, required=True)
    parser.add_argument("--taxonomy-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/topic_validation"))
    parser.add_argument("--id-column", default="message_id")
    parser.add_argument("--text-column", default="message_text")
    parser.add_argument(
        "--allow-text-input",
        action="store_true",
        help="Use only in an institutionally approved secure environment.",
    )
    args = parser.parse_args()

    privacy = PrivacyConfig(
        safe_mode=True,
        allow_text_input=args.allow_text_input,
    )
    topic = TopicModelConfig(
        id_column=args.id_column,
        text_column=args.text_column,
    )
    artifacts = run_bertopic_validation(
        pd.read_csv(args.messages_csv),
        pd.read_csv(args.taxonomy_csv),
        topic,
        privacy,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts["topic_info"].to_csv(args.out_dir / "topic_info.csv", index=False)
    artifacts["aligned_topics"].to_csv(args.out_dir / "aligned_topics.csv", index=False)
    pd.DataFrame(
        [{"n_docs": artifacts["n_docs"], "n_topics": artifacts["n_topics"]}]
    ).to_csv(args.out_dir / "validation_summary.csv", index=False)


if __name__ == "__main__":
    main()
