#!/usr/bin/env python3
"""Run the modular hybrid-screening block from de-identified CSV inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from msg_str.screening import (
    attach_risk_rubric,
    compute_screening_features,
    fit_logistic_screening,
    scan_hybrid_thresholds,
    select_high_specificity_operating_point,
)


def parse_windows(value: str) -> list[int]:
    windows = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not windows or min(windows) <= 0:
        raise argparse.ArgumentTypeError("windows must be positive integers.")
    return windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--messages-csv", type=Path, required=True)
    parser.add_argument("--rubric-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/screening"))
    parser.add_argument("--windows", type=parse_windows, default=[3, 7, 14, 30, 60, 90])
    parser.add_argument("--prevalence", type=float, default=0.10)
    parser.add_argument("--minimum-specificity", type=float, default=0.90)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    messages = pd.read_csv(args.messages_csv)
    rubric = pd.read_csv(args.rubric_csv)
    if "time_difference_days" not in messages and "time_difference" in messages:
        messages["time_difference_days"] = (
            pd.to_timedelta(messages["time_difference"]).dt.total_seconds() / 86400
        )
    enriched = attach_risk_rubric(messages, rubric)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for window_days in args.windows:
        features = compute_screening_features(enriched, window_days=window_days)
        fitted = fit_logistic_screening(
            features,
            window_days=window_days,
            random_state=args.random_state,
        )
        scan = scan_hybrid_thresholds(
            fitted.test_predictions,
            window_days=window_days,
            symptom_thresholds=(1, 1, 1, 2),
            probability_thresholds=np.linspace(0, 1, 101),
            prevalence=args.prevalence,
        )
        operating_point = select_high_specificity_operating_point(
            scan,
            minimum_specificity=args.minimum_specificity,
        )
        output = args.out_dir / f"window_{window_days}d"
        output.mkdir(parents=True, exist_ok=True)
        features.to_csv(output / "person_level_features.csv", index=False)
        fitted.coefficient_table.to_csv(output / "logistic_coefficients.csv", index=False)
        scan.to_csv(output / "hybrid_threshold_scan.csv", index=False)
        summary = operating_point.to_dict()
        summary.update(
            {
                "window_days": window_days,
                "roc_auc": fitted.roc_auc,
                "average_precision": fitted.average_precision,
            }
        )
        summaries.append(summary)

    pd.DataFrame(summaries).sort_values("window_days").to_csv(
        args.out_dir / "screening_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
