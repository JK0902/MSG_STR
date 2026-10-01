"""Compute only the two graph-ablation results reported in the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, Tuple

import pandas as pd

from .gnn_ablation import (
    assign_candidate_sets,
    calculate_ablation_scores,
    overlap_coefficient,
    safe_spearman,
    top_symptom_ids,
)


def compute_reported_results(
    frame: pd.DataFrame,
    graph_weight: float = 0.5,
    percentile_threshold: float = 80.0,
    top_k_values: Sequence[int] = (10, 20),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the comparison table, one-row summary, and graph-only candidates."""
    comparison = calculate_ablation_scores(frame, graph_weight=graph_weight)
    comparison = assign_candidate_sets(comparison, percentile_threshold)

    rho, p_value = safe_spearman(
        comparison["event_score_no_graph"],
        comparison["event_score_with_graph"],
    )

    summary = {
        "n_symptoms": int(len(comparison)),
        "spearman_rho": float(rho),
        "spearman_p_value_two_sided": float(p_value),
        "mean_absolute_rank_change": float(
            comparison["event_rank_change_with_graph"].abs().mean()
        ),
    }

    for k in sorted(set(map(int, top_k_values))):
        if k <= 0:
            raise ValueError("All top-k values must be positive integers.")
        no_graph = top_symptom_ids(
            comparison,
            "event_score_no_graph",
            k,
            positive_only=True,
        )
        with_graph = top_symptom_ids(
            comparison,
            "event_score_with_graph",
            k,
            positive_only=True,
        )
        summary[f"top_{k}_overlap_n"] = int(len(no_graph & with_graph))
        summary[f"top_{k}_overlap_fraction"] = float(
            overlap_coefficient(no_graph, with_graph)
        )

    graph_only = (
        comparison.loc[
            comparison["high_risk_candidate_with_graph"]
            & ~comparison["high_risk_candidate_no_graph"]
        ]
        .sort_values("event_score_with_graph", ascending=False)
        .reset_index(drop=True)
    )
    summary["n_graph_only_candidates"] = int(len(graph_only))

    return comparison, pd.DataFrame([summary]), graph_only


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report ranking agreement and graph-only symptom candidates."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results/reported_results"))
    parser.add_argument("--graph-weight", type=float, default=0.5)
    parser.add_argument("--percentile-threshold", type=float, default=80.0)
    parser.add_argument("--top-k-values", default="10,20")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    top_k_values = tuple(
        sorted({int(value.strip()) for value in args.top_k_values.split(",") if value.strip()})
    )
    frame = pd.read_csv(args.input_csv)
    comparison, summary, graph_only = compute_reported_results(
        frame,
        graph_weight=args.graph_weight,
        percentile_threshold=args.percentile_threshold,
        top_k_values=top_k_values,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "reported_ranking_results.csv", index=False)

    candidate_columns = [
        "symptom_id",
        "symptom_name",
        "elasticnet_coef",
        "gnn_event_delta",
        "pagerank",
        "event_rank_no_graph",
        "event_rank_with_graph",
        "event_rank_change_with_graph",
        "importance_rank_no_graph",
        "importance_rank_with_graph",
        "importance_rank_change_with_graph",
    ]
    graph_only[candidate_columns].to_csv(
        args.out_dir / "graph_only_additional_symptoms.csv",
        index=False,
    )

    metadata = {
        "graph_weight": float(args.graph_weight),
        "percentile_threshold": float(args.percentile_threshold),
        "top_k_values": list(top_k_values),
        "top_k_definition": (
            "Top-k among symptoms with a positive Elastic Net coefficient; "
            "overlap is the intersection divided by the smaller set size."
        ),
        "graph_only_definition": (
            "Selected by the graph-augmented candidate rule but not by the "
            "no-graph candidate rule."
        ),
        "statistical_test": (
            "Two-sided Spearman rank-correlation test. Top-k overlap, mean "
            "absolute rank change, and graph-only counts are descriptive."
        ),
    }
    with open(args.out_dir / "reported_results_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(summary.to_string(index=False))
    print("\nGraph-only additional symptoms")
    if graph_only.empty:
        print("None")
    else:
        print(graph_only[["symptom_id", "symptom_name"]].to_string(index=False))


if __name__ == "__main__":
    main()
