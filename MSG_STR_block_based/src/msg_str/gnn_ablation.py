#!/usr/bin/env python3
"""
Evaluate the added value of graph-specific components in the submitted
GNN + Elastic Net/LASSO symptom-prioritization pipeline.

This analysis is intentionally a FEATURE-RANKING ABLATION, not a patient-level
prediction comparison. It starts from the `merged` symptom table produced by
the submitted pipeline and compares:

No-graph event-association score
    z(Elastic Net permutation importance)

Graph-augmented event-association score
    0.5 * z(Elastic Net permutation importance)
    + 0.5 * z(GNN event-loss delta)

No-graph structural importance
    mean[
        z(symptom coverage),
        z(recency-weighted coverage),
        z(mean message-to-symptom edge weight)
    ]

Graph-augmented structural importance
    mean[
        z(symptom coverage),
        z(recency-weighted coverage),
        z(mean message-to-symptom edge weight),
        z(PageRank)
    ]

The analysis reports:
  * Spearman and Kendall rank correlations
  * top-k overlap for event association and structural importance
  * graph/no-graph high-risk candidate-set overlap
  * symptoms promoted or demoted by the graph
  * graph-only and no-graph-only high-risk candidates
  * score-reproduction checks against the submitted composite columns
  * a four-panel comparison figure

Expected input
--------------
The CSV should normally be the existing output:

    symptom_importance_with_elasticnet.csv

Required columns:
    symptom_id
    coverage
    coverage_recency
    learned_link_prob
    pagerank
    en_perm_importance
    gnn_event_delta
    elasticnet_coef

Optional but useful:
    symptom_name
    event_assoc_score
    importance_score

Command-line example
--------------------
python Graph_Added_Value_Ablation.py \
    --input-csv ./GNN_Gemini3_all_no19/symptom_importance_with_elasticnet.csv \
    --out-dir ./Graph_Added_Value \
    --percentile-threshold 80 \
    --top-k-values 10,20

Notebook example
----------------
from Graph_Added_Value_Ablation import evaluate_graph_added_value

comparison, summary = evaluate_graph_added_value(
    merged,
    out_dir="./Graph_Added_Value",
    percentile_threshold=80,
    top_k_values=(10, 20),
)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "symptom_id",
    "coverage",
    "coverage_recency",
    "learned_link_prob",
    "pagerank",
    "en_perm_importance",
    "gnn_event_delta",
    "elasticnet_coef",
}


def population_zscore(values: pd.Series) -> pd.Series:
    """Use ddof=0 to match the submitted numpy implementation."""
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    standard_deviation = float(numeric.to_numpy().std(ddof=0))
    if standard_deviation <= 1e-12:
        return pd.Series(0.0, index=numeric.index, dtype=float)
    return (numeric - float(numeric.mean())) / standard_deviation


def percentile_rank(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").fillna(0.0).rank(pct=True) * 100.0


def safe_spearman(first: pd.Series, second: pd.Series) -> Tuple[float, float]:
    try:
        from scipy.stats import spearmanr

        statistic, p_value = spearmanr(first, second, nan_policy="omit")
        return float(statistic), float(p_value)
    except Exception:
        return float(first.rank().corr(second.rank())), float("nan")


def safe_kendall(first: pd.Series, second: pd.Series) -> Tuple[float, float]:
    try:
        from scipy.stats import kendalltau

        statistic, p_value = kendalltau(first, second, nan_policy="omit")
        return float(statistic), float(p_value)
    except Exception:
        return float(first.rank().corr(second.rank(), method="kendall")), float("nan")


def jaccard(first: Set[str], second: Set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def overlap_coefficient(first: Set[str], second: Set[str]) -> float:
    denominator = min(len(first), len(second))
    return len(first & second) / denominator if denominator else 1.0


def top_symptom_ids(
    frame: pd.DataFrame,
    score_column: str,
    k: int,
    positive_only: bool = False,
) -> Set[str]:
    subset = frame
    if positive_only:
        subset = subset[subset["positive_elasticnet_association"]]
    return set(
        subset.sort_values(score_column, ascending=False)
        .head(int(k))["symptom_id"]
        .astype(str)
    )


def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input ranking table is missing columns: {sorted(missing)}")

    output = frame.copy()
    output["symptom_id"] = output["symptom_id"].astype(str)
    if output["symptom_id"].duplicated().any():
        duplicates = output.loc[output["symptom_id"].duplicated(), "symptom_id"].tolist()
        raise ValueError(f"Duplicate symptom IDs found: {duplicates[:10]}")

    numeric_columns = sorted(REQUIRED_COLUMNS - {"symptom_id"})
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if output[numeric_columns].isna().any().any():
        bad_columns = output[numeric_columns].columns[
            output[numeric_columns].isna().any()
        ].tolist()
        raise ValueError(f"Missing or nonnumeric values found in: {bad_columns}")

    if "symptom_name" not in output.columns:
        output["symptom_name"] = output["symptom_id"]
    else:
        output["symptom_name"] = output["symptom_name"].fillna(output["symptom_id"]).astype(str)
    return output


def calculate_ablation_scores(frame: pd.DataFrame, graph_weight: float) -> pd.DataFrame:
    if not 0.0 <= float(graph_weight) <= 1.0:
        raise ValueError("graph_weight must be between 0 and 1.")

    output = validate_input(frame)

    # Standardized raw components.
    output["z_en_perm"] = population_zscore(output["en_perm_importance"])
    output["z_gnn_delta"] = population_zscore(output["gnn_event_delta"])
    output["z_coverage"] = population_zscore(output["coverage"])
    output["z_recency"] = population_zscore(output["coverage_recency"])
    output["z_link_weight"] = population_zscore(output["learned_link_prob"])
    output["z_pagerank"] = population_zscore(output["pagerank"])

    # Exact no-graph and graph-augmented event-association ablation.
    output["event_score_no_graph"] = output["z_en_perm"]
    output["event_score_with_graph"] = (
        (1.0 - float(graph_weight)) * output["z_en_perm"]
        + float(graph_weight) * output["z_gnn_delta"]
    )

    # Exact no-graph and graph-augmented structural-importance ablation.
    output["importance_score_no_graph"] = output[
        ["z_coverage", "z_recency", "z_link_weight"]
    ].mean(axis=1)
    output["importance_score_with_graph"] = output[
        ["z_coverage", "z_recency", "z_link_weight", "z_pagerank"]
    ].mean(axis=1)

    output["positive_elasticnet_association"] = output["elasticnet_coef"] > 0

    # Ranks: 1 is strongest. Positive change means the graph promoted a symptom.
    output["event_rank_no_graph"] = output["event_score_no_graph"].rank(
        ascending=False, method="min"
    )
    output["event_rank_with_graph"] = output["event_score_with_graph"].rank(
        ascending=False, method="min"
    )
    output["event_rank_change_with_graph"] = (
        output["event_rank_no_graph"] - output["event_rank_with_graph"]
    )

    output["importance_rank_no_graph"] = output["importance_score_no_graph"].rank(
        ascending=False, method="min"
    )
    output["importance_rank_with_graph"] = output["importance_score_with_graph"].rank(
        ascending=False, method="min"
    )
    output["importance_rank_change_with_graph"] = (
        output["importance_rank_no_graph"] - output["importance_rank_with_graph"]
    )

    # Score changes are descriptive because the two composites have different components.
    output["event_score_change_with_graph"] = (
        output["event_score_with_graph"] - output["event_score_no_graph"]
    )
    output["importance_score_change_with_graph"] = (
        output["importance_score_with_graph"] - output["importance_score_no_graph"]
    )

    return output


def assign_candidate_sets(frame: pd.DataFrame, percentile_threshold: float) -> pd.DataFrame:
    if not 0.0 < float(percentile_threshold) <= 100.0:
        raise ValueError("percentile_threshold must be in (0, 100].")

    output = frame.copy()
    output["event_percentile_no_graph"] = percentile_rank(output["event_score_no_graph"])
    output["event_percentile_with_graph"] = percentile_rank(output["event_score_with_graph"])
    output["importance_percentile_no_graph"] = percentile_rank(
        output["importance_score_no_graph"]
    )
    output["importance_percentile_with_graph"] = percentile_rank(
        output["importance_score_with_graph"]
    )

    positive = output["positive_elasticnet_association"]
    output["high_risk_candidate_no_graph"] = (
        positive
        & (output["event_percentile_no_graph"] >= float(percentile_threshold))
        & (output["importance_percentile_no_graph"] >= float(percentile_threshold))
    )
    output["high_risk_candidate_with_graph"] = (
        positive
        & (output["event_percentile_with_graph"] >= float(percentile_threshold))
        & (output["importance_percentile_with_graph"] >= float(percentile_threshold))
    )
    output["candidate_status"] = np.select(
        [
            output["high_risk_candidate_no_graph"]
            & output["high_risk_candidate_with_graph"],
            ~output["high_risk_candidate_no_graph"]
            & output["high_risk_candidate_with_graph"],
            output["high_risk_candidate_no_graph"]
            & ~output["high_risk_candidate_with_graph"],
        ],
        ["Selected by both", "Graph only", "No graph only"],
        default="Selected by neither",
    )
    return output


def summarize_ablation(
    comparison: pd.DataFrame,
    top_k_values: Sequence[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for comparison_name, first_column, second_column in [
        (
            "event-association ranking",
            "event_score_no_graph",
            "event_score_with_graph",
        ),
        (
            "structural-importance ranking",
            "importance_score_no_graph",
            "importance_score_with_graph",
        ),
    ]:
        rho, rho_p = safe_spearman(comparison[first_column], comparison[second_column])
        tau, tau_p = safe_kendall(comparison[first_column], comparison[second_column])
        rows.extend(
            [
                {
                    "comparison": comparison_name,
                    "metric": "spearman_rho",
                    "value": rho,
                    "p_value": rho_p,
                    "n_symptoms": len(comparison),
                },
                {
                    "comparison": comparison_name,
                    "metric": "kendall_tau",
                    "value": tau,
                    "p_value": tau_p,
                    "n_symptoms": len(comparison),
                },
            ]
        )

    rows.extend(
        [
            {
                "comparison": "event-association ranking",
                "metric": "mean_absolute_rank_change",
                "value": float(comparison["event_rank_change_with_graph"].abs().mean()),
                "p_value": np.nan,
                "n_symptoms": len(comparison),
            },
            {
                "comparison": "structural-importance ranking",
                "metric": "mean_absolute_rank_change",
                "value": float(comparison["importance_rank_change_with_graph"].abs().mean()),
                "p_value": np.nan,
                "n_symptoms": len(comparison),
            },
        ]
    )

    for k in sorted(set(map(int, top_k_values))):
        if k <= 0:
            raise ValueError("All top_k_values must be positive.")

        event_no_graph = top_symptom_ids(
            comparison, "event_score_no_graph", k, positive_only=True
        )
        event_with_graph = top_symptom_ids(
            comparison, "event_score_with_graph", k, positive_only=True
        )
        importance_no_graph = top_symptom_ids(
            comparison, "importance_score_no_graph", k
        )
        importance_with_graph = top_symptom_ids(
            comparison, "importance_score_with_graph", k
        )

        rows.extend(
            [
                {
                    "comparison": f"positive event-association top-{k}",
                    "metric": "overlap_fraction",
                    "value": overlap_coefficient(event_no_graph, event_with_graph),
                    "p_value": np.nan,
                    "n_symptoms": len(comparison),
                },
                {
                    "comparison": f"structural-importance top-{k}",
                    "metric": "overlap_fraction",
                    "value": len(importance_no_graph & importance_with_graph) / max(1, k),
                    "p_value": np.nan,
                    "n_symptoms": len(comparison),
                },
            ]
        )

    no_graph_candidates = set(
        comparison.loc[comparison["high_risk_candidate_no_graph"], "symptom_id"].astype(str)
    )
    graph_candidates = set(
        comparison.loc[comparison["high_risk_candidate_with_graph"], "symptom_id"].astype(str)
    )
    rows.extend(
        [
            {
                "comparison": "high-risk candidate sets",
                "metric": "n_no_graph",
                "value": len(no_graph_candidates),
                "p_value": np.nan,
                "n_symptoms": len(comparison),
            },
            {
                "comparison": "high-risk candidate sets",
                "metric": "n_with_graph",
                "value": len(graph_candidates),
                "p_value": np.nan,
                "n_symptoms": len(comparison),
            },
            {
                "comparison": "high-risk candidate sets",
                "metric": "n_shared",
                "value": len(no_graph_candidates & graph_candidates),
                "p_value": np.nan,
                "n_symptoms": len(comparison),
            },
            {
                "comparison": "high-risk candidate sets",
                "metric": "jaccard",
                "value": jaccard(no_graph_candidates, graph_candidates),
                "p_value": np.nan,
                "n_symptoms": len(comparison),
            },
            {
                "comparison": "high-risk candidate sets",
                "metric": "overlap_coefficient",
                "value": overlap_coefficient(no_graph_candidates, graph_candidates),
                "p_value": np.nan,
                "n_symptoms": len(comparison),
            },
        ]
    )
    return pd.DataFrame(rows)


def score_reproduction_checks(frame: pd.DataFrame) -> Dict[str, object]:
    checks: Dict[str, object] = {}
    if "event_assoc_score" in frame.columns:
        submitted = pd.to_numeric(frame["event_assoc_score"], errors="coerce")
        difference = (submitted - frame["event_score_with_graph"]).abs()
        checks["event_score_max_absolute_difference"] = float(difference.max())
        checks["event_score_reproduced_within_1e-8"] = bool(difference.max() <= 1e-8)
    else:
        checks["event_score_check"] = "event_assoc_score not supplied"

    if "importance_score" in frame.columns:
        submitted = pd.to_numeric(frame["importance_score"], errors="coerce")
        difference = (submitted - frame["importance_score_with_graph"]).abs()
        checks["importance_score_max_absolute_difference"] = float(difference.max())
        checks["importance_score_reproduced_within_1e-8"] = bool(difference.max() <= 1e-8)
    else:
        checks["importance_score_check"] = "importance_score not supplied"
    return checks


def make_ablation_plot(comparison: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    figure, axes = plt.subplots(2, 2, figsize=(11, 9))

    axes[0, 0].scatter(
        comparison["event_score_no_graph"],
        comparison["event_score_with_graph"],
        s=22,
        alpha=0.65,
    )
    axes[0, 0].set_xlabel("EN-only event-association score")
    axes[0, 0].set_ylabel("GNN+EN event-association score")
    axes[0, 0].set_title("Event-association scores")

    axes[0, 1].scatter(
        comparison["importance_score_no_graph"],
        comparison["importance_score_with_graph"],
        s=22,
        alpha=0.65,
    )
    axes[0, 1].set_xlabel("Importance without PageRank")
    axes[0, 1].set_ylabel("Importance with PageRank")
    axes[0, 1].set_title("Structural-importance scores")

    promoted = comparison[comparison["event_rank_change_with_graph"] > 0].nlargest(
        15, "event_rank_change_with_graph"
    ).sort_values("event_rank_change_with_graph")
    if promoted.empty:
        promoted = comparison.nlargest(15, "event_rank_change_with_graph").sort_values(
            "event_rank_change_with_graph"
        )
    axes[1, 0].barh(
        promoted["symptom_name"],
        promoted["event_rank_change_with_graph"],
        color="#2878B5",
    )
    axes[1, 0].set_xlabel("Rank positions promoted by graph")
    axes[1, 0].set_title("Symptoms most promoted by GNN loss delta")

    status_counts = comparison["candidate_status"].value_counts().reindex(
        ["Selected by both", "Graph only", "No graph only", "Selected by neither"],
        fill_value=0,
    )
    axes[1, 1].bar(
        status_counts.index,
        status_counts.values,
        color=["#4C956C", "#2878B5", "#D95F59", "#B8B8B8"],
    )
    axes[1, 1].tick_params(axis="x", rotation=25)
    axes[1, 1].set_ylabel("Number of symptoms")
    axes[1, 1].set_title("High-risk candidate selection")

    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate_graph_added_value(
    merged: pd.DataFrame,
    out_dir: Union[str, Path] = "./Graph_Added_Value",
    percentile_threshold: float = 80.0,
    top_k_values: Sequence[int] = (10, 20),
    graph_weight: float = 0.5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete graph-component ablation and save all outputs."""
    output_directory = Path(out_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    comparison = calculate_ablation_scores(merged, graph_weight=graph_weight)
    comparison = assign_candidate_sets(comparison, percentile_threshold)
    summary = summarize_ablation(comparison, top_k_values)
    checks = score_reproduction_checks(comparison)

    comparison = comparison.sort_values(
        ["event_rank_change_with_graph", "importance_rank_change_with_graph"],
        ascending=False,
    ).reset_index(drop=True)
    promoted = comparison[comparison["event_rank_change_with_graph"] > 0].nlargest(
        20, "event_rank_change_with_graph"
    )
    demoted = comparison[comparison["event_rank_change_with_graph"] < 0].nsmallest(
        20, "event_rank_change_with_graph"
    )
    graph_only = comparison[comparison["candidate_status"] == "Graph only"]
    no_graph_only = comparison[comparison["candidate_status"] == "No graph only"]
    shared = comparison[comparison["candidate_status"] == "Selected by both"]

    comparison.to_csv(output_directory / "graph_added_value_symptom_level.csv", index=False)
    summary.to_csv(output_directory / "graph_added_value_summary.csv", index=False)
    promoted.to_csv(output_directory / "symptoms_promoted_by_graph.csv", index=False)
    demoted.to_csv(output_directory / "symptoms_demoted_by_graph.csv", index=False)
    graph_only.to_csv(output_directory / "graph_only_high_risk_candidates.csv", index=False)
    no_graph_only.to_csv(output_directory / "no_graph_only_high_risk_candidates.csv", index=False)
    shared.to_csv(output_directory / "shared_high_risk_candidates.csv", index=False)
    make_ablation_plot(comparison, output_directory / "graph_added_value_ablation.png")

    metadata = {
        "analysis_type": "feature-ranking graph ablation",
        "graph_weight_in_event_score": float(graph_weight),
        "percentile_threshold": float(percentile_threshold),
        "top_k_values": list(map(int, top_k_values)),
        "n_symptoms": int(len(comparison)),
        "event_score_no_graph": "z(en_perm_importance)",
        "event_score_with_graph": (
            f"{1.0 - graph_weight:.3f}*z(en_perm_importance) + "
            f"{graph_weight:.3f}*z(gnn_event_delta)"
        ),
        "importance_score_no_graph": (
            "mean(z(coverage), z(coverage_recency), z(learned_link_prob))"
        ),
        "importance_score_with_graph": (
            "mean(z(coverage), z(coverage_recency), "
            "z(learned_link_prob), z(pagerank))"
        ),
        "score_reproduction_checks": checks,
        "interpretation": (
            "Positive rank change means the graph moved a symptom higher. "
            "This analysis measures incremental ranking contribution, not "
            "improvement in held-out patient-level prediction."
        ),
    }
    with open(output_directory / "graph_added_value_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print("\nGraph added-value summary")
    print(summary.to_string(index=False))
    print("\nScore-reproduction checks")
    print(json.dumps(checks, indent=2))
    print(f"\nOutputs saved to: {output_directory}")
    return comparison, summary


def parse_top_k_values(value: str) -> List[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("top-k values must be positive integers.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("./Graph_Added_Value"))
    parser.add_argument("--percentile-threshold", type=float, default=80.0)
    parser.add_argument("--top-k-values", type=parse_top_k_values, default=[10, 20])
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=0.5,
        help="Weight assigned to z(gnn_event_delta); 0.5 matches submitted code.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input_csv.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input ranking file not found: {input_path}")

    merged = pd.read_csv(input_path)
    evaluate_graph_added_value(
        merged,
        out_dir=args.out_dir,
        percentile_threshold=args.percentile_threshold,
        top_k_values=args.top_k_values,
        graph_weight=args.graph_weight,
    )


if __name__ == "__main__":
    main()
