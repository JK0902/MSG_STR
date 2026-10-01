import unittest

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from msg_str.gnn_ablation import (
    assign_candidate_sets,
    calculate_ablation_scores,
    overlap_coefficient,
    top_symptom_ids,
)
from msg_str.gnn_ablation_report import compute_reported_results


def symptom_table(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "symptom_id": [f"S{i:03d}" for i in range(n)],
            "symptom_name": [f"Synthetic symptom {i}" for i in range(n)],
            "coverage": rng.integers(1, 100, n),
            "coverage_recency": rng.uniform(0, 30, n),
            "learned_link_prob": rng.uniform(0.4, 0.9, n),
            "pagerank": rng.uniform(0.001, 0.05, n),
            "en_perm_importance": rng.normal(0.02, 0.01, n),
            "gnn_event_delta": rng.normal(0.00, 0.02, n),
            "elasticnet_coef": rng.normal(0.00, 0.5, n),
        }
    )


class GnnAblationTests(unittest.TestCase):
    def test_equal_weight_score_and_zero_weight_limit(self) -> None:
        equal = calculate_ablation_scores(symptom_table(), graph_weight=0.5)
        expected = 0.5 * equal["z_en_perm"] + 0.5 * equal["z_gnn_delta"]
        np.testing.assert_allclose(equal["event_score_with_graph"], expected)

        zero = calculate_ablation_scores(symptom_table(), graph_weight=0.0)
        np.testing.assert_allclose(
            zero["event_score_with_graph"], zero["event_score_no_graph"]
        )

    def test_reported_statistics_match_independent_calculation(self) -> None:
        comparison, summary_frame, graph_only = compute_reported_results(
            symptom_table(), top_k_values=(10, 20)
        )
        summary = summary_frame.iloc[0]
        rho, p_value = spearmanr(
            comparison["event_score_no_graph"],
            comparison["event_score_with_graph"],
        )
        self.assertAlmostEqual(summary["spearman_rho"], rho)
        self.assertAlmostEqual(summary["spearman_p_value_two_sided"], p_value)
        self.assertAlmostEqual(
            summary["mean_absolute_rank_change"],
            comparison["event_rank_change_with_graph"].abs().mean(),
        )
        for k in (10, 20):
            no_graph = top_symptom_ids(
                comparison, "event_score_no_graph", k, positive_only=True
            )
            with_graph = top_symptom_ids(
                comparison, "event_score_with_graph", k, positive_only=True
            )
            self.assertAlmostEqual(
                summary[f"top_{k}_overlap_fraction"],
                overlap_coefficient(no_graph, with_graph),
            )

        expected_graph_only = assign_candidate_sets(
            calculate_ablation_scores(symptom_table(), 0.5), 80
        ).query('candidate_status == "Graph only"')
        self.assertEqual(
            set(graph_only["symptom_id"]), set(expected_graph_only["symptom_id"])
        )


if __name__ == "__main__":
    unittest.main()
