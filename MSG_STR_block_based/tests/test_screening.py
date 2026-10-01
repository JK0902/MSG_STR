import unittest

import numpy as np
import pandas as pd

from msg_str.metrics import compute_binary_metrics
from msg_str.screening import (
    apply_symptom_rule,
    attach_risk_rubric,
    compute_screening_features,
    fit_logistic_screening,
    scan_hybrid_thresholds,
    screening_feature_columns,
    select_high_specificity_operating_point,
)


def synthetic_message_data(n_people: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    rows = []
    for person in range(n_people):
        outcome = int(person % 3 == 0)
        for message in range(3):
            symptom = 1 if outcome and message < 2 else int(rng.integers(1, 5))
            rows.append(
                {
                    "user_id": f"P{person:03d}",
                    "classifications": symptom,
                    "time_difference_days": float(rng.integers(0, 10)),
                    "stroke_event": outcome,
                }
            )
    rubric = pd.DataFrame(
        {
            "symptom_id": [1, 2, 3, 4],
            "symptom_risk_category": ["High", "Moderate", "Moderate-low", "Low"],
            "stroke_risk_score": [0.90, 0.70, 0.50, 0.20],
        }
    )
    return pd.DataFrame(rows), rubric


class ScreeningTests(unittest.TestCase):
    def test_feature_aggregation_keeps_people_without_window_messages(self) -> None:
        messages, rubric = synthetic_message_data(12)
        enriched = attach_risk_rubric(messages, rubric)
        features = compute_screening_features(enriched, window_days=3)
        self.assertEqual(len(features), 12)
        self.assertTrue(set(screening_feature_columns(3)).issubset(features.columns))
        self.assertFalse(features[screening_feature_columns(3)].isna().any().any())

    def test_logistic_probabilities_come_from_heldout_rows(self) -> None:
        messages, rubric = synthetic_message_data()
        features = compute_screening_features(
            attach_risk_rubric(messages, rubric), window_days=3
        )
        result = fit_logistic_screening(features, window_days=3)
        self.assertLess(len(result.test_predictions), len(features))
        self.assertTrue(result.test_predictions["p_3d"].between(0, 1).all())
        self.assertEqual(
            result.coefficient_table["feature"].tolist(),
            ["intercept"] + screening_feature_columns(3),
        )

    def test_rule_and_hybrid_scan(self) -> None:
        messages, rubric = synthetic_message_data()
        features = compute_screening_features(
            attach_risk_rubric(messages, rubric), window_days=3
        )
        fitted = fit_logistic_screening(features, window_days=3)
        rule = apply_symptom_rule(fitted.test_predictions, 3, 1, 1, 1, 2)
        self.assertEqual(len(rule), len(fitted.test_predictions))
        scan = scan_hybrid_thresholds(
            fitted.test_predictions,
            window_days=3,
            symptom_thresholds=(1, 1, 1, 2),
            probability_thresholds=(0.0, 0.5, 1.0),
        )
        self.assertEqual(len(scan), 3)
        selected = select_high_specificity_operating_point(scan, minimum_specificity=0.0)
        self.assertIn("threshold", selected.index)

    def test_metrics_include_alert_rate_and_adjusted_values(self) -> None:
        metrics = compute_binary_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            prevalence=0.10,
        )
        self.assertAlmostEqual(metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["specificity"], 0.5)
        self.assertAlmostEqual(metrics["alert_rate"], 0.75)
        self.assertIn("ppv_adjusted", metrics)


if __name__ == "__main__":
    unittest.main()
