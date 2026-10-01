"""Manuscript block: hybrid stroke-risk screening simulation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .metrics import compute_binary_metrics
from .privacy import assert_columns


RISK_CATEGORIES = ("High", "Moderate", "Moderate-low", "Low")


def count_column(category: str, window_days: int) -> str:
    """Return the standardized feature name for a risk category."""
    compact = category.replace("-", "").replace(" ", "")
    return f"{compact}_count_{window_days}d"


def screening_feature_columns(window_days: int) -> list[str]:
    """Return logistic-regression feature columns for one screening window."""
    return [
        f"VeryHigh_count_{window_days}d",
        count_column("High", window_days),
        count_column("Moderate", window_days),
        count_column("Moderate-low", window_days),
        count_column("Low", window_days),
    ]


def attach_risk_rubric(
    messages: pd.DataFrame,
    rubric: pd.DataFrame,
    classification_column: str = "classifications",
    symptom_id_column: str = "symptom_id",
    category_column: str = "symptom_risk_category",
    risk_score_column: str = "stroke_risk_score",
    very_high_threshold: float = 0.85,
) -> pd.DataFrame:
    """Attach symptom-risk categories and the very-high-risk indicator."""
    assert_columns(messages, [classification_column], "messages")
    assert_columns(
        rubric,
        [symptom_id_column, category_column, risk_score_column],
        "rubric",
    )
    rubric_copy = rubric.copy()
    rubric_copy["is_veryhigh"] = (
        rubric_copy[category_column].eq("High")
        & pd.to_numeric(rubric_copy[risk_score_column], errors="coerce").ge(
            very_high_threshold
        )
    )
    return messages.merge(
        rubric_copy[[symptom_id_column, category_column, "is_veryhigh"]],
        left_on=classification_column,
        right_on=symptom_id_column,
        how="left",
        validate="many_to_one",
    )


def compute_screening_features(
    frame: pd.DataFrame,
    window_days: int,
    person_column: str = "user_id",
    days_column: str = "time_difference_days",
    category_column: str = "symptom_risk_category",
    very_high_column: str = "is_veryhigh",
    outcome_column: str = "stroke_event",
) -> pd.DataFrame:
    """Aggregate message-level risk categories into person-level window features."""
    assert_columns(
        frame,
        [person_column, days_column, category_column, outcome_column],
        "screening input",
    )
    if window_days <= 0:
        raise ValueError("window_days must be positive.")

    working = frame.copy()
    working[days_column] = pd.to_numeric(working[days_column], errors="coerce")
    outcomes = (
        working.groupby(person_column, as_index=False)[outcome_column]
        .max()
        .astype({outcome_column: int})
    )
    window = working.loc[
        working[days_column].between(0, window_days, inclusive="both")
    ].copy()

    output = outcomes.copy()
    if not window.empty:
        category_counts = pd.crosstab(
            window[person_column], window[category_column]
        ).rename_axis(index=person_column)
        category_counts = category_counts.reset_index()
        output = output.merge(category_counts, on=person_column, how="left")

        if very_high_column in window.columns:
            very_high = (
                window.loc[window[very_high_column].fillna(False).astype(bool)]
                .groupby(person_column)
                .size()
                .rename(f"VeryHigh_count_{window_days}d")
                .reset_index()
            )
            output = output.merge(very_high, on=person_column, how="left")

    rename_map = {
        category: count_column(category, window_days) for category in RISK_CATEGORIES
    }
    output = output.rename(columns=rename_map)
    for column in screening_feature_columns(window_days):
        if column not in output:
            output[column] = 0
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0).astype(int)

    output[f"total_symptoms_{window_days}d"] = output[
        [count_column(category, window_days) for category in RISK_CATEGORIES]
    ].sum(axis=1)
    return output


@dataclass
class LogisticScreeningResult:
    """Artifacts from a leakage-controlled train/test split."""

    model: LogisticRegression
    coefficient_table: pd.DataFrame
    test_predictions: pd.DataFrame
    roc_auc: float
    average_precision: float
    feature_columns: list[str]


def fit_logistic_screening(
    feature_frame: pd.DataFrame,
    window_days: int,
    outcome_column: str = "stroke_event",
    test_size: float = 0.30,
    random_state: int = 42,
) -> LogisticScreeningResult:
    """Fit on the training split and return probabilities only for held-out rows."""
    features = screening_feature_columns(window_days)
    assert_columns(feature_frame, features + [outcome_column], "feature_frame")
    if feature_frame[outcome_column].nunique() != 2:
        raise ValueError("The outcome must contain both classes.")

    train_index, test_index = train_test_split(
        np.arange(len(feature_frame)),
        test_size=test_size,
        stratify=feature_frame[outcome_column].astype(int).to_numpy(),
        random_state=random_state,
    )
    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(
        feature_frame.iloc[train_index][features].astype(float),
        feature_frame.iloc[train_index][outcome_column].astype(int),
    )

    test = feature_frame.iloc[test_index].copy()
    test[f"p_{window_days}d"] = model.predict_proba(test[features].astype(float))[:, 1]
    y_test = test[outcome_column].astype(int).to_numpy()
    probabilities = test[f"p_{window_days}d"].to_numpy()

    coefficients = pd.DataFrame(
        {
            "feature": ["intercept"] + features,
            "coefficient": np.concatenate([model.intercept_, model.coef_[0]]),
        }
    )
    coefficients["odds_ratio"] = np.exp(coefficients["coefficient"])
    return LogisticScreeningResult(
        model=model,
        coefficient_table=coefficients,
        test_predictions=test,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        average_precision=float(average_precision_score(y_test, probabilities)),
        feature_columns=features,
    )


def threshold_scan(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    thresholds: Iterable[float],
    prevalence: float = 0.10,
) -> pd.DataFrame:
    """Evaluate probability thresholds on a held-out sample."""
    truth = np.asarray(y_true).astype(int)
    scores = np.asarray(probabilities).astype(float)
    rows = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(
            truth,
            (scores >= float(threshold)).astype(int),
            prevalence=prevalence,
        )
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return pd.DataFrame(rows)


def apply_symptom_rule(
    frame: pd.DataFrame,
    window_days: int,
    very_high_threshold: int,
    high_threshold: int,
    moderate_threshold: int,
    high_moderate_sum_threshold: int,
) -> np.ndarray:
    """Apply the manuscript symptom-count rule."""
    very_high, high, moderate = screening_feature_columns(window_days)[:3]
    assert_columns(frame, [very_high, high, moderate], "feature_frame")
    vh = frame[very_high]
    h = frame[high]
    m = frame[moderate]
    return (
        vh.ge(very_high_threshold)
        | h.ge(high_threshold)
        | (h.ge(1) & m.ge(moderate_threshold))
        | (h + m).ge(high_moderate_sum_threshold)
    ).astype(int).to_numpy()


def grid_search_symptom_rule(
    feature_frame: pd.DataFrame,
    window_days: int,
    very_high_values: Iterable[int] = (1,),
    high_values: Iterable[int] = range(1, 4),
    moderate_values: Iterable[int] = range(1, 4),
    combination_values: Iterable[int] = range(1, 6),
    prevalence: float = 0.10,
    outcome_column: str = "stroke_event",
) -> pd.DataFrame:
    """Grid-search symptom-rule thresholds using explicit positive thresholds."""
    assert_columns(feature_frame, [outcome_column], "feature_frame")
    truth = feature_frame[outcome_column].astype(int).to_numpy()
    rows = []
    for values in product(
        very_high_values, high_values, moderate_values, combination_values
    ):
        vh, high, moderate, combination = values
        predictions = apply_symptom_rule(
            feature_frame,
            window_days,
            vh,
            high,
            moderate,
            combination,
        )
        metrics = compute_binary_metrics(truth, predictions, prevalence)
        metrics.update(
            {
                "very_high_threshold": vh,
                "high_threshold": high,
                "moderate_threshold": moderate,
                "high_moderate_sum_threshold": combination,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def scan_hybrid_thresholds(
    test_predictions: pd.DataFrame,
    window_days: int,
    symptom_thresholds: tuple[int, int, int, int],
    probability_thresholds: Iterable[float],
    prevalence: float = 0.10,
    outcome_column: str = "stroke_event",
) -> pd.DataFrame:
    """Evaluate the OR combination of symptom and logistic flags."""
    probability_column = f"p_{window_days}d"
    assert_columns(
        test_predictions,
        [outcome_column, probability_column],
        "test_predictions",
    )
    symptom_flag = apply_symptom_rule(
        test_predictions,
        window_days,
        *symptom_thresholds,
    )
    truth = test_predictions[outcome_column].astype(int).to_numpy()
    probabilities = test_predictions[probability_column].astype(float).to_numpy()

    rows = []
    for threshold in probability_thresholds:
        logistic_flag = (probabilities >= float(threshold)).astype(int)
        hybrid_flag = ((symptom_flag == 1) | (logistic_flag == 1)).astype(int)
        metrics = compute_binary_metrics(truth, hybrid_flag, prevalence)
        metrics["threshold"] = float(threshold)
        rows.append(metrics)
    return pd.DataFrame(rows)


def select_high_specificity_operating_point(
    scan: pd.DataFrame,
    minimum_specificity: float = 0.90,
) -> pd.Series:
    """Select maximum sensitivity, then minimum alert rate, under specificity constraint."""
    assert_columns(scan, ["specificity", "sensitivity", "alert_rate"], "scan")
    eligible = scan.loc[scan["specificity"] >= minimum_specificity].copy()
    if eligible.empty:
        raise ValueError("No operating point meets the specificity requirement.")
    return eligible.sort_values(
        ["sensitivity", "alert_rate", "threshold"],
        ascending=[False, True, False],
    ).iloc[0]
