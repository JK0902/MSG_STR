"""Shared binary-classification metrics for screening analyses."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import confusion_matrix


def prevalence_adjusted_values(
    sensitivity: float,
    specificity: float,
    prevalence: float,
) -> tuple[float, float]:
    """Calculate PPV and NPV at a specified target-population prevalence."""
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must be between 0 and 1.")
    if np.isnan(sensitivity) or np.isnan(specificity):
        return float("nan"), float("nan")
    ppv_denominator = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    npv_denominator = (1 - sensitivity) * prevalence + specificity * (1 - prevalence)
    ppv = sensitivity * prevalence / ppv_denominator if ppv_denominator > 0 else np.nan
    npv = specificity * (1 - prevalence) / npv_denominator if npv_denominator > 0 else np.nan
    return float(ppv), float(npv)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    prevalence: float = 0.10,
) -> Dict[str, float]:
    """Calculate empirical and prevalence-adjusted screening metrics."""
    truth = np.asarray(y_true).astype(int)
    predicted = np.asarray(y_predicted).astype(int)
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_predicted must have the same shape.")

    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    total = int(tn + fp + fn + tp)
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    accuracy = (tp + tn) / total if total else np.nan
    adjusted_ppv, adjusted_npv = prevalence_adjusted_values(
        sensitivity, specificity, prevalence
    )

    return {
        "n": total,
        "prevalence_empirical": (tp + fn) / total if total else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "ppv_adjusted": adjusted_ppv,
        "npv_adjusted": adjusted_npv,
        "f1": float(f1),
        "accuracy": float(accuracy),
        "alert_rate": (tp + fp) / total if total else np.nan,
    }
