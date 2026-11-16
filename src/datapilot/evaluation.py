"""
Stage 5 (error analysis) + Stage 6 (held-out evaluation).

Fits the final chosen (pipeline, model) combo fresh on the training split
and evaluates it on the test split that has never been touched by any
fitting, tuning, or CV up to this point — the real, final number, not a CV
estimate. Error analysis (confusion matrix for classification, residuals
for regression) runs on those same real predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline

from .reasoner import TaskType


def _rmsle(y, pred) -> float | None:
    y = np.asarray(y, dtype=float)
    if np.nanmin(y) < 0:
        return None
    return float(np.sqrt(np.mean((np.log1p(np.clip(pred, 0, None)) - np.log1p(y)) ** 2)))


@dataclass
class RegressionEvaluation:
    r2: float
    rmse: float
    mae: float
    mape: float
    residual_mean: float
    residual_std: float
    worst_errors: list[dict] = field(default_factory=list)  # [{index, actual, predicted, abs_error}]
    rmsle: float | None = None   # only when the target is never negative


@dataclass
class ClassificationEvaluation:
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    roc_auc: float | None
    labels: list = field(default_factory=list)
    confusion: list[list[int]] = field(default_factory=list)  # rows=actual, cols=predicted


@dataclass
class FinalEvaluationResult:
    pipeline_name: str
    model_name: str
    task_type: TaskType
    n_test: int
    regression: RegressionEvaluation | None = None
    classification: ClassificationEvaluation | None = None
    error: str | None = None
    scored_by: str = "holdout"   # "holdout" / "external": rows the model never saw; "cv": cross-checked on training rows

    @property
    def ok(self) -> bool:
        return self.error is None


def evaluate_on_held_out_test(
    pipeline_name: str,
    model_name: str,
    fitted_pipeline: Pipeline,
    X_test, y_test,
    task_type: TaskType,
    n_worst: int = 5,
    label_encoder=None,
) -> FinalEvaluationResult:
    """`fitted_pipeline` must already be fit on the TRAINING split only —
    this function only predicts on X_test, it never fits on it. That
    ordering is the entire point of a held-out evaluation.

    `label_encoder`: the same encoder train_test_split_data() fit for
    classification tasks (y_test/preds are its encoded 0..n-1 integers).
    When given, the confusion matrix's row/column labels are decoded back
    to the original class names for display — scoring itself is computed
    on the encoded values either way, so this only affects what's shown,
    never what's measured.
    """
    try:
        preds = fitted_pipeline.predict(X_test)
    except Exception as e:  # noqa: BLE001
        return FinalEvaluationResult(
            pipeline_name=pipeline_name, model_name=model_name, task_type=task_type,
            n_test=len(X_test), error=f"{type(e).__name__}: {e}",
        )

    if task_type == TaskType.REGRESSION:
        residuals = np.asarray(y_test) - preds
        abs_err = np.abs(residuals)
        worst_idx = np.argsort(abs_err)[::-1][:n_worst]
        y_test_arr = np.asarray(y_test)
        worst = [
            {
                "index": int(i),
                "actual": float(y_test_arr[i]),
                "predicted": float(preds[i]),
                "abs_error": float(abs_err[i]),
            }
            for i in worst_idx
        ]
        reg = RegressionEvaluation(
            r2=float(r2_score(y_test, preds)),
            rmse=float(root_mean_squared_error(y_test, preds)),
            mae=float(mean_absolute_error(y_test, preds)),
            mape=float(mean_absolute_percentage_error(y_test, preds)),
            residual_mean=float(np.mean(residuals)),
            residual_std=float(np.std(residuals)),
            worst_errors=worst,
            rmsle=_rmsle(y_test, preds),
        )
        return FinalEvaluationResult(
            pipeline_name=pipeline_name, model_name=model_name, task_type=task_type,
            n_test=len(X_test), regression=reg,
        )

    # classification
    labels = sorted(np.unique(y_test).tolist())
    display_labels = labels
    if label_encoder is not None:
        try:
            display_labels = [str(v) for v in label_encoder.inverse_transform(labels)]
        except Exception:
            display_labels = labels  # decoding failed for some reason — show raw values rather than crash
    roc_auc = None
    try:
        if hasattr(fitted_pipeline, "predict_proba"):
            proba = fitted_pipeline.predict_proba(X_test)
            roc_auc = float(roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")) \
                if proba.shape[1] > 2 else float(roc_auc_score(y_test, proba[:, 1]))
    except Exception:
        roc_auc = None  # not every estimator/label setup supports this — leave honestly absent

    clf = ClassificationEvaluation(
        accuracy=float(accuracy_score(y_test, preds)),
        # macro averaging — see experiment.py's CLASSIFICATION_METRICS for
        # why weighted would be misleading here on imbalanced data.
        precision=float(precision_score(y_test, preds, average="macro", zero_division=0)),
        recall=float(recall_score(y_test, preds, average="macro", zero_division=0)),
        f1=float(f1_score(y_test, preds, average="macro", zero_division=0)),
        f2=float(fbeta_score(y_test, preds, beta=2, average="macro", zero_division=0)),
        roc_auc=roc_auc,
        labels=display_labels,
        confusion=confusion_matrix(y_test, preds, labels=labels).tolist(),
    )
    return FinalEvaluationResult(
        pipeline_name=pipeline_name, model_name=model_name, task_type=task_type,
        n_test=len(X_test), classification=clf,
    )


class PrecomputedPredictions:
    """Stands in for a fitted model when the predictions were already made out-of-fold, so the
    same scoring code produces the same metrics. Only exposes predict_proba when it has them."""

    def __init__(self, pred, proba=None):
        self._pred = np.asarray(pred)
        if proba is not None:
            self._proba = np.asarray(proba)
            self.predict_proba = lambda X: self._proba

    def predict(self, X):
        return self._pred
