"""Accuracy boosters that sit on top of the model zoo.

* Ensemble: soft-voting (classification) / averaging (regression) of the top
  distinct model types. Averaging models with different inductive biases is
  one of the most reliable accuracy gains in tabular ML.
* Tuned decision threshold (binary classification): the 0.5 cut-off is rarely
  best for the metric we optimise on imbalanced data; the threshold is chosen
  on out-of-fold predictions inside fit(), so cross-validating this estimator
  is honest (the threshold is re-chosen on each training fold only).

Both are ordinary ModelSpecs with self-describing names, so every consumer
(leaderboard, tuning, final fit, exported project) rebuilds them through
resolve_model_spec(name, task_type) instead of special-casing them.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .model_zoo import ModelSpec, get_models
from .reasoner import TaskType

ENSEMBLE_PREFIX = "Ensemble: "
STACK_PREFIX = "Stacking: "
THRESHOLD_SUFFIX = " + threshold"
LOG_TARGET_SUFFIX = " + log target"
BOOSTER_SUFFIXES = (THRESHOLD_SUFFIX, LOG_TARGET_SUFFIX)
THRESHOLD_SCORERS = ("f1", "accuracy", "f2", "precision", "recall")
ENSEMBLE_SIZE = 3


def inv_log1p(z):
    """Inverse of log(1+y), never below zero (a count or price cannot be negative)."""
    return np.clip(np.expm1(z), 0, None)


def log_target_model(estimator):
    from sklearn.compose import TransformedTargetRegressor
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=inv_log1p)


def _score_threshold(scoring: str, y, pred) -> float:
    from sklearn.metrics import (
        accuracy_score,
        fbeta_score,
        precision_score,
        recall_score,
    )
    if scoring == "accuracy":
        return accuracy_score(y, pred)
    if scoring == "f2":
        return fbeta_score(y, pred, beta=2, average="macro", zero_division=0)
    if scoring == "precision":
        return precision_score(y, pred, average="macro", zero_division=0)
    if scoring == "recall":
        return recall_score(y, pred, average="macro", zero_division=0)
    return f1_score(y, pred, average="macro")


class ThresholdTunedClassifier(BaseEstimator, ClassifierMixin):
    """Binary classifier wrapper that picks the probability cut-off maximising
    macro-F1 on out-of-fold training predictions."""

    def __init__(self, estimator=None, cv=3, random_state=42, scoring="f1"):
        self.scoring = scoring
        self.estimator = estimator
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("ThresholdTunedClassifier supports binary targets only")
        yb = (y == self.classes_[1]).astype(int)
        self.threshold_ = 0.5
        if min(np.bincount(yb)) >= self.cv:
            skf = StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state)
            oof = cross_val_predict(clone(self.estimator), X, yb, cv=skf, method="predict_proba")[:, 1]
            best, best_score = 0.5, -1.0
            for t in np.arange(0.05, 0.951, 0.01):
                s = _score_threshold(self.scoring, yb, (oof >= t).astype(int))
                if s > best_score + 1e-9 or (abs(s - best_score) <= 1e-9 and abs(t - 0.5) < abs(best - 0.5)):
                    best, best_score = float(t), s
            self.threshold_ = best
        self.estimator_ = clone(self.estimator).fit(X, yb)
        return self

    def predict_proba(self, X):
        p1 = self.estimator_.predict_proba(X)[:, 1]
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)]


def _make_ensemble_spec(component_names: list[str], task_type: TaskType) -> ModelSpec:
    by_name = {m.name: m for m in get_models(task_type, backups=True)}
    name = ENSEMBLE_PREFIX + " + ".join(component_names)

    def build():
        parts = [(f"m{i}", by_name[n].build()) for i, n in enumerate(component_names)]
        if task_type == TaskType.CLASSIFICATION:
            return VotingClassifier(parts, voting="soft", n_jobs=1)
        return VotingRegressor(parts, n_jobs=1)

    return ModelSpec(name, build, notes="Averages the top distinct model types.")


def _make_stack_spec(component_names: list[str], task_type: TaskType) -> ModelSpec:
    """Out-of-fold predictions of the base models feed a small linear meta-model, which
    learns how much to trust each one (unlike plain averaging, which trusts them equally)."""
    by_name = {m.name: m for m in get_models(task_type, backups=True)}
    name = STACK_PREFIX + " + ".join(component_names)

    def build():
        parts = [(f"m{i}", by_name[n].build()) for i, n in enumerate(component_names)]
        if task_type == TaskType.CLASSIFICATION:
            from sklearn.linear_model import LogisticRegression
            return StackingClassifier(parts, final_estimator=LogisticRegression(max_iter=2000),
                                      cv=3, stack_method="predict_proba", n_jobs=1)
        from sklearn.linear_model import RidgeCV
        return StackingRegressor(parts, final_estimator=RidgeCV(), cv=3, n_jobs=1)

    return ModelSpec(name, build, notes="Meta-model learns how much to trust each base model.")


def resolve_model_spec(name: str, task_type: TaskType) -> ModelSpec:
    """Rebuild any ModelSpec from its name: zoo model, ensemble, or threshold-tuned."""
    if name.endswith(THRESHOLD_SUFFIX):
        base = resolve_model_spec(name[: -len(THRESHOLD_SUFFIX)], task_type)
        return ModelSpec(name, lambda: ThresholdTunedClassifier(base.build()),
                         notes="Decision threshold tuned on out-of-fold predictions.")
    if name.endswith(LOG_TARGET_SUFFIX):
        base = resolve_model_spec(name[: -len(LOG_TARGET_SUFFIX)], task_type)
        from sklearn.compose import TransformedTargetRegressor
        return ModelSpec(name, lambda: TransformedTargetRegressor(
            regressor=base.build(), func=np.log1p, inverse_func=inv_log1p),
            notes="Fits on log(1+y) and converts predictions back; helps right-skewed targets.")
    if name.startswith(STACK_PREFIX):
        return _make_stack_spec(name[len(STACK_PREFIX):].split(" + "), task_type)
    if name.startswith(ENSEMBLE_PREFIX):
        return _make_ensemble_spec(name[len(ENSEMBLE_PREFIX):].split(" + "), task_type)
    for m in get_models(task_type, backups=True):
        if m.name == name:
            return m
    raise KeyError(f"Unknown model: {name!r}")


def extra_specs_for(entries, task_type: TaskType, is_binary: bool, y=None, primary: str = "f1",
                    n_rows: int | None = None):
    """Given scored leaderboard entries, return [(pipeline_name, ModelSpec)] for
    the accuracy boosters worth scoring. Only uses plain (non-SMOTE) entries."""
    known = {m.name for m in get_models(task_type, backups=True)}
    plain = [e for e in entries if e.ok and not e.uses_smote and not e.is_baseline and e.model_name in known]
    if not plain:
        return []
    best = max(plain, key=lambda e: e.primary_mean)
    out = []
    on_best = {}
    for e in plain:
        if e.pipeline_name == best.pipeline_name and not e.model_name.startswith((ENSEMBLE_PREFIX, STACK_PREFIX)) \
                and not e.model_name.endswith(BOOSTER_SUFFIXES):
            on_best[e.model_name] = e
    ranked = sorted(on_best.values(), key=lambda e: -e.primary_mean)
    # drop weak members: a clearly worse model drags an average down
    ranked = [e for e in ranked if e.primary_mean >= ranked[0].primary_mean - 0.05]
    names = [e.model_name for e in ranked[:ENSEMBLE_SIZE]]
    if len(names) >= 2:
        out.append((best.pipeline_name, _make_ensemble_spec(names, task_type)))
        # stacking only when there is enough data for the extra inner CV to be stable
        if len(names) >= 3 and (n_rows is None or n_rows >= 500):
            out.append((best.pipeline_name, _make_stack_spec(names, task_type)))
    # threshold tuning only helps metrics that depend on the cut-off (AUC does not)
    if task_type == TaskType.CLASSIFICATION and is_binary and primary in THRESHOLD_SCORERS:
        out.append((best.pipeline_name, resolve_model_spec(best.model_name + THRESHOLD_SUFFIX, task_type)))
    if task_type == TaskType.REGRESSION and y is not None:
        import pandas as pd
        ys = pd.Series(np.asarray(y, dtype=float))
        if ys.min() >= 0 and ys.skew() > 1.0:
            out.append((best.pipeline_name, resolve_model_spec(best.model_name + LOG_TARGET_SUFFIX, task_type)))
    return out
