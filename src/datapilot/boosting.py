"""Gradient boosting that picks its own number of trees on big tables.

A boosted model tuned on a 40,000-row sample is then refit on hundreds of thousands of rows
with the same number of trees. More rows can use more trees, and fixed counts either stop
short or overfit. For the FINAL fit only, a boosted model is wrapped so that it:

  * holds back 10% of its own training rows,
  * adds trees until that slice stops improving (early stopping),
  * keeps the best count.

Small tables are untouched (below MIN_ROWS the wrapper fits the model exactly as it was).
The held-out test rows or cross-check rows are never involved: the 10% is cut from the rows
the model was being trained on anyway.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import train_test_split

MIN_ROWS = 30_000     # below this the fixed 300 trees are fine and the extra split costs more than it gives
MAX_TREES = 1500
PATIENCE = 40
VALID_SHARE = 0.10


def _kind(est) -> str | None:
    n = type(est).__name__
    if "LGBM" in n:
        return "lgbm"
    if "XGB" in n:
        return "xgb"
    if "CatBoost" in n:
        return "cat"
    return None


def _count_param(est, kind: str) -> str:
    return "iterations" if kind == "cat" else "n_estimators"


class _EarlyStopped:
    def __init__(self, estimator=None, max_trees=MAX_TREES, patience=PATIENCE, random_state=42):
        self.estimator = estimator
        self.max_trees = max_trees
        self.patience = patience
        self.random_state = random_state

    def set_params(self, **params):
        """Settings meant for the boosted model inside (n_estimators, learning_rate...) are passed on to it."""
        own = {k: v for k, v in params.items() if k.split("__")[0] in ("estimator", "max_trees", "patience", "random_state")}
        rest = {k: v for k, v in params.items() if k not in own}
        if own:
            super().set_params(**own)
        if rest:
            self.estimator.set_params(**rest)
        return self

    def _fit(self, X, y, classify: bool):
        kind = _kind(self.estimator)
        est = clone(self.estimator)
        if len(X) < MIN_ROWS or kind is None:
            self.estimator_ = est.fit(X, y)
            self.n_trees_ = None
            return self
        y_arr = np.asarray(y)
        strat = y_arr if classify and min(np.bincount(y_arr.astype(int))) >= 2 else None
        try:
            Xt, Xv, yt, yv = train_test_split(X, y_arr, test_size=VALID_SHARE, random_state=self.random_state, stratify=strat)
        except ValueError:
            Xt, Xv, yt, yv = train_test_split(X, y_arr, test_size=VALID_SHARE, random_state=self.random_state)
        warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
        cap = max(int(est.get_params().get(_count_param(est, kind), 0) or 0), self.max_trees)
        est.set_params(**{_count_param(est, kind): cap})
        if kind == "lgbm":
            import lightgbm as lgb
            est.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(self.patience, verbose=False)])
            self.n_trees_ = int(getattr(est, "best_iteration_", 0) or cap)
        elif kind == "xgb":
            est.set_params(early_stopping_rounds=self.patience)
            est.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
            self.n_trees_ = int(getattr(est._inner if hasattr(est, "_inner") else est, "best_iteration", cap - 1)) + 1
        else:
            est.set_params(early_stopping_rounds=self.patience)
            est.fit(Xt, yt, eval_set=(Xv, yv))
            self.n_trees_ = int(est.get_best_iteration() or cap) + 1
        self.estimator_ = est
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "estimator_")

    def predict(self, X):
        return self.estimator_.predict(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_

    @property
    def n_features_in_(self):
        return self.estimator_.n_features_in_


class EarlyStoppedClassifier(ClassifierMixin, _EarlyStopped, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y))
        return self._fit(X, y, classify=True)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class EarlyStoppedRegressor(RegressorMixin, _EarlyStopped, BaseEstimator):
    def fit(self, X, y):
        return self._fit(X, y, classify=False)


def wrap_boosters(est):
    """Return `est` with every LightGBM / XGBoost / CatBoost inside it wrapped (ensembles included)."""
    if _kind(est) is not None:
        cls = EarlyStoppedClassifier if hasattr(est, "predict_proba") else EarlyStoppedRegressor
        return cls(estimator=est)
    subs = getattr(est, "estimators", None)
    if subs and hasattr(est, "set_params"):
        try:
            est.set_params(**{name: wrap_boosters(sub) for name, sub in subs if sub != "drop"})
        except Exception:  # noqa: BLE001 - an ensemble shape we do not know: leave it as it was
            pass
        return est
    inner = getattr(est, "estimator", None)       # e.g. the threshold-tuning wrapper
    if inner is not None and _kind(inner) is not None:
        est.set_params(estimator=wrap_boosters(inner))
    return est
