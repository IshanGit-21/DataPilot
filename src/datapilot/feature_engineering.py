"""
Stage 3 — Feature Engineering.

Two real, sklearn-compatible transformers — not a stand-in for this stage,
the actual thing:

  - SkewCorrector: Yeo-Johnson power-transforms numeric columns whose skew
    exceeds a threshold, computed fresh on whatever fold .fit() receives.
  - TopKInteractions: adds pairwise product features for the top-k numeric
    columns most associated with the target, also scored fresh per fold.

Both compute their own "what to do" decision inside fit() on the data
they're actually given — never from a pre-split, whole-dataset profile —
so which columns get transformed/interacted is itself leakage-safe, not
just the transform's fitted parameters. That distinction matters: a
transformer that decides "skew this column" using full-dataset statistics
computed before the train/test split would leak structural information
about the test set into pipeline construction, even if the transform's
own parameters were fit correctly afterward.

TopKInteractions is bounded to a small k specifically so it can't produce
a combinatorial explosion of near-useless interaction columns on wide
datasets — 3 columns means 3 new interaction features, not (n choose 2).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import skew as scipy_skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif, f_regression
from sklearn.preprocessing import PowerTransformer

from .reasoner import TaskType


class SkewCorrector(BaseEstimator, TransformerMixin):
    """Yeo-Johnson (not log/Box-Cox) specifically because it handles zero
    and negative values without extra guarding — real datasets routinely
    have both in otherwise-skewed numeric columns.
    """

    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.columns_ = list(X.columns)
        skews = {
            c: float(scipy_skew(X[c].dropna())) if X[c].dropna().nunique() > 1 else 0.0
            for c in self.columns_
        }
        self.flagged_ = [c for c in self.columns_ if abs(skews[c]) > self.skew_threshold]
        if self.flagged_:
            self.transformer_ = PowerTransformer(method="yeo-johnson")
            self.transformer_.fit(X[self.flagged_])
        return self

    def transform(self, X):
        X = self._as_frame(X).copy()
        if self.flagged_:
            X[self.flagged_] = self.transformer_.transform(X[self.flagged_])
        return X[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)

    @staticmethod
    def _as_frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


class TopKInteractions(BaseEstimator, TransformerMixin):
    def __init__(self, task_type: TaskType, k: int = 3):
        self.task_type = task_type
        self.k = k

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.columns_ = list(X.columns)
        top = self.columns_[: self.k]
        if y is not None and len(self.columns_) >= 2:
            score_fn = f_classif if self.task_type == TaskType.CLASSIFICATION else f_regression
            try:
                scores, _ = score_fn(X.to_numpy(dtype=float), np.asarray(y))
                order = np.argsort(np.nan_to_num(scores, nan=-1.0))[::-1]
                top = [self.columns_[i] for i in order[: self.k]]
            except Exception:
                pass  # fall back to the first k columns rather than crash fitting
        # constant columns would produce a meaningless (all-equal) interaction
        self.top_ = [c for c in top if X[c].nunique(dropna=True) > 1]
        self.pairs_ = list(itertools.combinations(self.top_, 2))
        return self

    def transform(self, X):
        X = self._as_frame(X)
        out = X[self.columns_].copy()
        for a, b in self.pairs_:
            out[f"{a}_x_{b}"] = X[a] * X[b]
        return out.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        names = list(self.columns_) + [f"{a}_x_{b}" for a, b in self.pairs_]
        return np.array(names)

    @staticmethod
    def _as_frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
