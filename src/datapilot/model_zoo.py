"""
Stage 4 — Modeling.

Registers the real candidate models per task type: a dummy baseline (always
first, so every other model has a floor to beat), then linear/logistic
regression, a decision tree, a random forest, and — when the optional
competition-grade libraries are actually installed — XGBoost and LightGBM.

Honesty over simulation: if xgboost/lightgbm aren't installed in this
environment, they are silently OMITTED from the list, not faked with a
stand-in. `available_model_names()` / `unavailable_libraries()` let the UI
tell the user the truth about what actually ran.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .boosting import wrap_boosters
from .reasoner import TaskType

try:
    from xgboost import XGBClassifier as _XGBClassifier
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True

    class XGBClassifier(_XGBClassifier):
        """XGBoost insists on labels 0..k-1 in every single fit. A CV fold that misses a rare
        class breaks that, so this wrapper encodes labels itself and fits a plain XGBClassifier
        inside. Parameters, cloning and tuning behave exactly as for XGBClassifier."""

        def fit(self, X, y, **kwargs):
            labels = np.unique(np.asarray(y))
            self._inner = _XGBClassifier(**self.get_params())
            self._inner.fit(X, np.searchsorted(labels, np.asarray(y)), **kwargs)
            self._label_values = labels
            return self

        def __sklearn_is_fitted__(self):
            return hasattr(self, "_inner")

        def predict(self, X, **kwargs):
            return self._label_values[np.asarray(self._inner.predict(X, **kwargs)).astype(int)]

        def predict_proba(self, X, **kwargs):
            return self._inner.predict_proba(X, **kwargs)

        @property
        def n_features_in_(self):
            return self._inner.n_features_in_

        @property
        def classes_(self):
            return self._label_values

        @property
        def feature_importances_(self):
            return self._inner.feature_importances_

except ImportError:
    _HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    _HAS_CATBOOST = True
except ImportError:
    _HAS_CATBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


@dataclass
class ModelSpec:
    name: str
    build: Callable[[], BaseEstimator]
    is_baseline: bool = False
    notes: str = ""


RANDOM_STATE = 42


# Backup models: different inductive biases that keep a run useful when the strong tree
# models misbehave on an odd dataset, and give ensembles more variety to draw on. The
# row limits keep the slow ones (distance-based, neural) out of large runs.
BACKUP_ROW_LIMITS = {"KNN": 20_000, "Neural Net (MLP)": 30_000, "AdaBoost": 60_000}


BIG_DATA_ROWS = 10_000   # above this, forests get fewer trees: each tree is deep and costs seconds
BIG_TREES = 100


def get_models(task_type: TaskType, backups: bool = False, n_rows: int | None = None) -> list[ModelSpec]:
    trees = BIG_TREES if (n_rows or 0) > BIG_DATA_ROWS else 300
    if task_type == TaskType.CLASSIFICATION:
        models = _classification_models(trees)
        extra = _classification_backups()
    elif task_type == TaskType.REGRESSION:
        models = _regression_models(trees)
        extra = _regression_backups()
    else:
        return []  # ANALYTICS — no downstream model by design
    if backups:
        models += [m for m in extra
                   if n_rows is None or n_rows <= BACKUP_ROW_LIMITS.get(m.name, float("inf"))]
    return models


def _classification_backups() -> list[ModelSpec]:
    return [
        ModelSpec("KNN", lambda: KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=1),
                  notes="Backup: judges a row by its nearest training rows; needs scaled inputs."),
        ModelSpec("AdaBoost", lambda: AdaBoostClassifier(n_estimators=200, learning_rate=0.5,
                                                         random_state=RANDOM_STATE),
                  notes="Backup: boosted shallow trees, a different bias from gradient boosting."),
        ModelSpec("Neural Net (MLP)", lambda: MLPClassifier(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=RANDOM_STATE),
                  notes="Backup: small neural network; smooth decision surface."),
    ]


def _regression_backups() -> list[ModelSpec]:
    return [
        ModelSpec("Ridge", lambda: Ridge(alpha=1.0), notes="Backup: linear model that survives collinear inputs."),
        ModelSpec("Elastic Net", lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=RANDOM_STATE),
                  notes="Backup: sparse linear model."),
        ModelSpec("KNN", lambda: KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=1),
                  notes="Backup: judges a row by its nearest training rows; needs scaled inputs."),
        ModelSpec("AdaBoost", lambda: AdaBoostRegressor(n_estimators=200, learning_rate=0.5,
                                                        random_state=RANDOM_STATE)),
        ModelSpec("Neural Net (MLP)", lambda: MLPRegressor(
            hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=RANDOM_STATE)),
    ]


def _classification_models(trees: int = 300) -> list[ModelSpec]:
    models = [
        ModelSpec(
            "Baseline (most frequent)",
            lambda: DummyClassifier(strategy="most_frequent"),
            is_baseline=True,
            notes="Predicts the majority class every time — the floor every real model must beat.",
        ),
        ModelSpec(
            "Logistic Regression",
            lambda: LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            notes="class_weight=balanced — a real, cheap partial answer to class imbalance.",
        ),
        ModelSpec(
            "Decision Tree",
            lambda: DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
        ),
        ModelSpec(
            "Random Forest",
            lambda: RandomForestClassifier(
                n_estimators=trees, max_depth=None, class_weight="balanced",
                n_jobs=1, random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "Extra Trees",
            lambda: ExtraTreesClassifier(
                n_estimators=trees, max_depth=None, class_weight="balanced",
                n_jobs=1, random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "Hist Gradient Boosting",
            lambda: HistGradientBoostingClassifier(
                learning_rate=0.08, max_iter=250, class_weight="balanced", random_state=RANDOM_STATE,
            ),
            notes="scikit-learn's native gradient boosting; handles missing values itself.",
        ),
    ]
    if _HAS_XGBOOST:
        models.append(ModelSpec(
            "XGBoost",
            lambda: XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1,
            ),
        ))
    if _HAS_CATBOOST:
        models.append(ModelSpec(
            "CatBoost",
            lambda: CatBoostClassifier(iterations=300, depth=6, learning_rate=0.08, random_seed=RANDOM_STATE,
                                       thread_count=1, verbose=0, allow_writing_files=False),
            notes="Ordered boosting: a different bias from XGBoost/LightGBM, strong on small and noisy tables.",
        ))
    if _HAS_LIGHTGBM:
        models.append(ModelSpec(
            "LightGBM",
            lambda: LGBMClassifier(
                n_estimators=300, max_depth=-1, learning_rate=0.08,
                random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
            ),
        ))
    return models


def _regression_models(trees: int = 300) -> list[ModelSpec]:
    models = [
        ModelSpec(
            "Baseline (mean)",
            lambda: DummyRegressor(strategy="mean"),
            is_baseline=True,
            notes="Predicts the training mean every time — the floor every real model must beat.",
        ),
        ModelSpec("Linear Regression", lambda: LinearRegression()),
        ModelSpec(
            "Decision Tree",
            lambda: DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        ),
        ModelSpec(
            "Random Forest",
            lambda: RandomForestRegressor(
                n_estimators=trees, max_depth=None, n_jobs=1, random_state=RANDOM_STATE,
            ),
        ),
        ModelSpec(
            "Extra Trees",
            lambda: ExtraTreesRegressor(n_estimators=trees, max_depth=None, n_jobs=1, random_state=RANDOM_STATE),
        ),
        ModelSpec(
            "Hist Gradient Boosting",
            lambda: HistGradientBoostingRegressor(learning_rate=0.08, max_iter=250, random_state=RANDOM_STATE),
            notes="scikit-learn's native gradient boosting; handles missing values itself.",
        ),
    ]
    if _HAS_XGBOOST:
        models.append(ModelSpec(
            "XGBoost",
            lambda: XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.08,
                random_state=RANDOM_STATE, n_jobs=1,
            ),
        ))
    if _HAS_CATBOOST:
        models.append(ModelSpec(
            "CatBoost",
            lambda: CatBoostRegressor(iterations=300, depth=6, learning_rate=0.08, random_seed=RANDOM_STATE,
                                       thread_count=1, verbose=0, allow_writing_files=False),
            notes="Ordered boosting: a different bias from XGBoost/LightGBM, strong on small and noisy tables.",
        ))
    if _HAS_LIGHTGBM:
        models.append(ModelSpec(
            "LightGBM",
            lambda: LGBMRegressor(
                n_estimators=300, max_depth=-1, learning_rate=0.08,
                random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
            ),
        ))
    return models


def use_all_cores(estimator):
    """Zoo models are built with n_jobs=1 because cross-validation already runs
    folds in parallel; nested threads oversubscribe the CPU (a 142-row
    LightGBM fit took 67 s that way). For a single final fit there is no outer
    parallelism, so hand the estimator every core."""
    # LogisticRegression's n_jobs has had no effect since scikit-learn 1.8 and warns
    if "n_jobs" in estimator.get_params() and type(estimator).__name__ != "LogisticRegression":
        estimator.set_params(n_jobs=-1)
    elif "thread_count" in estimator.get_params():
        estimator.set_params(thread_count=-1)
    return wrap_boosters(estimator)   # big tables: the boosted models choose their own number of trees


def unavailable_libraries() -> list[str]:
    """What's missing from this environment, so the UI can say so honestly
    instead of silently training fewer models than advertised."""
    missing = []
    if not _HAS_XGBOOST:
        missing.append("xgboost")
    if not _HAS_LIGHTGBM:
        missing.append("lightgbm")
    if not _HAS_CATBOOST:
        missing.append("catboost")
    return missing
