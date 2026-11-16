"""How much should one held-out score be trusted?

A single test split of a few hundred rows moves by several points just from which rows
landed in it. Two helpers make that visible: a confidence interval for the score
already measured, and a repeat check that refits the chosen pipeline on several
different splits (no new search, same model and settings)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from .evaluation import evaluate_on_held_out_test
from .reasoner import TaskType


def accuracy_interval(accuracy: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% interval for an accuracy measured on n rows."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1 + z * z / n
    centre = (accuracy + z * z / (2 * n)) / denom
    half = z * np.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class StabilityResult:
    metric: str
    scores: list[float]
    other: dict[str, list[float]]

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores))

    @property
    def std(self) -> float:
        return float(np.std(self.scores, ddof=1)) if len(self.scores) > 1 else 0.0


def stability_check(pipeline, X_train, y_train, X_test, y_test, task: TaskType,
                    label_encoder=None, n_splits: int = 5, test_size: float = 0.2,
                    seed: int = 7) -> StabilityResult:
    """Refit a clone of `pipeline` on `n_splits` fresh splits of all labelled rows."""
    X = pd.concat([X_train, X_test]).reset_index(drop=True)
    y = pd.concat([pd.Series(np.asarray(y_train)), pd.Series(np.asarray(y_test))]).reset_index(drop=True)
    if task == TaskType.CLASSIFICATION and y.value_counts().min() >= 2:
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
        parts = splitter.split(X, y)
    else:
        parts = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed).split(X)
    primary, other = [], {"f1": []} if task == TaskType.CLASSIFICATION else {"rmse": []}
    for tr, te in parts:
        model = clone(pipeline)
        model.fit(X.iloc[tr], y.iloc[tr])
        res = evaluate_on_held_out_test("check", "check", model, X.iloc[te], y.iloc[te], task,
                                        label_encoder=label_encoder)
        if not res.ok:
            continue
        if task == TaskType.CLASSIFICATION:
            primary.append(res.classification.accuracy)
            other["f1"].append(res.classification.f1)
        else:
            primary.append(res.regression.r2)
            other["rmse"].append(res.regression.rmse)
    if not primary:
        raise RuntimeError("Every repeat failed to fit.")
    return StabilityResult("accuracy" if task == TaskType.CLASSIFICATION else "r2", primary, other)
