"""Result diagnostics — the system's own sanity checks on what it produced.

Two kinds, both computed from real data/results (no heuristics on names):
  * leakage suspects: an input that on its own (almost) reproduces the target
  * result warnings: weak margin over the baseline, tiny test set, failures
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .reasoner import TaskType

NUMERIC_CORR_THRESHOLD = 0.98
CATEGORY_PURITY_THRESHOLD = 0.99
MAX_CATEGORIES_FOR_PURITY = 50
MIN_ROWS_PER_CATEGORY = 5
BASELINE_MARGIN_MIN = 0.02
SMALL_TEST_ROWS = 100


@dataclass
class LeakageSuspect:
    column: str
    evidence: str


def detect_leakage_suspects(X: pd.DataFrame, y, task_type: TaskType) -> list[LeakageSuspect]:
    """Run on TRAINING data only. A column is suspicious if alone it predicts
    the target almost perfectly, which on real data usually means it is a
    copy/derivative of the target or is only known after the outcome."""
    y = pd.Series(np.asarray(y), index=X.index)
    out: list[LeakageSuspect] = []
    for col in X.columns:
        s = X[col]
        valid = s.notna() & y.notna()
        if valid.sum() < 30:
            continue
        s_v, y_v = s[valid], y[valid]
        if pd.api.types.is_numeric_dtype(s_v) and not pd.api.types.is_bool_dtype(s_v):
            if s_v.nunique() < 2:
                continue
            if pd.api.types.is_numeric_dtype(y_v):
                r = abs(s_v.rank().corr(y_v.rank()))
                if pd.notna(r) and r >= NUMERIC_CORR_THRESHOLD:
                    out.append(LeakageSuspect(col, f"rank correlation with target = {r:.3f}"))
                    continue
        if task_type == TaskType.CLASSIFICATION:
            n_cat = s_v.nunique()
            if 2 <= n_cat <= MAX_CATEGORIES_FOR_PURITY and len(s_v) / n_cat >= MIN_ROWS_PER_CATEGORY:
                tab = pd.crosstab(s_v.astype(str), y_v)
                purity = tab.max(axis=1).sum() / tab.values.sum()
                base = y_v.value_counts(normalize=True).max()
                if purity >= CATEGORY_PURITY_THRESHOLD and purity - base >= 0.02:
                    out.append(LeakageSuspect(
                        col, f"each value of this column maps to one class {purity:.1%} of the time "
                             f"(majority-class rate is {base:.1%})"))
    return out


_SENSITIVE = re.compile(
    r"(^|_)(sex|gender|race|ethnic\w*|religio\w*|nationalit\w*|foreign\w*|marital\w*|disabilit\w*|caste|orientation)($|_)",
    re.IGNORECASE)


def sensitive_input_columns(columns) -> list[str]:
    """Inputs whose NAME marks a protected or sensitive attribute. A name-based
    flag only: it asks the user to review, it does not decide anything."""
    out = []
    for c in columns:
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(c))
        norm = re.sub(r"[^A-Za-z0-9]+", "_", spaced).strip("_")
        if _SENSITIVE.search(norm):
            out.append(c)
    return out


def assess_results(leaderboard_entries, evaluation_result=None) -> list[str]:
    """Plain-language warnings about the reliability of the outcome."""
    w: list[str] = []
    ok = [e for e in leaderboard_entries if e.ok]
    if not ok:
        return ["No model trained successfully."]
    failed = [e for e in leaderboard_entries if not e.ok]
    if failed:
        w.append(f"{len(failed)} model attempt(s) failed and were left out.")
    best = max(ok, key=lambda e: e.primary_mean)
    base = [e for e in ok if e.is_baseline]
    if base:
        b = max(e.primary_mean for e in base)
        if best.primary_mean - b < BASELINE_MARGIN_MIN:  # f1 / r2 are both bounded, so absolute margin is meaningful
            w.append(
                f"The best model ({best.primary_mean:.3f}) barely beats the naive baseline ({b:.3f}) "
                "in the practice rounds; the columns may say little about this target.")
    if best.primary_std and abs(best.primary_mean) > 1e-9 and best.primary_std / abs(best.primary_mean) > 0.15:
        w.append("Scores jump around between practice rounds, so the model ranking is not very steady.")
    ev = evaluation_result
    if ev is not None and not ev.error:
        if ev.n_test < SMALL_TEST_ROWS:
            w.append(f"Only {ev.n_test} rows were used for the final check, so the final score can move a lot.")
        if ev.classification and ev.classification.roc_auc is None:
            w.append("One answer never appears in the final-check rows, so one of the scores could not be worked out.")
    return w
