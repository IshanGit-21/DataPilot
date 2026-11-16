"""Guided setup — after Stage 1 analyses the dataset, suggest which column is
the output (target), what task it implies, and which columns to leave out as
inputs. Pure heuristics over the real profile; the user always decides."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .profiler import DatasetProfile
from .reasoner import TaskType

_NAME_HINT = re.compile(
    r"(^|_|\b)(target|label|class|outcome|result|status|survived|churn|fraud|default|"
    r"satisfaction|price|saleprice|salary|cost|revenue|sales|score|rating|value|amount|charges|rent|quality|demand|yield|rings|medv|mpg|y)($|_|\b)",
    re.IGNORECASE,
)
CLASSIFICATION_MAX_UNIQUE = 20


@dataclass
class TargetSuggestion:
    column: str
    score: float
    reason: str
    task_type: TaskType


def infer_task_type(series: pd.Series) -> TaskType:
    s = series.dropna()
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        # integer-valued with few distinct values is a class label, not a quantity
        if s.nunique() > CLASSIFICATION_MAX_UNIQUE:
            return TaskType.REGRESSION
        return TaskType.CLASSIFICATION
    return TaskType.CLASSIFICATION


def suggest_targets(df: pd.DataFrame, profile: DatasetProfile, top_n: int = 5) -> list[TargetSuggestion]:
    cols = list(df.columns)
    out: list[TargetSuggestion] = []
    for pos, name in enumerate(cols):
        cp = profile.columns.get(name)
        if cp is None or cp.is_constant or cp.is_likely_identifier or cp.is_free_text or cp.is_datetime:
            continue
        if cp.missing_ratio > 0.5:
            continue
        score, why = 0.0, []
        if _NAME_HINT.search(name):
            score += 3; why.append("name looks like an outcome")
        if pos == len(cols) - 1:
            score += 2; why.append("last column")
        if pos == 0 and pd.api.types.is_integer_dtype(df[name].dropna()) and 2 <= cp.n_unique <= CLASSIFICATION_MAX_UNIQUE:
            score += 1.5; why.append("first column with few integer values (class label convention, e.g. UCI files)")
        task = infer_task_type(df[name])
        nun = cp.n_unique
        if 2 <= nun <= CLASSIFICATION_MAX_UNIQUE:
            score += 1.5; why.append(f"{nun} distinct values (class-like)")
        elif task == TaskType.REGRESSION:
            score += 0.5; why.append("continuous numeric")
        else:
            continue  # high-cardinality non-numeric cannot be a target
        out.append(TargetSuggestion(name, score, "; ".join(why), task))
    out.sort(key=lambda t: -t.score)
    return out[:top_n]


def suggest_input_exclusions(profile: DatasetProfile, target: str | None) -> dict[str, str]:
    """Columns that should be unticked by default as inputs, with the reason."""
    ex: dict[str, str] = {}
    for name, cp in profile.columns.items():
        if name == target:
            continue
        if cp.is_constant:
            ex[name] = "constant (one value)"
        elif cp.is_likely_identifier:
            ex[name] = "looks like an identifier"
    return ex
