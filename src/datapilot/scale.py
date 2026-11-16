"""Scale handling: search on a representative sample, refit the winner on everything."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .reasoner import TaskType

DEFAULT_SEARCH_ROWS = 20_000   # model search and tuning run on this many rows at most
CHECK_ROWS = 30_000        # rows cross-checked for the final score when no rows are held out


def subsample_for_search(X: pd.DataFrame, y, task_type: TaskType,
                         max_rows: int = DEFAULT_SEARCH_ROWS, random_state: int = 42):
    """Returns (X_s, y_s, note). Model selection and tuning do not need every row of a
    700k-row table; a stratified sample ranks the candidates the same way at a fraction of
    the cost. The chosen model is then refit on the FULL training set by the caller."""
    n = len(X)
    if n <= max_rows:
        return X, y, ""
    y = pd.Series(y, index=X.index)
    strat = y if (task_type == TaskType.CLASSIFICATION and int(y.value_counts().min()) >= 2
                  and max_rows >= y.nunique() * 2) else None
    try:
        Xs, _, ys, _ = train_test_split(X, y, train_size=max_rows, random_state=random_state, stratify=strat)
    except ValueError:  # a class too small to stratify at this sample size
        Xs, _, ys, _ = train_test_split(X, y, train_size=max_rows, random_state=random_state)
    return Xs, ys, (f"Model search ran on a {max_rows:,}-row stratified sample of the {n:,} training rows; "
                    f"the selected model is refit on all {n:,}.")
