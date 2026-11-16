"""Kaggle-style submission file from a fitted pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_submission(fitted, test_df: pd.DataFrame, sample_submission: pd.DataFrame | None,
                    feature_columns: list[str], label_encoder=None,
                    id_column: str | None = None, target_column: str | None = None):
    """Returns (submission_df, note).

    Format comes from `sample_submission` when given: first column = id, second = the
    prediction column. If that column holds non-integer floats the competition wants
    probabilities, so the positive-class probability is written; otherwise predicted labels.
    """
    missing = [c for c in feature_columns if c not in test_df.columns]
    if missing:
        raise ValueError(f"The test file is missing training columns: {missing[:8]}")
    if sample_submission is not None and len(sample_submission.columns) >= 2:
        id_column = id_column or sample_submission.columns[0]
        target_column = target_column or sample_submission.columns[1]
    if id_column is None or id_column not in test_df.columns:
        if sample_submission is not None and len(sample_submission) == len(test_df) and id_column:
            ids = sample_submission[id_column].to_numpy()
        else:
            ids = np.arange(len(test_df))
            id_column = id_column or "id"
    else:
        ids = test_df[id_column].to_numpy()
    target_column = target_column or "prediction"

    X = test_df[feature_columns]
    want_proba = False
    if sample_submission is not None and target_column in sample_submission.columns:
        col = sample_submission[target_column]
        if pd.api.types.is_float_dtype(col) and not (col.dropna() % 1 == 0).all() or pd.api.types.is_float_dtype(col) and hasattr(fitted, "predict_proba") and label_encoder is not None \
                and len(label_encoder.classes_) == 2 and col.nunique() > 2:
            want_proba = True
    if want_proba and hasattr(fitted, "predict_proba"):
        proba = fitted.predict_proba(X)
        pred = proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
        note = "Wrote the probability of the positive class (the sample submission uses probabilities)."
    else:
        pred = fitted.predict(X)
        if label_encoder is not None:
            pred = label_encoder.inverse_transform(np.asarray(pred).astype(int))
        note = "Wrote predicted labels." if label_encoder is not None else "Wrote predicted values."
    sub = pd.DataFrame({id_column: ids, target_column: pred})
    if sample_submission is not None and len(sample_submission) != len(sub):
        note += f" Warning: sample submission has {len(sample_submission):,} rows but the test file produced {len(sub):,}."
    return sub, note


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame | None, test_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Plain sanity checks before upload: [(level, message)] with level ok / warn / error."""
    out: list[tuple[str, str]] = []
    if len(sub) != len(test_df):
        out.append(("error", f"{len(sub):,} predictions for {len(test_df):,} test rows."))
    else:
        out.append(("ok", f"{len(sub):,} rows, one per test row."))
    if sample is not None:
        if list(sub.columns) != list(sample.columns[:len(sub.columns)]):
            out.append(("error", f"Columns are {list(sub.columns)}; the sample has {list(sample.columns)}."))
        else:
            out.append(("ok", "Column names match the sample submission."))
        if len(sample) != len(sub):
            out.append(("warn", f"The sample has {len(sample):,} rows; this has {len(sub):,}."))
        elif sub.iloc[:, 0].astype(str).tolist() != sample.iloc[:, 0].astype(str).tolist():
            out.append(("warn", "Ids are in a different order than the sample submission."))
    if sub.iloc[:, 1].isna().any():
        out.append(("error", f"{int(sub.iloc[:, 1].isna().sum())} predictions are missing."))
    if sub.iloc[:, 0].duplicated().any():
        out.append(("warn", "Some ids appear more than once."))
    return out


def refit_on_all_rows(fitted, X_train, y_train, X_test, y_test):
    """Clone the winning pipeline and fit it on train + held-out rows together.

    For a competition file this is standard practice: the held-out score has already
    been measured, so the final model may use every labelled row."""
    from sklearn.base import clone
    model = clone(fitted)
    model.fit(pd.concat([X_train, X_test]), pd.concat([y_train, y_test]))
    return model
