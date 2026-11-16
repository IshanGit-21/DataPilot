"""Real-world data hygiene: inconsistent spellings, boolean labels written a
dozen ways, exact duplicate rows. Everything here is stateless or fit on the
training data only, so it is leakage-safe and reproduces at predict time."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_TRUE = {"1", "true", "t", "yes", "y"}
_FALSE = {"0", "false", "f", "no", "n"}


def category_key(value) -> str:
    """'Home & Garden', 'home and garden', 'HOME_AND_GARDEN' -> 'home and garden';
    'U.S.A' and 'USA' -> 'usa'."""
    s = str(value).strip().casefold().replace("&", " and ")
    s = re.sub(r"[._]+", "" if re.fullmatch(r"(?:[a-z]\.)+[a-z]?\.?", s) else " ", s)
    s = re.sub(r"[^\w]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def key_series(s: pd.Series) -> pd.Series:
    """Vectorised category_key; keeps missing values missing."""
    mask = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=object)
    if mask.any():
        uniq = pd.unique(s[mask])
        mapping = {u: category_key(u) for u in uniq}
        out[mask] = s[mask].map(mapping)
    return out


def n_unique_after_cleaning(s: pd.Series) -> int:
    return int(key_series(s).nunique(dropna=True))


_CURRENCY = re.compile(r"[\$£€¥₹\s,]")


def _parse_one(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return np.nan
    t = str(v).strip()
    neg = t.startswith("(") and t.endswith(")")
    t = _CURRENCY.sub("", t.strip("()")).rstrip("%")
    try:
        x = float(t)
    except ValueError:
        return np.nan
    return -x if neg else x


def parse_numeric_series(s: pd.Series) -> pd.Series:
    """'$1,234.50', '£82.32', '12%', '(5)' -> floats; anything unparseable -> NaN."""
    return pd.Series([_parse_one(v) for v in s.tolist()], index=s.index, dtype="float64")


def looks_numeric_text(s: pd.Series, min_rate: float = 0.9) -> bool:
    """A text column whose non-missing values are (almost) all numbers once currency
    symbols, thousands separators and percent signs are stripped."""
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    nn = s.dropna()
    if len(nn) < 10:
        return False
    if len(nn) > 5000:
        nn = nn.sample(5000, random_state=0)
    parsed = parse_numeric_series(nn)
    return bool(parsed.notna().mean() >= min_rate and parsed.nunique() > 2)


class RawCleaner(BaseEstimator, TransformerMixin):
    """First step of a pipeline: collapses spelling variants in the given
    string columns to one key form. Stateless, so train and predict behave the same."""

    def __init__(self, columns=None, numeric_columns=None):
        self.columns = columns
        self.numeric_columns = numeric_columns

    def fit(self, X, y=None):
        self.columns_ = list(self.columns or [])
        self.numeric_columns_ = list(self.numeric_columns or [])
        self.feature_names_in_ = np.array(list(X.columns)) if hasattr(X, "columns") else None
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.columns_:
            if c in X.columns:
                X[c] = key_series(X[c])
        for c in self.numeric_columns_:
            if c in X.columns:
                X[c] = parse_numeric_series(X[c])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.feature_names_in_, dtype=object)


def clean_class_target(y: pd.Series) -> tuple[pd.Series, str]:
    """Merge spelling variants of class labels. Boolean-looking labels
    (True/TRUE/Yes/Y/1 ...) become 0/1. Returns (cleaned, note); `note` is ''
    when nothing needed fixing."""
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_numeric_dtype(y):
        return y, ""
    keys = key_series(y)
    uniq_keys = set(keys.dropna().unique())
    n_raw = int(y.nunique(dropna=True))
    if uniq_keys and uniq_keys <= (_TRUE | _FALSE) and len(uniq_keys) >= 2 and n_raw > 2:
        out = keys.map(lambda k: np.nan if pd.isna(k) else (1 if k in _TRUE else 0))
        return out.astype("Int64").astype(object).where(y.notna()), (
            f"Target had {n_raw} spellings of a yes/no label; mapped to 1 (true/yes/y/1) and 0 (false/no/n/0).")
    if len(uniq_keys) == 2 and n_raw == 2:
        return y, ""  # already two clean labels (e.g. yes/no): keep the readable names
    if len(uniq_keys) < n_raw:
        # canonical spelling per group = the most frequent original spelling
        canon = (pd.DataFrame({"k": keys, "v": y}).dropna()
                 .groupby("k")["v"].agg(lambda v: v.value_counts().index[0]))
        out = keys.map(canon)
        return out.where(y.notna()), (
            f"Target had {n_raw} spellings for {len(uniq_keys)} classes; merged spelling variants.")
    return y, ""
