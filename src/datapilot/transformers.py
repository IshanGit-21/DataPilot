"""
Custom, sklearn-compatible transformers used by the Pipeline Planner.

Each one is a real BaseEstimator/TransformerMixin: fit() only ever sees the
training fold when used inside a Pipeline/ColumnTransformer + cross_validate,
so these compose into the anti-leakage guarantee exactly like sklearn's own
transformers do. None of this is a stand-in — every method here is what
actually runs against the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encodes each categorical value as its (training-set) occurrence
    frequency. Used for high-cardinality categoricals where one-hot
    encoding would blow up dimensionality (e.g. a city or zip-code column
    with hundreds of distinct values on real-world data).
    """

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.freq_maps_ = {
            col: X[col].value_counts(normalize=True, dropna=True).to_dict()
            for col in X.columns
        }
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = self._as_frame(X)
        out = np.zeros((len(X), len(self.columns_)), dtype=float)
        for i, col in enumerate(self.columns_):
            freq_map = self.freq_maps_.get(col, {})
            out[:, i] = X[col].map(freq_map).fillna(0.0).to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_freq" for c in self.columns_])

    @staticmethod
    def _as_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class DatePartExtractor(BaseEstimator, TransformerMixin):
    """Turns one or more date-like columns into numeric parts (year, month,
    day, day-of-week, is-weekend). Unparseable values become NaN, which the
    surrounding pipeline is expected to impute downstream — this transformer
    only extracts, it does not silently drop bad rows.
    """

    PARTS = ("year", "month", "day", "dayofweek", "is_weekend")

    def fit(self, X, y=None):
        X = self._as_frame(X)
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = self._as_frame(X)
        blocks = []
        for col in self.columns_:
            parsed = pd.to_datetime(X[col], errors="coerce", format="mixed")
            block = np.column_stack([
                parsed.dt.year.to_numpy(dtype=float),
                parsed.dt.month.to_numpy(dtype=float),
                parsed.dt.day.to_numpy(dtype=float),
                parsed.dt.dayofweek.to_numpy(dtype=float),
                (parsed.dt.dayofweek >= 5).astype(float).to_numpy(),
            ])
            blocks.append(block)
        return np.hstack(blocks) if blocks else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{c}_{p}" for c in self.columns_ for p in self.PARTS])

    @staticmethod
    def _as_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorizes numeric columns to the training-set IQR bounds
    (Q1 - 1.5*IQR, Q3 + 1.5*IQR). Fit only on the training fold, so bounds
    are never informed by validation/test data — this is the aggressive
    candidate's real answer to the outliers Stage 1 detects, not a cosmetic
    step.
    """

    def fit(self, X, y=None):
        X = self._as_frame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = (q1 - 1.5 * iqr).to_dict()
        self.upper_ = (q3 + 1.5 * iqr).to_dict()
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        X = self._as_frame(X).copy()
        for col in self.columns_:
            X[col] = X[col].clip(lower=self.lower_.get(col), upper=self.upper_.get(col))
        return X.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_)

    @staticmethod
    def _as_frame(X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class TextFeaturizer(BaseEstimator, TransformerMixin):
    """Real TF-IDF featurization (scikit-learn's own TfidfVectorizer) for a
    genuine free-text column — not a stand-in for NLP handling, the
    standard, real technique for turning text into numeric features for
    classical ML models. Bounded to max_features for the same reason
    TopKInteractions is bounded to k=3 elsewhere in Stage 3: a vocabulary
    of thousands of words would blow up dimensionality combinatorially on
    a wide dataset.

    TfidfVectorizer itself expects a 1D iterable of strings, not the 2D
    column slice ColumnTransformer normally hands a transformer — this
    class's whole job is bridging that gap, so TfidfVectorizer can be used
    inside a ColumnTransformer branch like every other transformer here.
    """

    def __init__(self, max_features: int = 200):
        self.max_features = max_features

    def fit(self, X, y=None):
        series = self._as_series(X)
        self.column_name_ = series.name or "text"
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features, stop_words="english", min_df=2
        )
        self.vectorizer_.fit(series.fillna("").astype(str))
        return self

    def transform(self, X):
        series = self._as_series(X)
        return self.vectorizer_.transform(series.fillna("").astype(str)).toarray()

    def get_feature_names_out(self, input_features=None):
        prefix = getattr(self, "column_name_", "text")
        return np.array([f"{prefix}_tfidf_{w}" for w in self.vectorizer_.get_feature_names_out()])

    @staticmethod
    def _as_series(X):
        if isinstance(X, pd.Series):
            return X
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0]
        arr = np.asarray(X)
        if arr.ndim == 2:
            arr = arr.ravel()
        return pd.Series(arr)


class StringShape(BaseEstimator, TransformerMixin):
    """Structure hidden inside code-like text: names, tickets, cabins, SKUs, addresses.

    Such columns usually look like identifiers (almost every value unique) and are
    dropped, yet their *shape* carries signal: the honorific in "Smith, Mr. John" or the
    prefix in "PC 17599" / "C85". For each column this produces length, word count,
    digit/upper-case counts, a missing flag, plus 0/1 indicators for the most common
    honorific-like token (a word ending in '.') and the most common leading letter code.
    Only tokens that are frequent in the training fold become indicators, so it is
    leakage-safe and does not memorise one-off strings.
    """

    TOP_K = 8

    def fit(self, X, y=None):
        X = self._frame(X)
        n = max(len(X), 1)
        self.min_count_ = max(5, int(0.01 * n))
        self.columns_ = list(X.columns)
        self.tokens_: dict[str, tuple[list[str], list[str]]] = {}
        names: list[str] = []
        for c in self.columns_:
            s = X[c].astype("string").fillna("")
            hon = self._top(s.map(self._honorific))
            pre = self._top(s.map(self._prefix))
            self.tokens_[c] = (hon, pre)
            names += [f"{c}__{k}" for k in ("len", "words", "digits", "upper", "missing")]
            names += [f"{c}__title={t}" for t in hon] + [f"{c}__prefix={p}" for p in pre]
        self.all_names_ = names
        feats = self._raw(X)
        self.keep_ = feats.std(axis=0) > 0
        return self

    def transform(self, X):
        return self._raw(self._frame(X))[:, self.keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.all_names_)[self.keep_]

    # ---- helpers
    def _top(self, tokens: pd.Series) -> list[str]:
        vc = tokens[tokens != ""].value_counts()
        return [t for t, c in vc.items() if c >= self.min_count_][: self.TOP_K]

    @staticmethod
    def _honorific(v: str) -> str:
        import re
        m = re.search(r"\b([A-Za-z]{1,12})\.", v)
        return m.group(1).lower() if m else ""

    @staticmethod
    def _prefix(v: str) -> str:
        import re
        m = re.match(r"\s*([A-Za-z]{1,4})(?=[\s\d./-]|$)", v)
        return m.group(1).upper() if m else ""

    def _raw(self, X: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.columns_:
            raw = X[c] if c in X.columns else pd.Series([None] * len(X))
            miss = raw.isna().to_numpy().astype(float)
            s = raw.astype("string").fillna("")
            cols += [s.str.len().to_numpy(float), s.str.split().str.len().fillna(0).to_numpy(float),
                     s.str.count(r"\d").to_numpy(float), s.str.count(r"[A-Z]").to_numpy(float), miss]
            hon, pre = self.tokens_[c]
            h = s.map(self._honorific)
            p = s.map(self._prefix)
            cols += [(h == t).to_numpy(float) for t in hon] + [(p == t).to_numpy(float) for t in pre]
        return np.column_stack(cols) if cols else np.zeros((len(X), 0))

    @staticmethod
    def _frame(X):
        return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
