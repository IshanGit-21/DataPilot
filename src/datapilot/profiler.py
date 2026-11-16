"""
Stage 1 — Dataset Profiler.

Inspects a raw dataframe and produces a machine-readable DatasetProfile.
This module never mutates the input dataframe (raw data stays immutable).

Covers every element of the Stage 1 spec: missing values, duplicates,
outliers, distributions, and relationships (correlations) — not just the
first two.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .cleaning import looks_numeric_text, n_unique_after_cleaning, parse_numeric_series


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_unique: int
    unique_ratio: float
    n_missing: int
    missing_ratio: float
    is_constant: bool
    is_likely_identifier: bool
    is_high_cardinality: bool
    is_datetime: bool = False
    is_boolean: bool = False
    is_numeric: bool = False
    is_free_text: bool = False
    n_unique_clean: int | None = None   # distinct values after merging spelling variants (strings only)
    has_spelling_variants: bool = False
    is_numeric_text: bool = False
    skew: float | None = None          # numeric columns only
    n_outliers_iqr: int | None = None  # numeric columns only
    outlier_ratio: float | None = None


@dataclass
class CorrelationPair:
    col_a: str
    col_b: str
    correlation: float


@dataclass
class DatasetProfile:
    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    target_column: str | None
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    target_distribution: dict | None = None
    top_correlations: list[CorrelationPair] = field(default_factory=list)
    target_correlations: list[CorrelationPair] = field(default_factory=list)


class DatasetProfiler:
    """Produces a DatasetProfile from a raw dataframe. Read-only."""

    IDENTIFIER_UNIQUE_RATIO_THRESHOLD = 0.95
    HIGH_CARDINALITY_THRESHOLD = 50
    DATETIME_PARSE_SUCCESS_THRESHOLD = 0.9  # fraction of non-null values that must parse
    CORRELATION_THRESHOLD = 0.7
    TOP_CORRELATIONS_LIMIT = 10
    ID_NAME_PATTERN = re.compile(r"(^|_)(id|uuid|guid|index|idx)($|_)", re.IGNORECASE)
    FREE_TEXT_MIN_AVG_WORDS = 5.0  # "This is a disaster tweet" (15 avg words) vs "New York" (2 avg words)

    def profile(self, df: pd.DataFrame, target: str | None = None) -> DatasetProfile:
        n_rows = len(df)
        columns: dict[str, ColumnProfile] = {}

        for col in df.columns:
            series = df[col]
            n_unique = series.nunique(dropna=True)
            unique_ratio = n_unique / n_rows if n_rows else 0.0
            n_missing = int(series.isna().sum())
            is_bool = pd.api.types.is_bool_dtype(series)
            is_numeric = pd.api.types.is_numeric_dtype(series) and not is_bool
            is_num_text = (not is_numeric and not is_bool and col != target
                           and looks_numeric_text(series))
            if is_num_text:
                series = parse_numeric_series(series)
                is_numeric = True
                n_unique = series.nunique(dropna=True)
                unique_ratio = n_unique / n_rows if n_rows else 0.0
            is_dt = self._looks_like_datetime(series)
            is_text = (not is_dt) and self._looks_like_free_text(series)
            is_str = (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and not is_num_text
            n_clean = n_unique
            if is_str and not is_dt and not is_text and 1 < n_unique <= 5000:
                n_clean = n_unique_after_cleaning(series)

            skew = None
            n_outliers = None
            outlier_ratio = None
            if is_numeric:
                skew, n_outliers, outlier_ratio = self._numeric_diagnostics(series)

            columns[col] = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                n_unique=n_unique,
                unique_ratio=unique_ratio,
                n_missing=n_missing,
                missing_ratio=n_missing / n_rows if n_rows else 0.0,
                is_constant=n_unique <= 1,
                is_likely_identifier=(
                    not is_text and self._looks_like_identifier(
                        col, series, unique_ratio, is_numeric, is_dt
                    )
                ),
                is_high_cardinality=(
                    (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series))
                    and n_clean > self.HIGH_CARDINALITY_THRESHOLD and not is_dt and not is_text
                ),
                n_unique_clean=n_clean if is_str else None,
                has_spelling_variants=bool(is_str and n_clean < n_unique),
                is_numeric_text=is_num_text,
                is_datetime=is_dt,
                is_boolean=is_bool,
                is_numeric=is_numeric,
                is_free_text=is_text,
                skew=skew,
                n_outliers_iqr=n_outliers,
                outlier_ratio=outlier_ratio,
            )

        target_distribution = None
        if target is not None and target in df.columns:
            if pd.api.types.is_numeric_dtype(df[target]):
                target_distribution = df[target].describe().to_dict()
            else:
                target_distribution = df[target].value_counts(normalize=True).to_dict()

        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
        ]
        # Feature<->feature correlations must exclude the target — otherwise
        # target<->feature pairs leak into this panel too, duplicating (and
        # mislabeling) what target_correlations already reports separately.
        non_target_numeric_cols = [c for c in numeric_cols if c != target]
        top_correlations = self._top_correlations(df, non_target_numeric_cols)
        target_correlations = self._target_correlations(df, numeric_cols, target)

        return DatasetProfile(
            n_rows=n_rows,
            n_cols=df.shape[1],
            n_duplicate_rows=int(df.duplicated().sum()),
            target_column=target,
            columns=columns,
            target_distribution=target_distribution,
            top_correlations=top_correlations,
            target_correlations=target_correlations,
        )

    def _looks_like_identifier(
        self, name: str, series: pd.Series, unique_ratio: float, is_numeric: bool, is_dt: bool
    ) -> bool:
        """A high unique-ratio alone is NOT enough: a continuous numeric
        measurement (price, a precise sensor reading) is also ~100% unique
        but is a real feature, not an identifier. Silently dropping it
        (the old behavior) starves every downstream model of its best
        signal — this is exactly the kind of bug that only shows up on
        real, non-toy data. An identifier is only flagged when the values
        look like a generated key: a sequential auto-increment integer, or
        the column name itself says so (id/uuid/guid/index).
        """
        if is_dt or unique_ratio < self.IDENTIFIER_UNIQUE_RATIO_THRESHOLD:
            return False
        if not is_numeric:
            return True  # near-unique strings (codes, UUIDs) are almost always identifiers
        if self._is_sequential_index(series):
            return True
        return bool(self.ID_NAME_PATTERN.search(self._normalise_name(name)))

    @staticmethod
    def _normalise_name(name: str) -> str:
        """'User ID', 'UserID', 'CustomerId', 'customer-id' -> 'user_id' style so
        one pattern catches every common spelling of an id column."""
        spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
        return re.sub(r"[^A-Za-z0-9]+", "_", spaced).strip("_").lower()

    @staticmethod
    def _is_sequential_index(series: pd.Series) -> bool:
        if not pd.api.types.is_integer_dtype(series):
            return False
        vals = series.dropna()
        if len(vals) == 0:
            return False
        uniq = np.sort(vals.unique())
        if len(uniq) != len(vals):
            return False  # had duplicates — not a pure index
        expected = np.arange(uniq[0], uniq[0] + len(uniq))
        return np.array_equal(uniq, expected)

    def _looks_like_free_text(self, series: pd.Series) -> bool:
        """Distinguishes genuine free text (tweets, reviews, descriptions)
        from short identifier-like or categorical strings. This matters
        because a free-text column is very often ~100% unique purely from
        natural-language variety — the exact same false-positive-identifier
        failure mode as a continuous numeric column, just on the string
        side. Average word count, not just uniqueness, is what actually
        distinguishes "the tweet text" from "a UUID" or "a city name."
        """
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return False
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        sample = non_null.sample(min(200, len(non_null)), random_state=0).astype(str)
        avg_words = sample.str.split().str.len().mean()
        return bool(avg_words >= self.FREE_TEXT_MIN_AVG_WORDS)

    def _looks_like_datetime(self, series: pd.Series) -> bool:
        """Detects real datetime dtype AND date-like strings read as
        string/object columns (the common case for CSV uploads). Uses
        pandas' type-checking API rather than raw dtype-string comparison,
        since pandas 3.x defaults string columns to StringDtype (not
        "object") — a literal `dtype == "object"` check silently misses
        every string column on this pandas version.
        """
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return False
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            return False
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        sample = non_null.sample(min(200, len(non_null)), random_state=0)
        # numeric-looking strings ("1", "2", "3"...) are not dates — reject
        # them early so ID-like integer columns aren't misdetected.
        if sample.astype(str).str.fullmatch(r"-?\d+(\.\d+)?").mean() > 0.5:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() >= self.DATETIME_PARSE_SUCCESS_THRESHOLD

    def _numeric_diagnostics(self, series: pd.Series) -> tuple[float | None, int | None, float | None]:
        clean = series.dropna()
        if len(clean) < 3:
            return None, None, None

        skew = float(scipy_stats.skew(clean)) if clean.nunique() > 1 else 0.0

        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            n_outliers = 0
        else:
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = int(((clean < lower) | (clean > upper)).sum())
        outlier_ratio = n_outliers / len(clean) if len(clean) else 0.0
        return skew, n_outliers, outlier_ratio

    def _top_correlations(self, df: pd.DataFrame, numeric_cols: list[str]) -> list[CorrelationPair]:
        if len(numeric_cols) < 2:
            return []
        corr = df[numeric_cols].corr(numeric_only=True)
        pairs = []
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                val = corr.loc[a, b]
                if pd.notna(val) and abs(val) >= self.CORRELATION_THRESHOLD:
                    pairs.append(CorrelationPair(a, b, float(val)))
        pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
        return pairs[: self.TOP_CORRELATIONS_LIMIT]

    def _target_correlations(
        self, df: pd.DataFrame, numeric_cols: list[str], target: str | None
    ) -> list[CorrelationPair]:
        if target is None or target not in df.columns:
            return []
        if not pd.api.types.is_numeric_dtype(df[target]):
            return []  # classification targets: left for Stage 4 (ANOVA F-test), not here
        feature_cols = [c for c in numeric_cols if c != target]
        if not feature_cols:
            return []
        # One vectorized correlation matrix instead of a separate .corr()
        # call per column — same result, no per-column pandas overhead.
        corr_with_target = df[feature_cols + [target]].corr(numeric_only=True)[target]
        pairs = [
            CorrelationPair(col, target, float(corr_with_target[col]))
            for col in feature_cols
            if pd.notna(corr_with_target[col])
        ]
        pairs.sort(key=lambda p: abs(p.correlation), reverse=True)
        return pairs[: self.TOP_CORRELATIONS_LIMIT]
