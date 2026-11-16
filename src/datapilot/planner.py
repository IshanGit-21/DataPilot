"""
Stage 2 — Preprocessing.

Covers the full Stage 2 spec: train_test_split -> imputation -> encoding ->
scaling -> ColumnTransformer -> Pipeline. Generates 2+ materially different
candidate pipelines (conservative/balanced/aggressive) rather than one
"universal best" transformation.

Real-world handling, not just the toy-CSV case:
  - low-cardinality categoricals -> one-hot encoded
  - HIGH-cardinality categoricals (zip codes, city names, ...) -> frequency
    encoded instead, so a 500-category column doesn't explode into 500
    one-hot columns
  - datetime-like columns (real dtype or date strings) -> decomposed into
    year/month/day/day-of-week/is-weekend, not dropped or garbage-encoded
  - Stage 3 (feature_engineering.py) lives inside the balanced/aggressive
    candidates: Yeo-Johnson skew correction + bounded, target-relevant
    interaction features, both decided fresh per training fold
  - the aggressive candidate clips outliers (IQR-based, fit on train only)
    before scaling, directly acting on what Stage 1 detects instead of
    just reporting it
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    TargetEncoder,
)

from .cleaning import RawCleaner, clean_class_target
from .decision import DecisionRecord
from .feature_engineering import SkewCorrector, TopKInteractions
from .profiler import DatasetProfile
from .reasoner import TaskType, UserPreferences
from .transformers import (
    DatePartExtractor,
    FrequencyEncoder,
    IQRClipper,
    StringShape,
    TextFeaturizer,
)


@dataclass
class PipelineCandidate:
    name: str  # "conservative" | "balanced" | "aggressive"
    pipeline: Pipeline
    decisions: list[DecisionRecord] = field(default_factory=list)
    description: str = ""


@dataclass
class ColumnGroups:
    numeric: list[str]
    categorical_low: list[str]
    categorical_high: list[str]
    datetime: list[str]
    free_text: list[str] = field(default_factory=list)
    numeric_text: list[str] = field(default_factory=list)  # numbers stored as text ('$12.50')
    spelling_variants: list[str] = field(default_factory=list)  # categorical columns needing spelling normalisation
    shape: list[str] = field(default_factory=list)  # code-like text columns (names, tickets, cabins) mined for structure


@dataclass
class SplitResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    n_dropped_missing_target: int
    label_encoder: object | None = None  # classification only — see train_test_split_data
    stratified: bool = False
    n_dropped_duplicates: int = 0
    target_note: str = ""
    stratify_note: str = ""
    score_mode: str = "holdout"   # "holdout" (20% set aside), "external" (your own test file), "cv" (no hidden rows)
    test_note: str = ""

    def __iter__(self):
        # lets callers keep using `X_train, X_test, y_train, y_test = split(...)`
        return iter((self.X_train, self.X_test, self.y_train, self.y_test))


def train_test_split_data(
    df: pd.DataFrame,
    target: str,
    task_type: TaskType,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_duplicates: bool = True,
    test_df: pd.DataFrame | None = None,
) -> SplitResult:
    """The real first step of Stage 2. Splits once, before any fitting, so
    the held-out test set is never touched by imputation/encoding/scaling
    fit on the training fold. Stratifies on the target for classification
    tasks so class balance is preserved in both splits — except when a
    class has fewer than 2 rows, which makes stratification mathematically
    impossible (each split needs at least 1 row per class); that case falls
    back to a plain split rather than crashing, and the fallback is recorded
    in `stratify_note` rather than happening silently.

    Rows with a missing target are dropped first — every model in Stage 4
    requires a known label to train against, so keeping them would crash
    every single model at fit time (not a hypothetical: this is exactly
    what happens on real datasets with incomplete labels). The drop count
    is returned, not hidden, so the UI can say so.

    For classification, the target is label-encoded to contiguous 0..n-1
    integers. This isn't cosmetic: XGBoost's sklearn API raises a hard
    error on any other label form (string labels like "yes"/"no", or
    integers that don't start at 0) — real, common label formats that every
    other model in the zoo handles natively. Encoding once here,
    consistently across train and test, fixes it for every model instead of
    special-casing XGBoost. The fitted encoder is returned so callers can
    map predictions back to the original class names for display.
    """
    n_missing_target = int(df[target].isna().sum())
    clean = df.dropna(subset=[target]) if n_missing_target else df

    # Exact duplicate rows would land in both train and test and inflate the
    # held-out score (the model "predicts" rows it has memorised).
    n_dupes = 0
    if drop_duplicates:
        n_dupes = int(clean.duplicated().sum())
        if n_dupes:
            clean = clean.drop_duplicates()

    target_note = ""
    if task_type == TaskType.CLASSIFICATION:
        cleaned_y, target_note = clean_class_target(clean[target])
        if target_note:
            clean = clean.assign(**{target: cleaned_y})

    X = clean.drop(columns=[target])
    y = clean[target]

    label_encoder = None
    if task_type == TaskType.CLASSIFICATION:
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)

    stratified = False
    stratify_note = ""
    stratify_arg = None
    if task_type == TaskType.CLASSIFICATION and y.nunique() > 1:
        min_class_count = int(y.value_counts().min())
        if min_class_count >= 2:
            stratify_arg = y
            stratified = True
        else:
            stratify_note = (
                f"Not stratified — at least one class has only {min_class_count} row, "
                f"fewer than the 2 required to appear in both the train and test split."
            )

    if test_df is not None:
        return _split_with_test_file(X, y, test_df, target, task_type, label_encoder, n_missing_target,
                                     stratified, stratify_note, n_dupes, target_note)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
    )
    return SplitResult(
        X_train, X_test, y_train, y_test, n_missing_target,
        label_encoder=label_encoder, stratified=stratified, stratify_note=stratify_note,
        n_dropped_duplicates=n_dupes, target_note=target_note,
    )


def file_has_answers(test_df: pd.DataFrame, target: str) -> bool:
    """A test file can be scored only if it carries the answers (most Kaggle test files do not)."""
    return target in test_df.columns and bool(test_df[target].notna().mean() > 0.5)


def _split_with_test_file(X, y, test_df, target, task_type, label_encoder, n_missing_target,
                          stratified, stratify_note, n_dupes, target_note) -> SplitResult:
    """Use every training row for learning and the user's own test file for the final check.

    * test file WITH the answers: it is the held-out test; nothing is cut from the training rows.
    * test file WITHOUT answers (Kaggle style): there is nothing to score against, so no rows are
      hidden either; the final score is worked out later by cross-checking on the training rows."""
    missing = [c for c in X.columns if c not in test_df.columns]
    if missing:
        raise ValueError(f"The test file is missing columns the training file has: {missing[:8]}")
    if file_has_answers(test_df, target):
        t = test_df.dropna(subset=[target])
        yt = t[target]
        note = f"Your test file ({len(t):,} rows) was used as the final test; every row of the training file was used to learn."
        if task_type == TaskType.CLASSIFICATION:
            cleaned, _ = clean_class_target(yt)
            known = set(label_encoder.classes_)
            keep = cleaned.isin(known)
            if int((~keep).sum()):
                note += f" {int((~keep).sum()):,} test rows with an answer never seen in training were left out."
            t, cleaned = t[keep.to_numpy()], cleaned[keep.to_numpy()]
            yt = pd.Series(label_encoder.transform(cleaned), index=t.index, name=target)
        if len(t) == 0:
            raise ValueError("None of the test file's answers match the training file's answers.")
        return SplitResult(X, t[list(X.columns)], y, yt, n_missing_target, label_encoder=label_encoder,
                           stratified=stratified, stratify_note=stratify_note, n_dropped_duplicates=n_dupes,
                           target_note=target_note, score_mode="external", test_note=note)
    note = (f"Your test file ({len(test_df):,} rows) has no '{target}' column, so it cannot be scored. "
            "Every training row is used to learn, the score comes from cross-checking on the training rows, "
            "and the test file gets its predictions.")
    from .scale import CHECK_ROWS, subsample_for_search
    Xc, yc, _ = subsample_for_search(X, y, task_type, max_rows=CHECK_ROWS)   # the rows that get cross-checked
    return SplitResult(X, Xc, y, yc, n_missing_target, label_encoder=label_encoder,
                       stratified=stratified, stratify_note=stratify_note, n_dropped_duplicates=n_dupes,
                       target_note=target_note, score_mode="cv", test_note=note)


def _make_target_encoder():
    """Newer scikit-learn (1.9) wants a shuffled splitter as `cv` and deprecates
    `random_state`; older releases only accept an int `cv`. Try the new form, check that this
    installation accepts it, and fall back to the old one."""
    from sklearn.model_selection import KFold
    try:
        enc = TargetEncoder(cv=KFold(5, shuffle=True, random_state=42))
        enc._validate_params()
        return enc
    except Exception:  # noqa: BLE001 — older scikit-learn: cv must be an int
        return TargetEncoder(cv=5, random_state=42)


@dataclass
class PipelineOptions:
    """User overrides for the planner. Every field defaults to "what the
    planner would do anyway", so PipelineOptions() changes nothing."""
    excluded_columns: list[str] = field(default_factory=list)  # inputs the user removed
    pipelines: list[str] | None = None        # subset of conservative/balanced/aggressive; None = all
    numeric_impute: str = "median"            # median | mean | most_frequent
    skew_correction: bool = True              # balanced + aggressive
    interactions: bool = True                 # balanced + aggressive
    clip_outliers: str = "auto"               # auto | yes | no  (aggressive)
    pca_variance: float = 0.95                # aggressive: explained variance kept
    tfidf_max_features: int | None = None     # None = planner default per pipeline
    missing_indicators: bool | None = None    # None = follow UserPreferences
    string_shape: bool = True                 # balanced + aggressive: mine structure (titles, prefixes, lengths) from code-like text columns
    target_encoding: bool = True              # balanced + aggressive: add cross-fitted target encoding for high-cardinality categoricals

    def __post_init__(self):
        if self.numeric_impute not in ("median", "mean", "most_frequent"):
            raise ValueError(f"numeric_impute must be median/mean/most_frequent, got {self.numeric_impute!r}")
        if self.clip_outliers not in ("auto", "yes", "no"):
            raise ValueError(f"clip_outliers must be auto/yes/no, got {self.clip_outliers!r}")
        if not 0.5 <= self.pca_variance < 1.0:
            raise ValueError("pca_variance must be in [0.5, 1.0)")
        if self.tfidf_max_features is not None and self.tfidf_max_features < 1:
            raise ValueError("tfidf_max_features must be >= 1")
        if self.pipelines is not None:
            bad = set(self.pipelines) - {"conservative", "balanced", "aggressive"}
            if bad or not self.pipelines:
                raise ValueError(f"pipelines must be a non-empty subset of conservative/balanced/aggressive, got {self.pipelines!r}")

    def overrides_summary(self) -> str:
        d = PipelineOptions()
        parts = []
        for f in ("numeric_impute", "skew_correction", "interactions", "clip_outliers",
                  "pca_variance", "tfidf_max_features", "missing_indicators", "target_encoding", "string_shape"):
            if getattr(self, f) != getattr(d, f):
                parts.append(f"{f}={getattr(self, f)}")
        if self.excluded_columns:
            parts.append(f"excluded_columns={self.excluded_columns}")
        return ", ".join(parts)


class PipelinePlanner:
    """
    Builds candidate pipelines from a DatasetProfile + UserPreferences.
    Deterministic rules; optional PipelineOptions let the user override them.
    """

    def plan(
        self, df_columns: list[str], profile: DatasetProfile, prefs: UserPreferences,
        options: PipelineOptions | None = None,
    ) -> list[PipelineCandidate]:
        opts = options or PipelineOptions()
        cols = [c for c in df_columns if c not in set(opts.excluded_columns)]
        groups = self.split_columns(cols, profile)

        candidates = [
            self._conservative(groups, profile, opts),
            self._balanced(groups, profile, prefs, opts),
            self._aggressive(groups, profile, prefs, opts),
        ]
        if groups.spelling_variants or groups.numeric_text:
            for c in candidates:
                c.pipeline.steps.insert(0, ("clean", RawCleaner(
                    columns=list(groups.spelling_variants), numeric_columns=list(groups.numeric_text))))
            if groups.numeric_text:
                for c in candidates:
                    c.decisions.insert(0, DecisionRecord(
                        what=f"Read {len(groups.numeric_text)} text column(s) as numbers: "
                             f"{', '.join(groups.numeric_text[:6])}",
                        why="Values such as '$114.17' or '1,200' are numbers with formatting, not categories",
                        evidence="Over 90% of non-missing values parse as numbers once currency symbols and separators are removed",
                        alternatives="Treat as categorical text",
                        risk="Entries that do not parse become missing and are imputed",
                        effect="Column joins the numeric branch (imputation, scaling, skew handling)",
                        reversible=True,
                    ))
        if groups.spelling_variants:
            raw = sum(profile.columns[c].n_unique for c in groups.spelling_variants)
            clean = sum(profile.columns[c].n_unique_clean for c in groups.spelling_variants)
            for c in candidates:
                c.decisions.insert(0, DecisionRecord(
                    what=f"Normalise spelling variants in {len(groups.spelling_variants)} categorical column(s): "
                         f"{', '.join(groups.spelling_variants[:6])}",
                    why="Case, punctuation and '&'/'and' differences split one category into several, "
                        "which weakens every encoder and model",
                    evidence=f"{raw} distinct raw values collapse to {clean} after normalisation",
                    alternatives="Treat each spelling as its own category",
                    risk="Aliases that are not spelling variants (e.g. 'DE' vs 'Germany') are not merged",
                    effect=f"{raw - clean} fewer categories before encoding",
                    reversible=True,
                ))
        if groups.categorical_high and opts.target_encoding:
            for c in candidates:
                if c.name == "conservative":
                    continue
                c.decisions.append(DecisionRecord(
                    what=f"Add cross-fitted target encoding for {len(groups.categorical_high)} "
                         f"high-cardinality categorical column(s), alongside frequency encoding",
                    why="Frequency encoding says how common a category is; target encoding says how "
                        "it relates to the outcome, which is often the stronger signal",
                    evidence="sklearn TargetEncoder fits with internal cross-fitting, so training rows never "
                             "see their own target value",
                    alternatives="Frequency encoding only (conservative)",
                    risk="Can overfit rare categories; smoothing in TargetEncoder limits this",
                    effect="One extra encoded column per high-cardinality categorical",
                    reversible=True,
                ))
        if opts.pipelines is not None:
            candidates = [c for c in candidates if c.name in opts.pipelines]
        note = opts.overrides_summary()
        if note:
            for c in candidates:
                c.decisions.append(DecisionRecord(
                    what=f"User override applied: {note}",
                    why="Chosen manually in the pipeline editor",
                    evidence="User setting, not data-driven",
                    alternatives="Planner defaults",
                    risk="Overrides bypass the planner's data-driven defaults",
                    effect="See settings listed",
                    reversible=True,
                ))
        return candidates

    def plan_rescue(self, df_columns, profile: DatasetProfile, options: PipelineOptions | None = None
                    ) -> PipelineCandidate:
        """A deliberately plain second approach, used only when the main pipelines crash for
        some model: median-impute numbers, ordinal-code every categorical (unknowns -> -1),
        decompose dates, drop free text. Fewer moving parts, so fewer ways to fail."""
        opts = options or PipelineOptions()
        cols = [c for c in df_columns if c not in set(opts.excluded_columns)]
        g = self.split_columns(cols, profile)
        cats = g.categorical_low + g.categorical_high
        transformers = [("num", SimpleImputer(strategy="median"), g.numeric)]
        if cats:
            transformers.append(("cat", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                       encoded_missing_value=-1)),
            ]), cats))
        transformers += self._datetime_transformer(g, scale=False)
        steps = []
        if g.spelling_variants or g.numeric_text:
            steps.append(("clean", RawCleaner(columns=list(g.spelling_variants),
                                              numeric_columns=list(g.numeric_text))))
        steps.append(("preprocess", ColumnTransformer(transformers, remainder="drop")))
        decision = DecisionRecord(
            what="Rescue pipeline: median-impute numbers, ordinal-code categoricals, decompose dates, drop free text",
            why="Used only for models that failed on the main pipelines, so every model still gets a result",
            evidence="Failure of at least one model x pipeline combination during cross-validation",
            alternatives="Leave the failed combinations out (the previous behaviour)",
            risk="Ordinal codes impose an arbitrary order on categories; tree models cope, linear models do not",
            effect="Fewer features and fewer transformation steps than any main pipeline",
            reversible=True,
        )
        return PipelineCandidate("rescue", Pipeline(steps), [decision],
                                 "Plain fallback used when the main pipelines fail for a model")

    def split_columns(self, columns, profile: DatasetProfile) -> ColumnGroups:
        numeric, cat_low, cat_high, datetime_cols, free_text, shape = [], [], [], [], [], []
        for name in columns:
            if name == profile.target_column:
                continue
            col = profile.columns.get(name)
            if col is None or col.is_constant:
                continue
            if col.is_likely_identifier:
                # dropped as an id, except text whose *shape* carries signal (names, tickets)
                if not (col.is_numeric or col.is_datetime or col.is_boolean) and (col.n_unique_clean is not None):
                    shape.append(name)
                continue  # dropped — reflected in the conservative DecisionRecord
            if col.is_free_text:
                free_text.append(name)
            elif col.is_datetime:
                datetime_cols.append(name)
            elif col.is_numeric or col.is_boolean:
                numeric.append(name)
            elif col.is_high_cardinality:
                cat_high.append(name)
                shape.append(name)
            else:
                cat_low.append(name)
        variants = [c for c in cat_low + cat_high if profile.columns[c].has_spelling_variants]
        num_text = [c for c in numeric if profile.columns[c].is_numeric_text]
        return ColumnGroups(numeric, cat_low, cat_high, datetime_cols, free_text, num_text, variants, shape)

    # ---------------------------------------------------------- helpers ---
    def _categorical_transformers(self, groups: ColumnGroups, target_encode: bool = False) -> list[tuple]:
        parts = []
        if groups.categorical_low:
            parts.append((
                "cat_low",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    # sparse_output=False: OneHotEncoder defaults to a sparse
                    # matrix, which ColumnTransformer then propagates to the
                    # WHOLE combined output once categorical columns make up
                    # enough of it (sparse_threshold=0.3 by default) — even
                    # though the numeric branch (StandardScaler, PCA, etc.)
                    # produced dense output on its own. On a real,
                    # heavily-categorical dataset (Ames Housing: 81 columns,
                    # many categoricals) this silently turned the aggressive
                    # candidate's PCA step into operating on a sparse matrix,
                    # which has different, stricter solver constraints and
                    # crashed outright. Every other custom transformer in
                    # this codebase already assumes dense input/output —
                    # forcing dense here keeps that consistent everywhere,
                    # not just papering over the one PCA symptom.
                    ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                groups.categorical_low,
            ))
        if groups.categorical_high:
            parts.append((
                "cat_high",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", FrequencyEncoder()),
                ]),
                groups.categorical_high,
            ))
            if target_encode:
                # sklearn's TargetEncoder cross-fits internally during fit_transform, so the
                # training rows never see their own target value (no leakage); transform()
                # on new data uses the full-train encoding.
                parts.append((
                    "cat_high_te",
                    Pipeline([
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", _make_target_encoder()),
                    ]),
                    groups.categorical_high,
                ))
        return parts

    def _shape_transformers(self, groups: ColumnGroups, opts: PipelineOptions, scale: bool = False) -> list[tuple]:
        if not (opts.string_shape and groups.shape):
            return []
        step = Pipeline([("shape", StringShape()), ("scale", StandardScaler())]) if scale else StringShape()
        return [("shape", step, groups.shape)]

    def _datetime_transformer(self, groups: ColumnGroups, scale: bool = False) -> list[tuple]:
        if not groups.datetime:
            return []
        steps = [
            ("extract", DatePartExtractor()),
            ("impute", SimpleImputer(strategy="median")),
        ]
        if scale:
            steps.append(("scale", StandardScaler()))
        return [("datetime", Pipeline(steps), groups.datetime)]

    def _text_transformers(self, groups: ColumnGroups, max_features: int = 150) -> list[tuple]:
        """One TF-IDF branch per free-text column — each needs its own
        vocabulary, so they can't share a single transformer the way
        numeric/categorical columns do. Dropping a free-text column
        entirely (the old behavior, via the identifier-mislabeling bug)
        meant the single most predictive column in a text-classification
        dataset never reached any model; every candidate gets this,
        including conservative — being "the minimal baseline" shouldn't
        mean discarding the main signal.
        """
        return [
            (f"text_{name}", TextFeaturizer(max_features=max_features), name)
            for name in groups.free_text
        ]

    # -------------------------------------------------------- candidates ---
    def _conservative(self, groups: ColumnGroups, profile, opts: PipelineOptions) -> PipelineCandidate:
        """Minimal, safe transformations — median/mode impute, one-hot /
        frequency encode, date decomposition. No selection, no scaling."""
        transformers = [
            ("num", SimpleImputer(strategy=opts.numeric_impute), groups.numeric),
            *self._categorical_transformers(groups),
            *self._datetime_transformer(groups, scale=False),
            *self._text_transformers(groups, max_features=opts.tfidf_max_features or 100),
        ]
        preprocessor = ColumnTransformer(transformers, remainder="drop")
        pipeline = Pipeline([("preprocess", preprocessor)])
        decisions = [
            DecisionRecord(
                what=(
                    f"Median-impute {len(groups.numeric)} numeric columns; one-hot encode "
                    f"{len(groups.categorical_low)} low-cardinality categoricals; frequency-encode "
                    f"{len(groups.categorical_high)} high-cardinality categoricals; decompose "
                    f"{len(groups.datetime)} datetime columns into year/month/day/day-of-week"
                ),
                why="Safest baseline with no assumptions about missingness meaning; avoids one-hot dimensionality blowup on high-cardinality columns",
                evidence="Standard practice; column groups taken directly from the Stage 1 profile (dtype + cardinality + datetime detection)",
                alternatives="Indicator-based imputation and feature selection (see balanced/aggressive)",
                risk="May lose information if missingness is meaningful; frequency encoding loses category identity",
                effect="No feature count change beyond one-hot expansion + 5 columns per datetime field",
                reversible=True,
            ),
        ]
        if groups.free_text:
            decisions.append(DecisionRecord(
                what=f"TF-IDF (up to 100 terms/column) on {len(groups.free_text)} free-text column(s): "
                     f"{', '.join(groups.free_text)}",
                why="A free-text column is routinely ~100% unique from natural language variety, not "
                    "identifier-ness — dropping it (the old failure mode) throws away the main signal "
                    "in any text-classification dataset; even the minimal baseline should use it",
                evidence="Stage 1 detected average word count >= 5 for this column (free text, not an identifier)",
                alternatives="Larger vocabulary (balanced/aggressive use up to 150 terms)",
                risk="TF-IDF ignores word order and context; a real NLP embedding would capture more",
                effect="Up to 100 new sparse term-frequency columns per free-text column",
                reversible=True,
            ))
        return PipelineCandidate("conservative", pipeline, decisions,
                                  "Minimal transformations, safest baseline")

    def _balanced(self, groups: ColumnGroups, profile, prefs, opts: PipelineOptions) -> PipelineCandidate:
        """Real Stage 3 lives here: skew-correct numeric columns (Yeo-Johnson,
        decided fresh per training fold) and add a small, bounded set of
        target-relevant interaction features — before selection picks the
        best of everything. Missingness indicators are a separate
        ColumnTransformer branch (sklearn's MissingIndicator) rather than
        SimpleImputer's own add_indicator, specifically so the engineered
        numeric branch keeps real, chainable feature names instead of a
        positional numpy array mixed with indicator columns.
        """
        add_indicator = (prefs.preserve_meaningful_missingness
                         if opts.missing_indicators is None else opts.missing_indicators)
        any_missing = any((profile.columns[c].n_missing or 0) > 0 for c in groups.numeric)

        bal_num_steps = [("impute", SimpleImputer(strategy=opts.numeric_impute).set_output(transform="pandas"))]
        if opts.skew_correction:
            bal_num_steps.append(("skew", SkewCorrector()))
        if opts.interactions:
            bal_num_steps.append(("interact", TopKInteractions(task_type=prefs.task_type)))
        transformers = [
            ("num", Pipeline(bal_num_steps), groups.numeric),
            *self._categorical_transformers(groups, target_encode=opts.target_encoding),
            *self._datetime_transformer(groups, scale=False),
            *self._text_transformers(groups, max_features=opts.tfidf_max_features or 150),
            *self._shape_transformers(groups, opts),
        ]
        if add_indicator and any_missing:
            # error_on_new=False: MissingIndicator's default (True) crashes
            # outright if a fold's transform-time data has a missing value
            # in a feature that happened to have none at fit-time — routine
            # sampling variance for a column with RARE missingness (e.g.
            # 0.04% missing), not a real anomaly worth crashing over. With
            # error_on_new=False it just reports False/0 for that feature
            # on rows where fit never saw missingness, which is the
            # correct, honest behavior: this candidate didn't learn that
            # feature could be missing, so it doesn't claim to flag it.
            transformers.append((
                "num_missing_flag",
                MissingIndicator(features="missing-only", error_on_new=False),
                groups.numeric,
            ))
        preprocessor = ColumnTransformer(transformers, remainder="drop")
        score_fn = f_classif if prefs.task_type == TaskType.CLASSIFICATION else f_regression
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("select", SelectKBest(score_func=score_fn, k="all")),  # k tuned later via CV
        ])
        decisions = [
            DecisionRecord(
                what="Impute + missingness indicators, same encoding scheme as conservative",
                why="Preserve structurally-meaningful missingness as requested",
                evidence=f"preserve_meaningful_missingness={add_indicator}",
                alternatives="No indicators (conservative) or PCA (aggressive)",
                risk="Indicator columns increase dimensionality slightly",
                effect="Feature count grows by number of numeric columns with meaningful gaps",
                reversible=True,
            ),
            DecisionRecord(
                what="Stage 3: Yeo-Johnson skew correction on numeric columns whose training-fold "
                     "skew exceeds 1.0, plus up to 3 pairwise interaction features from the numeric "
                     "columns most associated with the target",
                why="Acts on what Stage 1's skew/relationship detection actually found, instead of "
                    "only reporting it — both decisions are computed fresh per training fold, not "
                    "from a pre-split profile, so they stay leakage-safe",
                evidence="scipy skew computed on the training fold; f_regression/f_classif score "
                         "for interaction-candidate ranking, also on the training fold only",
                alternatives="No feature construction (conservative); PCA-compressed features (aggressive)",
                risk="Interaction features can overfit on small datasets; skew correction changes a "
                     "feature's native units, costing some interpretability",
                effect="Up to 3 new interaction columns; flagged numeric columns replaced with their "
                       "power-transformed version (same column count)",
                reversible=True,
            ),
            DecisionRecord(
                what="Univariate feature selection (SelectKBest)",
                why="Let cross-validation pick how many of the (original + engineered) features "
                    "actually help, rather than keeping all of them by default",
                evidence=f"score_func={'f_classif' if prefs.task_type == TaskType.CLASSIFICATION else 'f_regression'}",
                alternatives="Keep all features (conservative); PCA compression (aggressive)",
                risk="Univariate scoring ignores feature interactions when ranking",
                effect="k tuned later via CV; currently k='all' as the leaderboard baseline",
                reversible=True,
            ),
        ]
        if groups.free_text:
            decisions.append(DecisionRecord(
                what=f"TF-IDF (up to 150 terms/column) on {len(groups.free_text)} free-text column(s): "
                     f"{', '.join(groups.free_text)}",
                why="Larger vocabulary than conservative's 100 terms, still bounded to avoid dimensionality blowup",
                evidence="Stage 1 detected average word count >= 5 for this column (free text, not an identifier)",
                alternatives="Smaller vocabulary (conservative); same vocabulary size feeds into PCA (aggressive)",
                risk="TF-IDF ignores word order and context; a real NLP embedding would capture more",
                effect="Up to 150 new sparse term-frequency columns per free-text column, subject to selection",
                reversible=True,
            ))
        return PipelineCandidate("balanced", pipeline, decisions,
                                  "Preserves meaningful missingness, engineers and selects features")

    def _aggressive(self, groups: ColumnGroups, profile, prefs, opts: PipelineOptions) -> PipelineCandidate:
        """Skew-corrects and adds interaction features (same Stage 3 as
        balanced), THEN clips outliers, scales, and reduces dimensionality
        via PCA — traded against interpretability."""
        has_outlier_signal = any(
            (profile.columns[c].outlier_ratio or 0) > 0 for c in groups.numeric
        )
        if opts.clip_outliers != "auto":
            has_outlier_signal = opts.clip_outliers == "yes"
        numeric_steps = [("impute", SimpleImputer(strategy=opts.numeric_impute).set_output(transform="pandas"))]
        if opts.skew_correction:
            numeric_steps.append(("skew", SkewCorrector()))
        if opts.interactions:
            numeric_steps.append(("interact", TopKInteractions(task_type=prefs.task_type)))
        if has_outlier_signal:
            numeric_steps.append(("clip", IQRClipper()))
        numeric_steps.append(("scale", StandardScaler()))

        transformers = [
            ("num", Pipeline(numeric_steps), groups.numeric),
            *self._categorical_transformers(groups, target_encode=opts.target_encoding),
            *self._datetime_transformer(groups, scale=True),
            *self._text_transformers(groups, max_features=opts.tfidf_max_features or 150),
            *self._shape_transformers(groups, opts, scale=True),
        ]
        preprocessor = ColumnTransformer(transformers, remainder="drop")
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("pca", PCA(n_components=opts.pca_variance, svd_solver="full")),
        ])
        decisions = [
            DecisionRecord(
                what="Stage 3: Yeo-Johnson skew correction + up to 3 target-relevant interaction "
                     "features (same as balanced), computed fresh per training fold",
                why="PCA compresses whatever it's given — feeding it skew-corrected, "
                    "interaction-enriched features gives it better raw material than the "
                    "untouched originals would",
                evidence="scipy skew + f_regression/f_classif scoring, training fold only",
                alternatives="No feature construction (conservative); keep original features "
                             "uncompressed (balanced)",
                risk="Interaction features add noise for PCA to compress away if they don't "
                     "actually help — mitigated by scoring against the target before including them",
                effect="Up to 3 new columns feed into PCA alongside the originals",
                reversible=True,
            ),
            DecisionRecord(
                what=(
                    f"Clip outliers to IQR bounds (train-fold only), scale, encode, then PCA to "
                    f"{opts.pca_variance:.0%} explained variance"
                    if has_outlier_signal else
                    f"Scale, encode, then PCA to {opts.pca_variance:.0%} explained variance (no numeric column showed IQR outliers)"
                ),
                why="Tests transformed-component representation against selected originals; acts on the outliers Stage 1 detected instead of only reporting them",
                evidence=f"Stage 1 outlier detection: {'outliers present in at least one numeric column' if has_outlier_signal else 'no IQR outliers found'}",
                alternatives="Balanced pipeline (keeps original, interpretable features, no clipping)",
                risk="Loses direct feature interpretability; clipping can suppress genuine extreme values, not just noise",
                effect=f"Dimensionality reduced to components explaining {opts.pca_variance:.0%} variance",
                reversible=False,
            )
        ]
        if groups.free_text:
            decisions.append(DecisionRecord(
                what=f"TF-IDF (up to 150 terms/column) on {len(groups.free_text)} free-text column(s): "
                     f"{', '.join(groups.free_text)}, compressed into PCA alongside everything else",
                why="Same vocabulary size as balanced; PCA still compresses whatever it's given, so a "
                    "richer input (real text signal) gives it better raw material than dropping the "
                    "column would",
                evidence="Stage 1 detected average word count >= 5 for this column (free text, not an identifier)",
                alternatives="Keep TF-IDF features uncompressed (balanced); smaller vocabulary (conservative)",
                risk="TF-IDF ignores word order and context; PCA further removes direct term interpretability",
                effect="Up to 150 new sparse term-frequency columns per free-text column, feeding into PCA",
                reversible=True,
            ))
        return PipelineCandidate("aggressive", pipeline, decisions,
                                  "Feature-engineered, outlier-clipped, scaled, PCA-reduced")
