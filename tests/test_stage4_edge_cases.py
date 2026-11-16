import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from datapilot.evaluation import evaluate_on_held_out_test
from datapilot.experiment import ExperimentEngine
from datapilot.model_zoo import get_models
from datapilot.planner import PipelinePlanner, train_test_split_data
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import Priority, TaskType, UserPreferences


def make_base_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n) * 1000,
        "cat": rng.choice(list("ABC"), n),
    })
    signal = base["x1"] * 2 + base["x2"] / 1000
    return base, signal


def _run(df, target, n_splits=3):
    profiler = DatasetProfiler()
    profile = profiler.profile(df, target=target)
    prefs = UserPreferences(target_column=target, task_type=TaskType.CLASSIFICATION, priority=Priority.BALANCED)
    split = train_test_split_data(df, target, TaskType.CLASSIFICATION)
    planner = PipelinePlanner()
    candidates = planner.plan(list(df.columns), profile, prefs)
    models = get_models(TaskType.CLASSIFICATION)
    engine = ExperimentEngine(task_type=TaskType.CLASSIFICATION, n_splits=n_splits, n_repeats=1)
    entries = engine.run_leaderboard(candidates, models, split.X_train, split.y_train)
    return engine, entries, candidates, models, split


def test_string_labels_do_not_break_xgboost():
    """Regression test for a real bug: XGBoost's sklearn API raises
    'Invalid classes inferred from unique values of y' on any label format
    other than contiguous 0..n-1 integers — including ordinary string
    labels like 'yes'/'no', which every other model in the zoo handles
    fine. train_test_split_data must label-encode the target for
    classification so every model gets a label format it can actually use.
    """
    base, signal = make_base_df()
    df = base.copy()
    df["y"] = np.where(signal > 0, "yes", "no")

    engine, entries, candidates, models, split = _run(df, "y")
    xgb_entries = [e for e in entries if e.model_name.startswith("XGBoost")]
    assert xgb_entries, "expected XGBoost entries to exist at all"
    failed = [e for e in xgb_entries if not e.ok]
    assert not failed, f"XGBoost should handle string labels via encoding, got: {[e.error for e in failed]}"

    # label_encoder must actually be fit and able to round-trip real class names
    assert split.label_encoder is not None
    assert set(split.label_encoder.classes_) == {"no", "yes"}


def test_non_zero_indexed_int_labels_do_not_break_xgboost():
    base, signal = make_base_df()
    df = base.copy()
    df["y"] = pd.qcut(signal, 3, labels=[1, 2, 3]).astype(int)

    engine, entries, candidates, models, split = _run(df, "y")
    xgb_entries = [e for e in entries if e.model_name.startswith("XGBoost")]
    failed = [e for e in xgb_entries if not e.ok]
    assert not failed, f"XGBoost should handle 1/2/3 labels via encoding, got: {[e.error for e in failed]}"


def test_singleton_class_does_not_crash_the_split():
    """Regression test for a real bug: a class with only 1 row crashed
    train_test_split_data outright with sklearn's stratify ValueError —
    not a graceful per-model failure, a hard crash before anything else
    could run. Real datasets have rare/singleton classes constantly (a
    one-off data-entry category, a rare fraud type). Must fall back to a
    non-stratified split instead, with the fallback explicitly reported.
    """
    base, signal = make_base_df()
    df = base.copy()
    df["y"] = (signal > 0).astype(int)
    df.loc[df.index[0], "y"] = 2  # class '2' now has exactly 1 row

    split = train_test_split_data(df, "y", TaskType.CLASSIFICATION)  # must not raise
    assert split.stratified is False
    assert "1 row" in split.stratify_note or "1" in split.stratify_note
    assert len(split.y_train) + len(split.y_test) == len(df)

    # the full leaderboard sweep must also complete without crashing
    engine, entries, candidates, models, split2 = _run(df, "y")
    assert len(entries) > 0
    assert any(e.ok for e in entries)


def test_confusion_matrix_shows_real_class_names_not_encoded_integers():
    base, signal = make_base_df()
    df = base.copy()
    df["y"] = np.where(signal > 0, "yes", "no")

    engine, entries, candidates, models, split = _run(df, "y")
    best = next(e for e in entries if e.ok)
    candidate = next(c for c in candidates if c.name == best.pipeline_name)
    base_name = best.model_name.removesuffix(" + SMOTE")
    spec = next(m for m in models if m.name == base_name)
    fitted = Pipeline(candidate.pipeline.steps + [("model", spec.build())])
    fitted.fit(split.X_train, split.y_train)

    result = evaluate_on_held_out_test(
        best.pipeline_name, best.model_name, fitted,
        split.X_test, split.y_test, TaskType.CLASSIFICATION,
        label_encoder=split.label_encoder,
    )
    assert result.ok
    assert set(result.classification.labels) == {"yes", "no"}, (
        f"confusion matrix labels should be decoded class names, got: {result.classification.labels}"
    )


def test_cv_fold_note_flags_small_minority_class_without_crashing():
    base, signal = make_base_df()
    df = base.copy()
    y = (signal > 0).astype(int)
    df["y"] = y
    # collapse the minority class down to 3 rows — fewer than n_splits=5,
    # enough to survive the 2-per-class stratify requirement in the split
    minority_idx = df[df["y"] == 1].index
    keep = list(minority_idx[:3])
    drop = [i for i in minority_idx if i not in keep]
    df.loc[drop, "y"] = 0

    engine, entries, candidates, models, split = _run(df, "y", n_splits=5)
    assert len(entries) > 0  # must not crash
    assert engine.cv_fold_note, "expected a real note about the small minority class"
    assert "fold" in engine.cv_fold_note.lower()

    # the note is a heads-up, not an excuse: every metric except possibly
    # roc_auc (which can legitimately go NaN on a fold missing the rare
    # class — see _safe_roc_auc_ovr_weighted) must still be a real number
    for e in entries:
        if not e.ok:
            continue
        for key in ("accuracy", "precision", "recall", "f1", "f2"):
            assert np.isfinite(e.metrics[key]), f"{e.model_name}: {key} should be finite, got {e.metrics[key]}"


def test_aggressive_pipeline_pca_survives_wide_categorical_data():
    """Regression test for a real bug found on the actual Ames Housing
    benchmark: OneHotEncoder defaults to sparse output, and ColumnTransformer
    propagates that sparsity to the WHOLE combined matrix once categoricals
    make up enough of it — even when the numeric branch (StandardScaler,
    IQRClipper) produced dense output on its own. PCA's svd_solver='arpack'
    (auto-selected for sparse input) then crashes outright whenever
    explaining 95% variance requires close to the full component count,
    which is routine on wide, heavily one-hot-encoded real data. Fixed with
    OneHotEncoder(sparse_output=False), keeping the whole pipeline
    consistently dense like every other custom transformer already is.

    Reproduces the shape that triggered it: many low-cardinality categorical
    columns (each exploding into several one-hot columns) relative to a
    modest row count, so 95%-variance PCA needs close to full rank.
    """
    rng = np.random.default_rng(42)
    n = 400
    data = {"id": range(n)}
    # 20 categorical columns, 4-6 categories each — one-hot explodes this
    # to ~100 columns, mimicking Ames Housing's real categorical density
    for i in range(20):
        data[f"cat_{i}"] = rng.choice(list("ABCDE"), n)
    data["num1"] = rng.normal(size=n)
    data["num2"] = rng.normal(size=n)
    target = data["num1"] * 50 + data["num2"] * 20 + rng.normal(0, 5, n)
    data["target"] = target
    df = pd.DataFrame(data)

    profile = DatasetProfiler().profile(df, target="target")
    prefs = UserPreferences(target_column="target", task_type=TaskType.REGRESSION, priority=Priority.BALANCED)
    split = train_test_split_data(df, "target", TaskType.REGRESSION)
    candidates = PipelinePlanner().plan(list(df.columns), profile, prefs)
    aggressive = next(c for c in candidates if c.name == "aggressive")

    aggressive.pipeline.fit(split.X_train, split.y_train)  # must not raise
    Xt_train = aggressive.pipeline.transform(split.X_train)
    Xt_test = aggressive.pipeline.transform(split.X_test)
    assert np.all(np.isfinite(Xt_train))
    assert np.all(np.isfinite(Xt_test))
    assert Xt_train.shape[1] == Xt_test.shape[1]


def test_rare_missingness_does_not_crash_balanced_pipeline_on_unlucky_split():
    """Regression test for a real bug found on a real, large (700k-row)
    Kaggle dataset: a column with very RARE missingness (0.04% in the real
    case) can, purely by sampling variance, end up with zero missing rows
    in the training split but one or more missing rows in the test split.
    sklearn's MissingIndicator defaults to error_on_new=True and crashes
    outright in exactly that situation — not a hypothetical, this is
    routine for any rare-missingness column under normal train/test
    splitting or cross-validation, on any dataset. Fixed with
    error_on_new=False, which reports "not missing" for that feature on
    rows the model never learned could be missing, rather than crashing.
    """
    rng = np.random.default_rng(7)
    n = 500
    df = pd.DataFrame({
        "id": range(n),
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "rarely_missing": rng.normal(size=n),
        "cat": rng.choice(["X", "Y"], n),
    })
    # Construct the exact failure condition directly: make sure the
    # missing row lands in the LAST few rows, then force a split where
    # that row is held out for test and never seen during fit — this
    # reproduces "no missing values in fit, but missing in transform"
    # deterministically rather than hoping a random split happens to do it.
    df.loc[n - 1, "rarely_missing"] = None
    target = (df["a"] + df["b"] > 0).astype(int)
    df["y"] = target

    profile = DatasetProfiler().profile(df, target="y")
    prefs = UserPreferences(target_column="y", task_type=TaskType.CLASSIFICATION, priority=Priority.BALANCED)
    planner = PipelinePlanner()
    candidates = planner.plan(list(df.columns), profile, prefs)
    balanced = next(c for c in candidates if c.name == "balanced")

    # fit on everything EXCEPT the one row with the missing value, then
    # transform data that DOES include it — must not raise
    fit_idx = df.index[df.index != n - 1]
    X = df.drop(columns=["y"])
    balanced.pipeline.fit(X.loc[fit_idx], target.loc[fit_idx])
    result = balanced.pipeline.transform(X)  # includes the row with the missing value
    assert result.shape[0] == n
    assert np.all(np.isfinite(result))
