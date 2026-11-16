import numpy as np
import pandas as pd

from datapilot.experiment import ExperimentEngine
from datapilot.model_zoo import get_models, unavailable_libraries
from datapilot.planner import PipelinePlanner, train_test_split_data
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import Priority, TaskType, UserPreferences


def make_regression_df(n=600, seed=1):
    rng = np.random.default_rng(seed)
    cities = [f"city_{i}" for i in range(120)]
    sqft = rng.normal(1800, 400, n)
    beds = rng.integers(1, 6, n)
    noise = rng.normal(0, 20000, n)
    price = 50000 + sqft * 120 + beds * 8000 + noise  # computed BEFORE injecting missingness

    df = pd.DataFrame({
        "id": range(n),
        "sqft": sqft,
        "beds": beds,
        "grade": rng.choice(["A", "B", "C"], n),
        "city": rng.choice(cities, n),
        "price": price,
    })
    df.loc[df.sample(frac=0.05, random_state=2).index, "sqft"] = None
    return df


def make_classification_df(n=600, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id": range(n),
        "age": rng.integers(18, 80, n),
        "income": rng.normal(50000, 15000, n),
        "segment": rng.choice(["retail", "sme", "enterprise"], n),
    })
    df.loc[df.sample(frac=0.05, random_state=4).index, "income"] = None
    # imbalanced-ish binary target, driven partly by income
    logit = (df["income"] - 50000) / 15000 + rng.normal(0, 1, n)
    df["churn"] = (logit > 0.6).astype(int)
    return df


def _run_leaderboard_for(df, target, task_type):
    profiler = DatasetProfiler()
    profile = profiler.profile(df, target=target)
    prefs = UserPreferences(target_column=target, task_type=task_type, priority=Priority.BALANCED)

    X_train, X_test, y_train, y_test = train_test_split_data(df, target, task_type)

    planner = PipelinePlanner()
    candidates = planner.plan(list(df.columns), profile, prefs)

    models = get_models(task_type)
    assert len(models) >= 4, "expected at least baseline + 3 real models"
    assert models[0].is_baseline, "baseline must be first so every model has a floor to beat"

    engine = ExperimentEngine(task_type=task_type, n_splits=4, n_repeats=1)
    entries = engine.run_leaderboard(candidates, models, X_train, y_train)
    return entries, candidates, models, X_test, y_test


def test_model_zoo_lists_expected_models_and_reports_missing_libs():
    reg_models = get_models(TaskType.REGRESSION)
    clf_models = get_models(TaskType.CLASSIFICATION)
    reg_names = {m.name for m in reg_models}
    clf_names = {m.name for m in clf_models}

    assert "Baseline (mean)" in reg_names
    assert "Linear Regression" in reg_names
    assert "Random Forest" in reg_names
    assert "Baseline (most frequent)" in clf_names
    assert "Logistic Regression" in clf_names

    missing = unavailable_libraries()
    assert isinstance(missing, list)  # must not crash whether or not xgboost/lightgbm installed


def test_leaderboard_regression_real_data_baseline_gets_beaten():
    df = make_regression_df()
    entries, candidates, models, X_test, y_test = _run_leaderboard_for(df, "price", TaskType.REGRESSION)

    boosters = [e for e in entries if e.model_name.startswith("Ensemble: ") or e.model_name.endswith((" + threshold", " + log target"))]
    assert len(entries) - len(boosters) == len(candidates) * len(models)  # every pipeline x model scored once
    assert len(boosters) >= 1  # plus the accuracy boosters (ensemble at least)
    ok_entries = [e for e in entries if e.ok]
    assert len(ok_entries) > 0, f"every entry failed: {[e.error for e in entries if not e.ok]}"

    # leaderboard must be sorted best-first (R^2 higher is better)
    r2_values = [e.primary_mean for e in ok_entries]
    assert r2_values == sorted(r2_values, reverse=True)

    # a real model must beat the dummy mean-baseline on data with genuine signal
    baseline_entries = [e for e in ok_entries if e.is_baseline]
    real_entries = [e for e in ok_entries if not e.is_baseline]
    assert baseline_entries and real_entries
    best_real = max(e.primary_mean for e in real_entries)
    best_baseline = max(e.primary_mean for e in baseline_entries)
    assert best_real > best_baseline, "a real model should beat the dummy baseline when there's real signal"

    # all four regression metrics must be present and finite
    top = ok_entries[0]
    for key in ("r2", "rmse", "mae", "mape"):
        assert key in top.metrics
        assert np.isfinite(top.metrics[key])
    assert top.metrics["rmse"] > 0  # sign-corrected, must read positive


def test_leaderboard_classification_real_data():
    df = make_classification_df()
    entries, candidates, models, X_test, y_test = _run_leaderboard_for(df, "churn", TaskType.CLASSIFICATION)

    ok_entries = [e for e in entries if e.ok]
    assert len(ok_entries) > 0, f"every entry failed: {[e.error for e in entries if not e.ok]}"

    top = ok_entries[0]
    for key in ("accuracy", "precision", "recall", "f1", "f2", "roc_auc"):
        assert key in top.metrics
        assert np.isfinite(top.metrics[key])
    assert 0.0 <= top.metrics["accuracy"] <= 1.0
    assert 0.0 <= top.metrics["roc_auc"] <= 1.0


def test_best_pipeline_and_model_actually_fit_and_predict_on_held_out_test():
    """The real end-to-end proof: take the winning (pipeline, model) combo
    from the leaderboard, fit it fresh on the training split, and confirm
    it produces predictions on the untouched test split — this is Stage 6's
    held-out evaluation, exercised for real."""
    df = make_regression_df()
    entries, candidates, models, X_test, y_test = _run_leaderboard_for(df, "price", TaskType.REGRESSION)
    best = next(e for e in entries if e.ok)

    candidate = next(c for c in candidates if c.name == best.pipeline_name)
    spec = next(m for m in models if m.name == best.model_name)

    from sklearn.pipeline import Pipeline
    final_pipeline = Pipeline(candidate.pipeline.steps + [("model", spec.build())])

    X_train, X_test2, y_train, y_test2 = train_test_split_data(df, "price", TaskType.REGRESSION)
    final_pipeline.fit(X_train, y_train)
    preds = final_pipeline.predict(X_test2)

    assert len(preds) == len(X_test2)
    assert np.all(np.isfinite(preds))


def test_missing_target_rows_are_dropped_not_crashed_on():
    """The real bug this pass found: a target column with missing values
    used to crash every model at fit time with 'Input y contains NaN'.
    Now those rows are dropped before the split, and the count is reported."""
    df = make_regression_df()
    n_missing = 15
    idx = df.sample(n=n_missing, random_state=99).index
    df.loc[idx, "price"] = None

    result = train_test_split_data(df, "price", TaskType.REGRESSION)
    assert result.n_dropped_missing_target == n_missing
    assert len(result.y_train) + len(result.y_test) == len(df) - n_missing
    assert result.y_train.isna().sum() == 0
    assert result.y_test.isna().sum() == 0

    # unpacking as a 4-tuple must still work for existing call sites
    X_train, X_test, y_train, y_test = train_test_split_data(df, "price", TaskType.REGRESSION)
    assert len(X_train) == len(y_train)
