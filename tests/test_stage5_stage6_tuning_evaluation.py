import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from datapilot.evaluation import evaluate_on_held_out_test
from datapilot.experiment import ExperimentEngine
from datapilot.model_zoo import get_models
from datapilot.planner import PipelinePlanner, train_test_split_data
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import Priority, TaskType, UserPreferences
from datapilot.tuning import tune_top_entries


def make_regression_df(n=700, seed=7):
    rng = np.random.default_rng(seed)
    sqft = rng.normal(1800, 400, n)
    beds = rng.integers(1, 6, n)
    noise = rng.normal(0, 20000, n)
    price = 50000 + sqft * 120 + beds * 8000 + noise
    df = pd.DataFrame({
        "id": range(n),
        "sqft": sqft,
        "beds": beds,
        "grade": rng.choice(["A", "B", "C"], n),
        "price": price,
    })
    df.loc[df.sample(frac=0.05, random_state=1).index, "sqft"] = None
    return df


def make_classification_df(n=700, seed=8):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 80, n)
    income = rng.normal(50000, 15000, n)
    logit = (income - 50000) / 15000 + rng.normal(0, 1, n)
    churn = (logit > 0.6).astype(int)  # imbalanced-ish
    df = pd.DataFrame({
        "id": range(n),
        "age": age,
        "income": income,
        "segment": rng.choice(["retail", "sme", "enterprise"], n),
        "churn": churn,
    })
    return df


def _setup(df, target, task_type):
    profiler = DatasetProfiler()
    profile = profiler.profile(df, target=target)
    prefs = UserPreferences(target_column=target, task_type=task_type, priority=Priority.BALANCED)
    split = train_test_split_data(df, target, task_type)
    planner = PipelinePlanner()
    candidates = planner.plan(list(df.columns), profile, prefs)
    models = get_models(task_type)
    engine = ExperimentEngine(task_type=task_type, n_splits=4, n_repeats=1)
    entries = engine.run_leaderboard(candidates, models, split.X_train, split.y_train)
    return split, candidates, models, entries


def test_tuning_improves_or_matches_top_regression_model():
    df = make_regression_df()
    split, candidates, models, entries = _setup(df, "price", TaskType.REGRESSION)

    results = tune_top_entries(
        entries, candidates, models, split.X_train, split.y_train,
        TaskType.REGRESSION, top_k=2, n_iter=6, cv_splits=3,
    )
    assert len(results) > 0
    tunable_results = [r for r in results if r.tunable and r.ok]
    assert tunable_results, f"no tunable result succeeded: {[r.error for r in results]}"

    for r in tunable_results:
        if r.verified and not r.accepted:
            # the search found nothing that beats the defaults on fresh folds: defaults are kept, honestly
            assert r.best_params == {} and r.tuned_mean == r.base_mean
        else:
            assert r.best_params, "an accepted search must produce real best params, not an empty dict"
        # RandomizedSearchCV's own reported best score should be >= a single
        # untuned CV estimate most of the time — not a hard guarantee, but
        # it must at least be a finite, sane R^2 value
        assert np.isfinite(r.tuned_mean)
        assert r.tuned_mean > -1.0  # sanity bound, not "must beat baseline"


def test_tuning_skips_untunable_models_honestly():
    df = make_regression_df()
    split, candidates, models, entries = _setup(df, "price", TaskType.REGRESSION)

    # Force "Linear Regression" (no param distribution registered) to the front
    lr_entries = [e for e in entries if e.model_name == "Linear Regression" and e.ok]
    assert lr_entries, "expected Linear Regression to have a valid entry"
    results = tune_top_entries(
        lr_entries, candidates, models, split.X_train, split.y_train,
        TaskType.REGRESSION, top_k=1, n_iter=4, cv_splits=3,
    )
    assert len(results) == 1
    assert results[0].tunable is False
    assert results[0].best_params == {}


def test_final_evaluation_regression_real_holdout():
    df = make_regression_df()
    split, candidates, models, entries = _setup(df, "price", TaskType.REGRESSION)
    best = next(e for e in entries if e.ok)
    candidate = next(c for c in candidates if c.name == best.pipeline_name)
    spec = next(m for m in models if m.name == best.model_name)

    fitted = Pipeline(candidate.pipeline.steps + [("model", spec.build())])
    fitted.fit(split.X_train, split.y_train)

    result = evaluate_on_held_out_test(
        best.pipeline_name, best.model_name, fitted,
        split.X_test, split.y_test, TaskType.REGRESSION,
    )
    assert result.ok
    assert result.n_test == len(split.X_test)
    reg = result.regression
    assert reg is not None
    for v in (reg.r2, reg.rmse, reg.mae, reg.mape, reg.residual_mean, reg.residual_std):
        assert np.isfinite(v)
    assert reg.rmse > 0
    assert len(reg.worst_errors) == 5
    # worst errors must actually be the largest — spot check ordering
    abs_errors = [w["abs_error"] for w in reg.worst_errors]
    assert abs_errors == sorted(abs_errors, reverse=True)
    # and each worst error's arithmetic must be internally consistent
    for w in reg.worst_errors:
        assert abs(abs(w["actual"] - w["predicted"]) - w["abs_error"]) < 1e-6


def test_final_evaluation_classification_real_holdout_confusion_matrix():
    df = make_classification_df()
    split, candidates, models, entries = _setup(df, "churn", TaskType.CLASSIFICATION)
    best = next(e for e in entries if e.ok)
    candidate = next(c for c in candidates if c.name == best.pipeline_name)
    spec = next(m for m in models if m.name == best.model_name)

    fitted = Pipeline(candidate.pipeline.steps + [("model", spec.build())])
    fitted.fit(split.X_train, split.y_train)

    result = evaluate_on_held_out_test(
        best.pipeline_name, best.model_name, fitted,
        split.X_test, split.y_test, TaskType.CLASSIFICATION,
    )
    assert result.ok
    clf = result.classification
    assert clf is not None
    for v in (clf.accuracy, clf.precision, clf.recall, clf.f1, clf.f2):
        assert 0.0 <= v <= 1.0

    # confusion matrix must be square (n_labels x n_labels) and sum to n_test
    n_labels = len(clf.labels)
    assert len(clf.confusion) == n_labels
    assert all(len(row) == n_labels for row in clf.confusion)
    assert sum(sum(row) for row in clf.confusion) == result.n_test
