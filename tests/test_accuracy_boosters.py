import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from datapilot import (
    DatasetProfiler,
    ExperimentEngine,
    PipelineOptions,
    PipelinePlanner,
    TaskType,
    ThresholdTunedClassifier,
    UserPreferences,
    get_models,
    resolve_model_spec,
    train_test_split_data,
)
from datapilot.experiment import LeaderboardEntry
from datapilot.tuning import TuningResult, pick_best_entry


def _leaderboard(df, target, task):
    profile = DatasetProfiler().profile(df, target=target)
    prefs = UserPreferences(target_column=target, task_type=task)
    split = train_test_split_data(df, target, task)
    cands = PipelinePlanner().plan(list(df.columns), profile, prefs)
    entries = ExperimentEngine(task_type=task, n_splits=3, n_repeats=1).run_leaderboard(
        cands, get_models(task), split.X_train, split.y_train)
    return entries, cands, split


def test_binary_leaderboard_has_ensemble_and_threshold_entries_that_rebuild_from_name():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = ((df.a + df.b + rng.normal(0, 1, n)) > 1.2).astype(int)
    entries, cands, split = _leaderboard(df, "y", TaskType.CLASSIFICATION)
    names = [e.model_name for e in entries if e.ok]
    assert any(m.startswith("Ensemble: ") for m in names)
    assert any(m.endswith(" + threshold") for m in names)
    for e in entries:
        if e.ok and (e.model_name.startswith("Ensemble: ") or e.model_name.endswith(" + threshold")):
            spec = resolve_model_spec(e.model_name, TaskType.CLASSIFICATION)
            cand = next(c for c in cands if c.name == e.pipeline_name)
            fitted = Pipeline(cand.pipeline.steps + [("model", spec.build())]).fit(split.X_train, split.y_train)
            assert len(fitted.predict(split.X_test)) == len(split.X_test)


def test_threshold_classifier_picks_better_cutoff_than_half_on_imbalanced_data():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(1)
    n = 3000
    X = rng.normal(size=(n, 3))
    y = (X[:, 0] + rng.normal(0, 1.2, n) > 1.8).astype(int)  # ~7% positives
    Xtr, ytr, Xte, yte = X[:2000], y[:2000], X[2000:], y[2000:]
    plain = LogisticRegression().fit(Xtr, ytr)
    tuned = ThresholdTunedClassifier(LogisticRegression()).fit(Xtr, ytr)
    assert tuned.threshold_ < 0.5
    f_plain = f1_score(yte, plain.predict(Xte), average="macro")
    f_tuned = f1_score(yte, tuned.predict(Xte), average="macro")
    assert f_tuned >= f_plain - 0.01
    assert set(tuned.predict(Xte)) <= {0, 1}
    with pytest.raises(ValueError):
        ThresholdTunedClassifier(LogisticRegression()).fit(X, np.arange(n) % 3)


def test_log_target_booster_only_for_skewed_nonnegative_regression():
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["skewed"] = np.exp(1 + df.a + 0.3 * rng.normal(size=n))
    entries, _, _ = _leaderboard(df.drop(columns=[]), "skewed", TaskType.REGRESSION)
    assert any(e.model_name.endswith(" + log target") for e in entries if e.ok)
    df2 = df.copy(); df2["sym"] = df.a * 2 + rng.normal(size=n)
    entries2, _, _ = _leaderboard(df2.drop(columns=["skewed"]), "sym", TaskType.REGRESSION)
    assert not any(e.model_name.endswith(" + log target") for e in entries2)


def test_resolve_unknown_model_raises_and_pick_best_prefers_tuned_scores():
    with pytest.raises(KeyError):
        resolve_model_spec("Nope", TaskType.CLASSIFICATION)
    a = LeaderboardEntry("balanced", "A", False, "f1", 0.70, 0.01)
    b = LeaderboardEntry("balanced", "B", False, "f1", 0.65, 0.01)
    bad = LeaderboardEntry("balanced", "C", False, "f1", float("nan"), 0, error="x")
    tuned = TuningResult("balanced", "B", 0.65, 0.74, 0.01, best_params={"n": 5})
    best, params, score = pick_best_entry([a, b, bad], [tuned])
    assert best.model_name == "B" and params == {"n": 5} and score == 0.74
    best, params, _ = pick_best_entry([a, b], [])
    assert best.model_name == "A" and params == {}


def test_target_encoding_added_for_high_cardinality_and_toggle_works():
    rng = np.random.default_rng(3)
    n = 600
    df = pd.DataFrame({"x": rng.normal(size=n), "city": rng.choice([f"c{i}" for i in range(80)], n)})
    df["y"] = (df.x + rng.normal(size=n) > 0).astype(int)
    profile = DatasetProfiler().profile(df, target="y")
    prefs = UserPreferences(target_column="y", task_type=TaskType.CLASSIFICATION)
    split = train_test_split_data(df, "y", TaskType.CLASSIFICATION)
    on = {c.name: c for c in PipelinePlanner().plan(list(df.columns), profile, prefs)}
    off = {c.name: c for c in PipelinePlanner().plan(list(df.columns), profile, prefs, PipelineOptions(target_encoding=False))}
    names_on = [n for n, *_ in on["balanced"].pipeline.named_steps["preprocess"].transformers]
    names_off = [n for n, *_ in off["balanced"].pipeline.named_steps["preprocess"].transformers]
    assert "cat_high_te" in names_on and "cat_high_te" not in names_off
    assert "cat_high_te" not in [n for n, *_ in on["conservative"].pipeline.named_steps["preprocess"].transformers]
    on["balanced"].pipeline.fit(split.X_train, split.y_train)
    assert on["balanced"].pipeline.transform(split.X_test).shape[0] == len(split.X_test)


@pytest.mark.parametrize("model_name", ["Ensemble: Logistic Regression + Random Forest", "Logistic Regression + threshold"])
def test_exported_project_reproduces_booster_models(tmp_path, model_name):
    import io
    import subprocess
    import sys
    import zipfile

    from datapilot import build_project_zip
    rng = np.random.default_rng(7)
    n = 400
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = np.where(df.a + rng.normal(0, .8, n) > 0.3, "yes", "no")
    csv = tmp_path / "d.csv"; df.to_csv(csv, index=False)
    opts = PipelineOptions(pipelines=["conservative"])
    profile = DatasetProfiler().profile(df, target="y")
    prefs = UserPreferences(target_column="y", task_type=TaskType.CLASSIFICATION)
    split = train_test_split_data(df, "y", TaskType.CLASSIFICATION)
    cand = PipelinePlanner().plan(list(df.columns), profile, prefs, opts)[0]
    fitted = Pipeline(cand.pipeline.steps + [("model", resolve_model_spec(model_name, TaskType.CLASSIFICATION).build())]).fit(split.X_train, split.y_train)
    blob = build_project_zip(target="y", task_type_value="classification", priority_value="balanced",
                             preserve_missingness=True, options=opts, pipeline_name="conservative",
                             model_name=model_name, best_params={}, uses_smote=False,
                             feature_columns=list(split.X_train.columns), fitted_pipeline=fitted,
                             label_encoder=split.label_encoder)
    zipfile.ZipFile(io.BytesIO(blob)).extractall(tmp_path / "p")
    proj = tmp_path / "p" / "datapilot_project"
    r = subprocess.run([sys.executable, "train.py", str(csv)], cwd=proj, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1200:]
    assert "accuracy" in r.stdout
