"""Guided setup / audit report / project export cross-checks against real Stage 1-6 outputs."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from datapilot import (
    DatasetProfiler,
    ExperimentEngine,
    PipelinePlanner,
    Priority,
    TaskType,
    UserPreferences,
    build_audit_report,
    evaluate_on_held_out_test,
    get_models,
    interpret,
    train_test_split_data,
    tune_top_entries,
)


def _run(df, target, task, tune=False):
    profile = DatasetProfiler().profile(df, target=target)
    prefs = UserPreferences(target_column=target, task_type=task, priority=Priority.BALANCED)
    split = train_test_split_data(df, target, task)
    cands = PipelinePlanner().plan(list(df.columns), profile, prefs)
    models = get_models(task)
    entries = ExperimentEngine(task_type=task, n_splits=3, n_repeats=1).run_leaderboard(
        cands, models, split.X_train, split.y_train)
    ok = [e for e in entries if e.ok]
    best = max(ok, key=lambda e: e.primary_mean)
    tuning = tune_top_entries(entries, cands, models, split.X_train, split.y_train, task, top_k=1, n_iter=2, cv_splits=3) if tune else []
    cand = next(c for c in cands if c.name == best.pipeline_name)
    spec = next(m for m in models if m.name == best.model_name)
    fitted = Pipeline(cand.pipeline.steps + [("model", spec.build())]).fit(split.X_train, split.y_train)
    ev = evaluate_on_held_out_test(best.pipeline_name, best.model_name, fitted, split.X_test, split.y_test,
                                   task, label_encoder=split.label_encoder)
    interp = interpret(fitted, X_train=split.X_train, top_n=8)
    return profile, prefs, cands, entries, tuning, ev, interp


@pytest.fixture(scope="module")
def clf_run():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "id": range(n),
        "age": rng.normal(40, 10, n),
        "city": rng.choice(["a", "b", "c"], n),
        "notes": rng.choice(["pipe | in text", "plain"], n),
    })
    df["label"] = np.where(df["age"] + rng.normal(0, 5, n) > 40, "yes", "no")  # string labels
    return _run(df, "label", TaskType.CLASSIFICATION, tune=True)


@pytest.fixture(scope="module")
def reg_run():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n), "c": rng.choice(["u", "v"], n)})
    df["y"] = 3 * df.x1 + df.x2 + rng.normal(0, .3, n)
    return _run(df, "y", TaskType.REGRESSION)


def test_report_and_context_match_real_classification_results(clf_run):
    profile, prefs, cands, entries, tuning, ev, interp = clf_run
    md = build_audit_report(profile, prefs, cands, entries, tuning, ev, interp)
    assert ev.ok
    assert f"{ev.classification.accuracy:.4f}" in md
    for c in cands:
        assert f"Pipeline: {c.name}" in md
    assert "yes" in md and "no" in md  # decoded string labels, not 0/1





def test_regression_report(reg_run):
    profile, prefs, cands, entries, tuning, ev, interp = reg_run
    md = build_audit_report(profile, prefs, cands, entries, tuning, ev, interp)
    assert f"{ev.regression.r2:.4f}" in md
    assert "Tuning was not run" in md


def test_failed_entries_and_missing_pieces_do_not_crash(clf_run):
    profile, prefs, cands, entries, _, _, _ = clf_run
    from datapilot.experiment import LeaderboardEntry
    bad = LeaderboardEntry("conservative", "boom", False, "f1", 0.0, 0.0, error="kaboom | x")
    md = build_audit_report(profile, prefs, cands, entries + [bad], None, None, None)
    assert "FAILED" in md and "kaboom" in md
    assert "not run" in md













# ---------------------------------------------------------------- setup ---
def test_target_suggestion_and_task_inference():
    from datapilot import infer_task_type, suggest_input_exclusions, suggest_targets
    rng = np.random.default_rng(3)
    n = 300
    df = pd.DataFrame({"row_id": range(n), "age": rng.normal(40, 9, n), "const": 1,
                       "notes": ["free text with many words in it " + str(i) for i in range(n)]})
    df["Survived"] = rng.integers(0, 2, n)
    profile = DatasetProfiler().profile(df)
    sug = suggest_targets(df, profile)
    assert sug[0].column == "Survived" and sug[0].task_type == TaskType.CLASSIFICATION
    assert all(s.column not in ("row_id", "const", "notes") for s in sug)
    ex = suggest_input_exclusions(profile, "Survived")
    assert "row_id" in ex and "const" in ex and "Survived" not in ex
    assert infer_task_type(pd.Series(rng.normal(size=100))) == TaskType.REGRESSION
    assert infer_task_type(pd.Series(["a", "b"] * 50)) == TaskType.CLASSIFICATION


# -------------------------------------------------------------- options ---
def test_pipeline_options_change_real_pipelines_and_default_is_noop():
    from datapilot import PipelineOptions
    rng = np.random.default_rng(4)
    n = 200
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.exponential(size=n), "c": rng.choice(["x", "y"], n)})
    df["t"] = df.a * 2 + rng.normal(size=n)
    profile = DatasetProfiler().profile(df, target="t")
    prefs = UserPreferences(target_column="t", task_type=TaskType.REGRESSION)
    base = PipelinePlanner().plan(list(df.columns), profile, prefs)
    same = PipelinePlanner().plan(list(df.columns), profile, prefs, PipelineOptions())
    assert [c.name for c in base] == [c.name for c in same]
    assert [len(c.decisions) for c in base] == [len(c.decisions) for c in same]

    opts = PipelineOptions(pipelines=["aggressive"], excluded_columns=["b"], interactions=False,
                           skew_correction=False, pca_variance=0.8, numeric_impute="mean")
    cands = PipelinePlanner().plan(list(df.columns), profile, prefs, opts)
    assert [c.name for c in cands] == ["aggressive"]
    split = train_test_split_data(df, "t", TaskType.REGRESSION)
    cands[0].pipeline.fit(split.X_train, split.y_train)
    pca = cands[0].pipeline.named_steps["pca"]
    assert pca.n_components == 0.8
    num = cands[0].pipeline.named_steps["preprocess"].named_transformers_["num"]
    assert [n for n, _ in num.steps] == ["impute", "clip", "scale"] or [n for n, _ in num.steps] == ["impute", "scale"]
    assert "b" not in cands[0].pipeline.named_steps["preprocess"].transformers_[0][2]
    assert any("User override" in d.what for d in cands[0].decisions)
    for bad in [dict(pipelines=["nope"]), dict(numeric_impute="x"), dict(pca_variance=1.5), dict(clip_outliers="maybe")]:
        with pytest.raises(ValueError):
            PipelineOptions(**bad)


# ----------------------------------------------------- downloadable project ---
def test_project_zip_runs_standalone_and_reproduces(tmp_path):
    import io
    import subprocess
    import sys
    import zipfile

    from datapilot import PipelineOptions, build_project_zip
    rng = np.random.default_rng(5)
    n = 400
    df = pd.DataFrame({"rid": range(n), "age": rng.normal(40, 10, n), "city": rng.choice(["a", "b", "c"], n)})
    df["label"] = np.where(df.age + rng.normal(0, 4, n) > 40, "yes", "no")
    csv = tmp_path / "data.csv"; df.to_csv(csv, index=False)

    opts = PipelineOptions(excluded_columns=["rid"], pipelines=["conservative"])
    task = TaskType.CLASSIFICATION
    profile = DatasetProfiler().profile(df, target="label")
    prefs = UserPreferences(target_column="label", task_type=task)
    df_model = df.drop(columns=["rid"])
    split = train_test_split_data(df_model, "label", task)
    cand = PipelinePlanner().plan(list(df.columns), profile, prefs, opts)[0]
    spec = next(m for m in get_models(task) if not m.is_baseline)
    fitted = Pipeline(cand.pipeline.steps + [("model", spec.build())]).fit(split.X_train, split.y_train)

    blob = build_project_zip(
        target="label", task_type_value=task.value, priority_value="balanced", preserve_missingness=True,
        options=opts, pipeline_name=cand.name, model_name=spec.name, best_params={}, uses_smote=False,
        feature_columns=list(split.X_train.columns), fitted_pipeline=fitted, label_encoder=split.label_encoder)
    out = tmp_path / "proj"; zipfile.ZipFile(io.BytesIO(blob)).extractall(out)
    proj = out / "datapilot_project"
    assert (proj / "train.py").exists() and (proj / "datapilot" / "planner.py").exists()
    assert not (proj / "datapilot" / "app.py").exists()

    r = subprocess.run([sys.executable, "train.py", str(csv)], cwd=proj, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-1500:]
    assert "accuracy" in r.stdout
    # the shipped project's own library copy must be what ran
    chk = subprocess.run([sys.executable, "-c", "import datapilot;print(datapilot.__file__)"],
                         cwd=proj, capture_output=True, text=True)
    assert str(proj) in chk.stdout

    new = df.drop(columns=["label"]).head(5); new.to_csv(tmp_path / "new.csv", index=False)
    r2 = subprocess.run([sys.executable, "predict.py", str(tmp_path / "new.csv")], cwd=proj,
                        capture_output=True, text=True, timeout=120)
    assert r2.returncode == 0, r2.stderr[-1500:]
    assert "yes" in r2.stdout or "no" in r2.stdout
