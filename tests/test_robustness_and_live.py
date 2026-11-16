"""Fallback chain, backup models, live-status HTML."""
import numpy as np
import pandas as pd
import pytest

from datapilot.experiment import ExperimentEngine
from datapilot.finalize import fit_best_with_fallback
from datapilot.live import (
    chips_html,
    hero_bars_svg,
    pill_html,
    progress_html,
    steps_html,
)
from datapilot.model_zoo import get_models
from datapilot.planner import PipelineOptions, PipelinePlanner, train_test_split_data
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import Priority, TaskType, UserPreferences


def _data(n=240, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.choice(list("xyz"), n)})
    df["t"] = ((df.a + 0.5 * df.b + rng.normal(scale=.5, size=n)) > 0).astype(int)
    return df


def _setup(task=TaskType.CLASSIFICATION):
    df = _data()
    prof = DatasetProfiler().profile(df, target="t")
    prefs = UserPreferences(target_column="t", task_type=task, priority=Priority.BALANCED)
    opts = PipelineOptions()
    split = train_test_split_data(df, "t", task)
    pl = PipelinePlanner()
    return df, prof, split, pl.plan(list(df.columns), prof, prefs, opts), pl.plan_rescue(list(df.columns), prof, opts)


def test_backup_models_are_opt_in_and_row_limited():
    core = {m.name for m in get_models(TaskType.CLASSIFICATION)}
    full = {m.name for m in get_models(TaskType.CLASSIFICATION, backups=True)}
    assert {"KNN", "AdaBoost", "Neural Net (MLP)"} <= full - core
    big = {m.name for m in get_models(TaskType.CLASSIFICATION, backups=True, n_rows=500_000)}
    assert not ({"KNN", "Neural Net (MLP)", "AdaBoost"} & big)
    reg = {m.name for m in get_models(TaskType.REGRESSION, backups=True)}
    assert {"Ridge", "Elastic Net", "KNN"} <= reg


def test_backup_models_actually_fit_and_predict():
    df, prof, split, cands, _ = _setup()
    names = {"KNN", "AdaBoost", "Neural Net (MLP)"}
    models = [m for m in get_models(TaskType.CLASSIFICATION, backups=True) if m.name in names]
    entries = ExperimentEngine(task_type=TaskType.CLASSIFICATION, n_splits=3).run_leaderboard(
        cands[:1], models, split.X_train, split.y_train)
    assert all(e.ok and e.primary_mean > 0.6 for e in entries), [e.error for e in entries if not e.ok]


def test_final_fit_falls_back_when_best_model_cannot_be_refit(monkeypatch):
    df, prof, split, cands, rescue = _setup()
    models = [m for m in get_models(TaskType.CLASSIFICATION) if m.name in
              {"Baseline (most frequent)", "Logistic Regression", "Decision Tree"}]
    entries = ExperimentEngine(task_type=TaskType.CLASSIFICATION, n_splits=3).run_leaderboard(
        cands, models, split.X_train, split.y_train, rescue=rescue)
    best = max((e for e in entries if e.ok and not e.is_baseline), key=lambda e: e.primary_mean)

    import datapilot.finalize as fin
    real = fin.resolve_model_spec

    def flaky(name, task):
        if name == best.model_name:
            raise RuntimeError("simulated crash")
        return real(name, task)
    monkeypatch.setattr(fin, "resolve_model_spec", flaky)
    ff = fit_best_with_fallback(entries, cands, [], TaskType.CLASSIFICATION, split.X_train, split.y_train,
                                split.X_test, split.y_test, split.label_encoder, rescue=rescue)
    assert ff.model_name != best.model_name and ff.notes and "simulated crash" in ff.notes[0]
    assert ff.evaluation.ok


def test_final_fit_last_resort_uses_rescue_pipeline(monkeypatch):
    df, prof, split, cands, rescue = _setup()
    models = [m for m in get_models(TaskType.CLASSIFICATION) if m.name in {"Decision Tree"}]
    entries = ExperimentEngine(task_type=TaskType.CLASSIFICATION, n_splits=3).run_leaderboard(
        cands, models, split.X_train, split.y_train)
    import datapilot.ensemble as ens
    import datapilot.finalize as fin
    real = ens.resolve_model_spec

    def flaky(n, t):
        if n.startswith("Decision Tree"):
            raise RuntimeError("boom")
        return real(n, t)
    monkeypatch.setattr(fin, "resolve_model_spec", flaky)
    monkeypatch.setattr(ens, "resolve_model_spec", flaky)
    ff = fit_best_with_fallback(entries, cands, [], TaskType.CLASSIFICATION, split.X_train, split.y_train,
                                split.X_test, split.y_test, split.label_encoder, rescue=rescue)
    assert ff.model_name == "Logistic Regression" and ff.pipeline_name == "rescue" and "rescue" in ff.notes[0]


def test_final_fit_raises_clear_error_when_nothing_works(monkeypatch):
    df, prof, split, cands, rescue = _setup()
    import datapilot.finalize as fin
    monkeypatch.setattr(fin, "resolve_model_spec", lambda n, t: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="No model could be fitted"):
        fit_best_with_fallback([], cands, [], TaskType.CLASSIFICATION, split.X_train, split.y_train,
                               split.X_test, split.y_test, split.label_encoder, rescue=rescue)


def test_live_html_builders_escape_and_validate():
    h = steps_html("T<b>", [("a<script>", "d&", "done"), ("b", "", "active"), ("c", "", "pending")])
    assert "<script>" not in h and "&lt;script&gt;" in h and "dp-step done" in h
    with pytest.raises(ValueError):
        steps_html("t", [("a", "", "weird")])
    assert 'style="width:50%"' in progress_html("t", 5, 10)
    assert 'style="width:100%"' in progress_html("t", 0, 0, finished=True)
    assert "<img" not in chips_html("t", [("k<img>", "v")])
    assert "busy" in pill_html("x", busy=True) and "busy" not in pill_html("x")
    assert hero_bars_svg(10, seed=1) == hero_bars_svg(10, seed=1)
