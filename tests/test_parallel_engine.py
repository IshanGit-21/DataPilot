"""Parallel engine: same numbers as cross_validate, early stopping, errors, RMSLE mode, ensemble importances."""
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datapilot import parallel
from datapilot.interpretation import interpret
from datapilot.parallel import ModelJob, cross_validate_jobs
from datapilot.reasoner import TaskType
from datapilot.runner import RunConfig, build_state, finalize, race, tune

STEPS = [("scale", StandardScaler())]


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.uniform(size=n)})
    y = np.exp(0.8 * X["a"] + 0.3 * X["c"] + rng.normal(0, 0.2, n))
    return X, y


def test_matches_sklearn_cross_validate():
    X, y = _data()
    splits = list(KFold(4, shuffle=True, random_state=1).split(X))
    got = cross_validate_jobs(STEPS, [ModelJob("Ridge", Ridge(alpha=2.0))], X, y, splits, {"r2": "r2"})[0][0]
    ref = cross_validate(Pipeline(STEPS + [("m", Ridge(alpha=2.0))]), X, y, cv=splits, scoring={"r2": "r2"})
    assert np.allclose([f["r2"] for f in got], ref["test_r2"], atol=1e-10)


def test_one_bad_model_does_not_sink_the_others():
    class Boom(Ridge):
        def fit(self, X, y):
            raise ValueError("nope")
    X, y = _data()
    splits = list(KFold(3).split(X))
    res = cross_validate_jobs(STEPS, [ModelJob("Boom", Boom()), ModelJob("Ridge", Ridge())], X, y, splits, {"r2": "r2"})
    assert res[0][0] is None and "nope" in res[0][1]
    assert len(res[1][0]) == 3


def test_slow_model_that_trails_is_stopped_after_fold_one(monkeypatch):
    monkeypatch.setattr(parallel, "PRUNE_MIN_ROWS", 10)
    X, y = _data()
    splits = list(KFold(4, shuffle=True, random_state=1).split(X))
    jobs = [ModelJob("Ridge", Ridge()), ModelJob("Random Forest", DummyRegressor())]   # costly by name, scores ~0
    res = cross_validate_jobs(STEPS, jobs, X, y, splits, {"r2": "r2"}, primary="r2")
    assert len(res[0][0]) == 4 and len(res[1][0]) == 1


def test_sequential_mode_gives_same_scores(monkeypatch):
    X, y = _data()
    splits = list(KFold(3).split(X))
    many = cross_validate_jobs(STEPS, [ModelJob("Ridge", Ridge())], X, y, splits, {"r2": "r2"})
    monkeypatch.setenv("DATAPILOT_JOBS", "1")
    one = cross_validate_jobs(STEPS, [ModelJob("Ridge", Ridge())], X, y, splits, {"r2": "r2"})
    assert many[0][0] == one[0][0]


def test_rmsle_run_trains_on_log_target_and_reports_rmsle():
    X, y = _data(500)
    df = X.assign(price=y)
    cfg = RunConfig("price", TaskType.REGRESSION, effort="Quick", primary="rmsle")
    state = build_state(df, cfg, "t")
    assert state["log_target"]
    lb = race(state, cfg, "t")
    assert lb["primary"] == "rmsle" and all(e.primary_metric_name == "rmsle" for e in lb["entries"] if e.ok)
    tun = tune(state, lb, cfg, "t")
    fin = finalize(state, lb, tun, "t")
    reg = fin["result"].regression
    assert reg.rmsle is not None and reg.rmsle < 0.5
    preds = fin["final_pipeline"].predict(state["X_test"])
    assert (preds >= 0).all()   # converted back to the real scale


def test_rmsle_falls_back_when_target_has_negatives():
    X, y = _data(300)
    df = X.assign(price=y - y.mean())
    state = build_state(df, RunConfig("price", TaskType.REGRESSION, effort="Quick", primary="rmsle"), "t")
    assert not state["log_target"] and "never negative" in state["log_note"]


def test_ensemble_importances_are_explained():
    X, y = _data(300)
    ens = VotingRegressor([("a", RandomForestRegressor(30, random_state=0)), ("b", ExtraTreesRegressor(30, random_state=0))])
    pipe = Pipeline(STEPS + [("model", ens)]).fit(X, y)
    r = interpret(pipe, X)
    assert r.kind == "tree_importances" and r.features[0].feature.endswith("a")
