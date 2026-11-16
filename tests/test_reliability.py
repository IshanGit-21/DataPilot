"""Score reliability: confidence interval, repeat-split check, and honest tuning."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from datapilot.reasoner import TaskType
from datapilot.stability import accuracy_interval, stability_check
from datapilot.tuning import MIN_TUNING_GAIN, TuningResult, _verify_against_defaults


def _xy(n=300, seed=0, signal=1.0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (signal * X["a"] + rng.normal(0, 1, n) > 0).astype(int)
    return X, y


def test_accuracy_interval_is_sane():
    lo, hi = accuracy_interval(0.80, 176)
    assert 0.72 < lo < 0.80 < hi < 0.87
    lo2, hi2 = accuracy_interval(0.80, 5000)
    assert (hi2 - lo2) < (hi - lo)  # more rows, tighter
    assert accuracy_interval(0.5, 0) == (0.0, 1.0)


def test_stability_check_reports_spread():
    X, y = _xy()
    pipe = Pipeline([("s", StandardScaler()), ("m", RandomForestClassifier(n_estimators=40, random_state=0))]).fit(X[:240], y[:240])
    r = stability_check(pipe, X[:240], y[:240], X[240:], y[240:], TaskType.CLASSIFICATION, n_splits=4)
    assert len(r.scores) == 4 and 0.5 < r.mean <= 1.0 and r.std >= 0 and r.metric == "accuracy"


def test_tuning_verification_rejects_noise_and_returns_tuple():
    X, y = _xy()
    pipe = Pipeline([("s", StandardScaler()), ("model", RandomForestClassifier(n_estimators=40, random_state=0))])
    scorer = make_scorer(accuracy_score)
    out = _verify_against_defaults(pipe, {"model__min_samples_leaf": 140}, X, y, TaskType.CLASSIFICATION, scorer, 42)
    base, tuned, std, accept = out
    assert tuned < base and accept is False  # a crippled setting must not be accepted
    out2 = _verify_against_defaults(pipe, {"model__max_depth": None}, X, y, TaskType.CLASSIFICATION, scorer, 42)
    assert out2[3] is False  # identical to defaults: no gain, so not accepted
    assert _verify_against_defaults(pipe, {"nope": 1}, X, y, TaskType.CLASSIFICATION, scorer, 42) is None


def test_rejected_tuning_keeps_defaults_in_result():
    r = TuningResult("balanced", "XGBoost", base_mean=0.8, tuned_mean=0.8, tuned_std=0.01, best_params={},
                     verified=True, accepted=False)
    assert r.improvement == 0 and MIN_TUNING_GAIN > 0
