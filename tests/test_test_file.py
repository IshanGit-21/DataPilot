"""A test file added at the start: with answers it is the final test; without answers (Kaggle style)
every training row is used to learn and the score comes from cross-checking."""
import numpy as np
import pandas as pd
import pytest

from datapilot import boosting
from datapilot.planner import file_has_answers as has_answers, train_test_split_data
from datapilot.reasoner import TaskType
from datapilot.runner import RunConfig, build_state, finalize, race, tune


def _frame(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], size=n)
    y = np.where(x1 + 0.5 * x2 + (cat == "a") + rng.normal(scale=0.7, size=n) > 0.5, "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "target": y})


def test_labelled_test_file_becomes_the_test_and_nothing_is_cut_from_training():
    train, test = _frame(500, 0), _frame(120, 1)
    split = train_test_split_data(train, "target", TaskType.CLASSIFICATION, test_df=test)
    assert split.score_mode == "external"
    assert len(split.X_train) == 500 and len(split.X_test) == 120
    assert set(split.y_test.unique()) <= {0, 1}
    assert "your test file" in split.test_note.lower()


def test_test_file_without_answers_keeps_every_row_for_training():
    train, test = _frame(500, 0), _frame(120, 1).drop(columns="target")
    assert not has_answers(test, "target")
    split = train_test_split_data(train, "target", TaskType.CLASSIFICATION, test_df=test)
    assert split.score_mode == "cv"
    assert len(split.X_train) == 500           # nothing hidden
    assert len(split.X_test) == len(split.y_test) > 0   # the rows that get cross-checked


def test_test_file_with_missing_columns_is_refused_plainly():
    train, test = _frame(200, 0), _frame(50, 1).drop(columns=["x2", "target"])
    with pytest.raises(ValueError, match="missing columns"):
        train_test_split_data(train, "target", TaskType.CLASSIFICATION, test_df=test)


def test_labels_unseen_in_training_are_left_out_not_crashed_on():
    train, test = _frame(300, 0), _frame(60, 1)
    test.loc[:4, "target"] = "maybe"
    split = train_test_split_data(train, "target", TaskType.CLASSIFICATION, test_df=test)
    assert len(split.X_test) == 55 and "never seen" in split.test_note


@pytest.mark.parametrize("labelled", [True, False])
def test_run_with_a_test_file_end_to_end(labelled):
    train, test = _frame(400, 0), _frame(100, 1)
    if not labelled:
        test = test.drop(columns="target")
    cfg = RunConfig("target", TaskType.CLASSIFICATION, effort="Quick", test_df=test)
    st = build_state(train, cfg, "t")
    lb = race(st, cfg, "t")
    fin = finalize(st, lb, tune(st, lb, cfg, "t"), "t")
    res = fin["result"]
    assert res.ok
    assert res.scored_by == ("external" if labelled else "cv")
    assert res.classification.accuracy > 0.6
    if not labelled:   # the model learned from all rows and can predict the test file directly
        assert len(fin["final_pipeline"].predict(test[fin["feature_columns"]])) == len(test)


def test_booster_wrapper_is_plain_on_small_tables_and_stops_early_on_big_ones(monkeypatch):
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(800, 4)), columns=list("abcd"))
    y = (X.a + X.b * 0.5 + rng.normal(scale=0.5, size=800) > 0).astype(int)
    plain = boosting.wrap_boosters(LGBMClassifier(n_estimators=40, verbose=-1)).fit(X, y)
    assert plain.n_trees_ is None and plain.predict_proba(X).shape == (800, 2)
    monkeypatch.setattr(boosting, "MIN_ROWS", 500)
    stopped = boosting.wrap_boosters(LGBMClassifier(n_estimators=40, verbose=-1)).fit(X, y)
    assert 0 < stopped.n_trees_ < boosting.MAX_TREES
    assert (stopped.predict(X) == y).mean() > 0.7
    stopped.set_params(learning_rate=0.2)            # settings for the model inside are passed on
    assert stopped.estimator.get_params()["learning_rate"] == 0.2
