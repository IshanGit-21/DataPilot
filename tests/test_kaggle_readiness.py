"""Cleaning, scale, submission, metric override and clustering."""
import numpy as np
import pandas as pd
import pytest

from datapilot.cleaning import (
    RawCleaner,
    category_key,
    clean_class_target,
    looks_numeric_text,
    parse_numeric_series,
)
from datapilot.clustering import run_clustering
from datapilot.experiment import metrics_for
from datapilot.planner import train_test_split_data
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import TaskType
from datapilot.scale import subsample_for_search
from datapilot.submission import make_submission


def test_category_key_merges_variants():
    keys = {category_key(v) for v in ["Home & Garden", "home and garden", "HOME_AND_GARDEN", " home  &  garden "]}
    assert len(keys) == 1
    assert category_key("U.S.A") == category_key("USA")


def test_parse_numeric_series_currency_and_missing():
    out = parse_numeric_series(pd.Series(["$1,234.50", "£82.32", "12%", "(5)", None, "abc"]))
    assert out.iloc[0] == 1234.5 and out.iloc[1] == 82.32 and out.iloc[2] == 12 and out.iloc[3] == -5
    assert out.iloc[4:].isna().all()


def test_looks_numeric_text():
    assert looks_numeric_text(pd.Series([f"${i}.5" for i in range(50)]))
    assert not looks_numeric_text(pd.Series(["a", "b", "c"] * 20))
    assert not looks_numeric_text(pd.Series(range(50)))


def test_raw_cleaner_is_stateless_and_handles_both_kinds():
    df = pd.DataFrame({"c": ["Home & Garden", "home and garden", None], "p": ["$1", "2", "x"]})
    rc = RawCleaner(columns=["c"], numeric_columns=["p"]).fit(df)
    out = rc.transform(df)
    assert out["c"].iloc[0] == out["c"].iloc[1] and pd.isna(out["c"].iloc[2])
    assert out["p"].iloc[0] == 1.0 and np.isnan(out["p"].iloc[2])
    assert df["c"].iloc[0] == "Home & Garden"  # input untouched


def test_clean_class_target_boolean_variants():
    y, note = clean_class_target(pd.Series(["TRUE", "True", "1", "no", "N", "False", "0", "yes"]))
    assert set(y.unique()) == {0, 1} and note
    y3, n3 = clean_class_target(pd.Series(["yes", "no", "yes"]))
    assert n3 == "" and list(y3) == ["yes", "no", "yes"]
    y2, note2 = clean_class_target(pd.Series(["cat", "dog", "cat"]))
    assert note2 == "" and list(y2) == ["cat", "dog", "cat"]


def test_split_drops_duplicates_before_splitting():
    base = pd.DataFrame({"a": range(60), "t": [0, 1] * 30})
    df = pd.concat([base, base.iloc[:20]], ignore_index=True)
    sp = train_test_split_data(df, "t", TaskType.CLASSIFICATION)
    assert sp.n_dropped_duplicates == 20
    assert len(sp.X_train) + len(sp.X_test) == 60
    keep = train_test_split_data(df, "t", TaskType.CLASSIFICATION, drop_duplicates=False)
    assert keep.n_dropped_duplicates == 0


def test_profiler_flags_variants_and_numeric_text():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "cat": rng.choice(["Home & Garden", "home and garden", "Toys"], n),
        "price": [f"${x:.2f}" for x in rng.uniform(1, 100, n)],
        "y": rng.integers(0, 2, n),
    })
    p = DatasetProfiler().profile(df, target="y")
    assert p.columns["cat"].has_spelling_variants and p.columns["cat"].n_unique_clean == 2
    assert p.columns["price"].is_numeric_text and p.columns["price"].is_numeric


def test_subsample_for_search():
    X = pd.DataFrame({"a": range(1000)})
    y = pd.Series([0] * 900 + [1] * 100)
    Xs, ys, note = subsample_for_search(X, y, TaskType.CLASSIFICATION, max_rows=200)
    assert len(Xs) == 200 and ys.mean() == pytest.approx(0.1, abs=0.02) and note
    Xs2, _, note2 = subsample_for_search(X, y, TaskType.CLASSIFICATION, max_rows=5000)
    assert len(Xs2) == 1000 and note2 == ""


def test_metric_override_validation():
    _, p = metrics_for(TaskType.CLASSIFICATION, "roc_auc")
    assert p == "roc_auc"
    with pytest.raises(ValueError):
        metrics_for(TaskType.REGRESSION, "accuracy")


def test_make_submission_probabilities_and_labels():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=100)}); y = (X.a > 0).astype(int)
    m = LogisticRegression().fit(X, y)
    test = pd.DataFrame({"id": [10, 11, 12], "a": [-2.0, 0.0, 2.0]})
    sample_p = pd.DataFrame({"id": [10, 11, 12], "t": [0.44, 0.44, 0.44]})
    sub, note = make_submission(m, test, sample_p, ["a"])
    assert list(sub.columns) == ["id", "t"] and sub.t.between(0, 1).all() and sub.t.iloc[0] < sub.t.iloc[2]
    le = LabelEncoder().fit(["no", "yes"])
    sample_l = pd.DataFrame({"id": [10, 11, 12], "t": ["no", "no", "no"]})
    sub2, _ = make_submission(m, test, sample_l, ["a"], le)
    assert set(sub2.t) <= {"no", "yes"}
    with pytest.raises(ValueError):
        make_submission(m, test.drop(columns="a"), sample_p, ["a"])


def test_clustering_recovers_obvious_groups():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "x": np.r_[rng.normal(0, .3, 60), rng.normal(8, .3, 60)],
        "y": np.r_[rng.normal(0, .3, 60), rng.normal(8, .3, 60)],
        "row_id": range(120),
    })
    p = DatasetProfiler().profile(df)
    r = run_clustering(df, p)
    assert r.k == 2 and r.silhouette > 0.8 and "row_id" not in r.inputs
    assert r.sizes.sum() == 120 and len(r.pca) == 120
    with pytest.raises(ValueError):
        run_clustering(df.head(5), p)


def test_category_key_keeps_non_latin_labels_distinct():
    assert category_key("草") != category_key("水") and category_key("草") == "草"
    y, note = clean_class_target(pd.Series(["草", "水", "草", "炎"]))
    assert y.nunique() == 3 and note == ""


def test_xgboost_survives_a_fold_missing_a_class():
    from sklearn.base import clone

    from datapilot.model_zoo import get_models
    spec = [m for m in get_models(TaskType.CLASSIFICATION) if m.name == "XGBoost"]
    if not spec:
        pytest.skip("xgboost not installed")
    X = np.random.default_rng(0).random((50, 3))
    y = np.array([0] * 25 + [2] * 24 + [5])      # labels are not contiguous
    m = clone(spec[0].build()).fit(X, y)
    assert set(m.predict(X)) <= {0, 2, 5} and list(m.classes_) == [0, 2, 5]
    assert m.predict_proba(X).shape == (50, 3)


def test_xgboost_wrapper_works_inside_pipeline_and_cv():
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from datapilot.model_zoo import get_models
    spec = [m for m in get_models(TaskType.CLASSIFICATION) if m.name == "XGBoost"]
    if not spec:
        pytest.skip("xgboost not installed")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 3))); y = (X[0] > 0).astype(int)
    pipe = Pipeline([("s", StandardScaler()), ("m", spec[0].build())])
    assert cross_val_score(pipe, X, y, cv=3).mean() > 0.8
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (120,)


def test_rescue_pipeline_recovers_a_crashing_model():
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.dummy import DummyClassifier

    from datapilot.experiment import ExperimentEngine
    from datapilot.model_zoo import ModelSpec
    from datapilot.planner import PipelineOptions, PipelinePlanner
    from datapilot.reasoner import Priority, UserPreferences

    class FussyModel(BaseEstimator, ClassifierMixin):
        """Refuses the main pipelines' output (NaN-free but with many columns) -> use column count."""
        def fit(self, X, y):
            if X.shape[1] > 3:
                raise ValueError("too many columns for this model")
            self.inner_ = DummyClassifier(strategy="prior").fit(X, y)
            self.classes_ = self.inner_.classes_
            return self
        def predict(self, X):
            return self.inner_.predict(X)
        def predict_proba(self, X):
            return self.inner_.predict_proba(X)

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                       "c": rng.choice(list("pqrstuvwxy"), n), "y": rng.integers(0, 2, n)})
    prof = DatasetProfiler().profile(df, target="y")
    prefs = UserPreferences(target_column="y", task_type=TaskType.CLASSIFICATION, priority=Priority.BALANCED)
    planner = PipelinePlanner()
    cands = planner.plan(list(df.columns), prof, prefs, PipelineOptions())
    rescue = planner.plan_rescue(list(df.columns), prof)
    models = [ModelSpec("Baseline", lambda: DummyClassifier(), is_baseline=True),
              ModelSpec("Fussy", lambda: FussyModel())]
    eng = ExperimentEngine(TaskType.CLASSIFICATION, n_splits=3)
    entries = eng.run_leaderboard(cands, models, df.drop(columns="y"), df["y"], rescue=rescue)
    fussy = [e for e in entries if e.model_name == "Fussy"]
    assert any(e.ok and e.pipeline_name == "rescue" for e in fussy)
    assert "rescue" in [c.name for c in cands] and "recovered" in eng.rescue_note


def test_stacking_spec_resolves_and_fits():
    from datapilot.ensemble import resolve_model_spec
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(150, 3))); y = (X[0] + X[1] > 0).astype(int)
    spec = resolve_model_spec("Stacking: Logistic Regression + Random Forest + Decision Tree", TaskType.CLASSIFICATION)
    m = spec.build().fit(X, y)
    assert (m.predict(X) == y).mean() > 0.9
    yr = X[0] * 2 + rng.normal(size=150) * .1
    r = resolve_model_spec("Stacking: Linear Regression + Random Forest", TaskType.REGRESSION).build().fit(X, yr)
    assert r.predict(X).shape == (150,)


def test_catboost_in_zoo():
    from datapilot.model_zoo import get_models
    names = [m.name for m in get_models(TaskType.CLASSIFICATION)]
    pytest.importorskip("catboost")
    assert "CatBoost" in names and "CatBoost" in [m.name for m in get_models(TaskType.REGRESSION)]
