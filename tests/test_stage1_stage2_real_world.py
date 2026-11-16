import numpy as np
import pandas as pd

from datapilot.planner import PipelinePlanner, train_test_split_data
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import Priority, TaskType, UserPreferences


def make_realistic_df(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    cities = [f"city_{i}" for i in range(300)]  # high-cardinality categorical
    grades = ["A", "B", "C", "D"]                # low-cardinality categorical

    dates = pd.date_range("2015-01-01", periods=2000, freq="D")
    df = pd.DataFrame({
        "id": range(n),
        "price": rng.normal(250000, 60000, n).round(2),
        "sqft": rng.normal(1800, 400, n).round(0),
        "grade": rng.choice(grades, n),
        "city": rng.choice(cities, n),                      # high-cardinality
        "listed_date": rng.choice(dates.astype(str), n),    # date-as-string
        "sale_price": rng.normal(300000, 80000, n).round(2),
    })

    # inject missingness
    df.loc[df.sample(frac=0.08, random_state=1).index, "sqft"] = None
    # inject real outliers
    outlier_idx = df.sample(frac=0.02, random_state=2).index
    df.loc[outlier_idx, "price"] = df.loc[outlier_idx, "price"] * 8
    # correlated feature (price and sale_price should correlate)
    df["sale_price"] = df["price"] * 1.15 + rng.normal(0, 15000, n)
    return df


def test_stage1_detects_outliers_datetime_and_correlations():
    df = make_realistic_df()
    profiler = DatasetProfiler()
    profile = profiler.profile(df, target="sale_price")

    assert profile.columns["listed_date"].is_datetime, "date-as-string column should be detected"
    assert profile.columns["city"].is_high_cardinality, "300-category column should be high-cardinality"
    assert not profile.columns["grade"].is_high_cardinality
    assert profile.columns["price"].n_outliers_iqr is not None and profile.columns["price"].n_outliers_iqr > 0
    assert profile.columns["price"].skew is not None

    # price<->sale_price is a target relationship, not a feature<->feature
    # one — it must NOT appear in top_correlations (see the dedicated
    # regression test for this); this dataset has no other engineered
    # feature<->feature correlation, so top_correlations is expected empty here.
    pairs = {(p.col_a, p.col_b) for p in profile.top_correlations}
    assert ("price", "sale_price") not in pairs and ("sale_price", "price") not in pairs

    # target correlations should be populated (numeric target)
    assert len(profile.target_correlations) > 0
    top = profile.target_correlations[0]
    assert top.col_a == "price"  # strongest driver of sale_price by construction


def test_target_does_not_leak_into_feature_feature_correlations():
    """Regression test for a real bug: top_correlations (meant to be pure
    feature<->feature) used to include the target column in its input list,
    so target<->feature pairs leaked into that panel too, duplicating (and
    mislabeling) what target_correlations already reports separately.
    """
    df = make_realistic_df()
    profiler = DatasetProfiler()
    profile = profiler.profile(df, target="sale_price")

    for pair in profile.top_correlations:
        assert pair.col_a != "sale_price" and pair.col_b != "sale_price", (
            "the target must never appear in feature<->feature correlations "
            "— it belongs only in target_correlations"
        )
    # the price<->sale_price relationship must still be reachable, just in
    # the correct (target_correlations) bucket
    assert any(p.col_a == "price" for p in profile.target_correlations)


def test_continuous_numeric_feature_not_misflagged_as_identifier():
    """Regression test for a real bug: a continuous numeric column (price)
    can be ~100% unique purely because it's a precise measurement, not
    because it's an identifier. The old heuristic (unique_ratio >= 0.95)
    flagged it as an identifier and silently dropped it from every
    pipeline, starving every model of its strongest feature — every real
    model scored WORSE than a dummy baseline as a result. Only a sequential
    auto-increment integer, or a column whose name says id/uuid/index,
    should be flagged.
    """
    df = make_realistic_df()
    profiler = DatasetProfiler()
    profile = profiler.profile(df, target="sale_price")

    # "id" is a real 0..n-1 auto-increment column — SHOULD be flagged
    assert profile.columns["id"].is_likely_identifier

    # "price" is ~100% unique (continuous float) but is NOT an identifier —
    # and it's the strongest predictor of the target, so dropping it would
    # be catastrophic for every downstream model
    assert profile.columns["price"].unique_ratio > 0.95
    assert not profile.columns["price"].is_likely_identifier

    planner = PipelinePlanner()
    groups = planner.split_columns(list(df.columns), profile)
    assert "price" in groups.numeric, "price must survive into the feature set"
    assert "id" not in groups.numeric


def test_stage2_full_pipeline_real_world():
    df = make_realistic_df()
    profiler = DatasetProfiler()
    target = "sale_price"
    profile = profiler.profile(df, target=target)

    prefs = UserPreferences(target_column=target, task_type=TaskType.REGRESSION, priority=Priority.BALANCED)

    # Stage 2, step 1: real train/test split, before any fitting
    X_train, X_test, y_train, y_test = train_test_split_data(df, target, TaskType.REGRESSION)
    assert len(X_train) + len(X_test) == len(df)
    assert len(X_train) > len(X_test)  # 80/20 default

    planner = PipelinePlanner()
    groups = planner.split_columns(list(df.columns), profile)
    assert "city" in groups.categorical_high
    assert "grade" in groups.categorical_low
    assert "listed_date" in groups.datetime
    assert "id" not in (groups.numeric + groups.categorical_low + groups.categorical_high)  # identifier dropped

    candidates = planner.plan(list(df.columns), profile, prefs)
    assert {c.name for c in candidates} == {"conservative", "balanced", "aggressive"}

    for cand in candidates:
        # fit ONLY on training data — this is the leakage-safety contract
        cand.pipeline.fit(X_train, y_train)
        Xt_train = cand.pipeline.transform(X_train)
        Xt_test = cand.pipeline.transform(X_test)
        assert Xt_train.shape[0] == len(X_train)
        assert Xt_test.shape[0] == len(X_test)
        assert Xt_train.shape[1] == Xt_test.shape[1]  # same feature space both sides

    # frequency encoding must NOT have exploded the high-cardinality column
    # into hundreds of one-hot columns — conservative candidate should have
    # far fewer output features than (n numeric + n one-hot-of-300 + ...)
    conservative = next(c for c in candidates if c.name == "conservative")
    Xt = conservative.pipeline.transform(X_train)
    assert Xt.shape[1] < 50, f"expected compact output, got {Xt.shape[1]} columns (frequency encoding should prevent blowup)"
