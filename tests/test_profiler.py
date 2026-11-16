import pandas as pd

from datapilot.profiler import DatasetProfiler


def test_profiler_basic():
    df = pd.DataFrame({
        "id": range(100),
        "age": [25, 30, None] * 33 + [25],
        "city": ["A", "B", "C"] * 33 + ["A"],
        "constant_col": [1] * 100,
    })

    profiler = DatasetProfiler()
    profile = profiler.profile(df)

    assert profile.n_rows == 100
    assert profile.n_cols == 4
    assert profile.columns["id"].is_likely_identifier
    assert profile.columns["constant_col"].is_constant
    assert profile.columns["age"].n_missing > 0
