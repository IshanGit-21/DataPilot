import numpy as np
import pandas as pd

from datapilot import TaskType, assess_results, detect_leakage_suspects
from datapilot.benchmark import benchmark_to_markdown, run_case
from datapilot.experiment import LeaderboardEntry


def _entry(name, score, base=False, std=0.01, err=None):
    return LeaderboardEntry("balanced", name, base, "f1", score, std, error=err)


def test_leakage_detects_target_copy_and_class_mapped_category_but_not_honest_features():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, n)
    X = pd.DataFrame({
        "honest": rng.normal(size=n),
        "noisy_signal": y + rng.normal(0, 2, n),
        "leak_numeric": y * 3.0 + 1,
        "leak_category": np.where(y == 1, "paid", "unpaid"),
        "weather": rng.choice(list("abc"), n),
    })
    found = {s.column for s in detect_leakage_suspects(X, y, TaskType.CLASSIFICATION)}
    assert {"leak_numeric", "leak_category"} <= found
    assert not ({"honest", "noisy_signal", "weather"} & found)


def test_leakage_regression_numeric_copy():
    rng = np.random.default_rng(1)
    n = 300
    t = rng.normal(size=n)
    X = pd.DataFrame({"a": rng.normal(size=n), "price_copy": t * 100 + 5})
    found = {s.column for s in detect_leakage_suspects(X, t, TaskType.REGRESSION)}
    assert found == {"price_copy"}


def test_assess_results_flags_weak_margin_failures_and_small_test():
    w = assess_results([_entry("Baseline", 0.50, base=True), _entry("RF", 0.51), _entry("X", 0, err="boom")])
    assert any("barely beats" in m for m in w) and any("failed" in m for m in w)
    assert assess_results([_entry("Baseline", 0.50, base=True), _entry("RF", 0.90)]) == []
    assert assess_results([_entry("X", 0, err="boom")]) == ["No model trained successfully."]


def test_run_case_records_real_results_and_failures():
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = np.where(df.a + 0.3 * rng.normal(size=n) > 0, "pos", "neg")
    row = run_case("toy", df, "y")
    assert row.error is None and row.task == "classification"
    assert row.target_suggested_first is True
    assert row.heldout_primary > 0.7 and row.uplift > 0.1
    md = benchmark_to_markdown([row])
    assert "toy" in md and "classification" in md
    bad = run_case("bad", df, "missing_column")
    assert bad.error and "FAILED" in benchmark_to_markdown([bad])


# ------------------------------------------------ real-world edge cases ---
def test_read_table_detects_headerless_csv_and_leaves_normal_alone(tmp_path):
    from datapilot import read_table
    (tmp_path / "a.csv").write_text("1,14.23,1.71\n1,13.2,1.78\n2,12.3,.9\n")
    df, hl = read_table(tmp_path / "a.csv")
    assert hl and list(df.columns) == ["col_0", "col_1", "col_2"] and len(df) == 3
    (tmp_path / "b.csv").write_text("a,b\n1,2\n3,4\n")
    df, hl = read_table(tmp_path / "b.csv")
    assert not hl and list(df.columns) == ["a", "b"]


def test_identifier_names_in_every_common_spelling():
    from datapilot import DatasetProfiler
    rng = np.random.default_rng(0)
    n = 300
    base = pd.DataFrame({"x": rng.normal(size=n)})
    for name in ["User ID", "UserID", "CustomerId", "customer-id", "user_id"]:
        df = base.copy(); df[name] = rng.permutation(np.arange(n) * 7 + 15000000)  # non-sequential
        assert DatasetProfiler().profile(df).columns[name].is_likely_identifier, name
    df = base.copy(); df["Paid"] = rng.permutation(np.arange(n) * 7.5)  # 'paid' must not match 'id'
    assert not DatasetProfiler().profile(df).columns["Paid"].is_likely_identifier


def test_headerless_class_first_column_is_suggested_as_target():
    from datapilot import DatasetProfiler, suggest_targets
    rng = np.random.default_rng(1)
    n = 150
    df = pd.DataFrame({"col_0": rng.integers(1, 4, n)})
    for i in range(1, 6):
        df[f"col_{i}"] = rng.normal(size=n) + df.col_0
    df["col_6"] = rng.normal(500, 100, n)  # continuous last column must not win
    assert suggest_targets(df, DatasetProfiler().profile(df))[0].column == "col_0"


def test_sensitive_input_columns_by_name():
    from datapilot import sensitive_input_columns
    got = sensitive_input_columns(["Sex", "Gender", "ForeignWorker", "Age", "LoanAmount", "marital_status", "Sexton"])
    assert set(got) == {"Sex", "Gender", "ForeignWorker", "marital_status"}
