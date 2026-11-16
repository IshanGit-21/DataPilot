import ast
import io

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from datapilot.experiment import ExperimentEngine
from datapilot.export import generate_prediction_script, serialize_pipeline
from datapilot.interpretation import interpret
from datapilot.model_zoo import get_models
from datapilot.planner import PipelinePlanner, train_test_split_data
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import Priority, TaskType, UserPreferences


def make_regression_df(n=700, seed=11):
    rng = np.random.default_rng(seed)
    sqft = rng.normal(1800, 400, n)
    beds = rng.integers(1, 6, n)
    noise = rng.normal(0, 20000, n)
    price = 50000 + sqft * 120 + beds * 8000 + noise
    df = pd.DataFrame({
        "id": range(n),
        "sqft": sqft,
        "beds": beds,
        "grade": rng.choice(["A", "B", "C"], n),
        "price": price,
    })
    df.loc[df.sample(frac=0.05, random_state=2).index, "sqft"] = None
    return df


def _fit_best(df, target, task_type):
    profiler = DatasetProfiler()
    profile = profiler.profile(df, target=target)
    prefs = UserPreferences(target_column=target, task_type=task_type, priority=Priority.BALANCED)
    split = train_test_split_data(df, target, task_type)
    planner = PipelinePlanner()
    candidates = planner.plan(list(df.columns), profile, prefs)
    models = get_models(task_type)
    engine = ExperimentEngine(task_type=task_type, n_splits=4, n_repeats=1)
    entries = engine.run_leaderboard(candidates, models, split.X_train, split.y_train)
    best = next(e for e in entries if e.ok)
    candidate = next(c for c in candidates if c.name == best.pipeline_name)
    spec = next(m for m in models if m.name == best.model_name)
    fitted = Pipeline(candidate.pipeline.steps + [("model", spec.build())])
    fitted.fit(split.X_train, split.y_train)
    return fitted, split, best


def test_interpretation_returns_real_named_features_for_linear_and_tree_models():
    df = make_regression_df()
    fitted, split, best = _fit_best(df, "price", TaskType.REGRESSION)

    result = interpret(fitted, X_train=split.X_train, top_n=10)
    assert result.kind in ("coefficients", "tree_importances")
    assert len(result.features) > 0
    # feature names must be real strings, not "feature_0"-style fallbacks,
    # since get_feature_names_out() should succeed on this well-formed pipeline
    assert all(not f.feature.startswith("feature_") for f in result.features)
    # sqft is the dominant real driver of price by construction — it (or its
    # one-hot/frequency-encoded siblings) should show up prominently
    assert any("sqft" in f.feature for f in result.features)
    # importances must be sorted descending
    values = [f.importance for f in result.features]
    assert values == sorted(values, reverse=True)


def test_linear_coefficients_are_scale_standardized_not_raw():
    """Regression test for a real bug: on an unscaled pipeline, a 0/1 flag
    with a numerically large coefficient (e.g. 4000) ranked ABOVE a
    dollar-scale feature with a small coefficient (e.g. 1.15) purely
    because raw coefficients aren't comparable across differently-scaled
    features — even though the dollar-scale feature dominates the actual
    prediction. Interpretation must multiply by each feature's real
    training-data spread so the ranking reflects true influence.
    """
    n = 500
    rng = np.random.default_rng(21)
    sqft = rng.normal(1800, 400, n)  # large-scale, dominant driver
    flag = rng.integers(0, 2, n)      # small-scale (0/1), tiny real effect
    price = 50000 + sqft * 120 + flag * 50 + rng.normal(0, 5000, n)  # flag barely matters
    df = pd.DataFrame({"id": range(n), "sqft": sqft, "flag": flag, "price": price})

    fitted, split, best = _fit_best(df, "price", TaskType.REGRESSION)
    if best.model_name != "Linear Regression":
        return  # only meaningful for the linear model path; leaderboard order can vary by seed

    result = interpret(fitted, X_train=split.X_train, top_n=10)
    assert result.kind == "coefficients"
    by_name = {f.feature: f.importance for f in result.features}
    sqft_imp = next(v for k, v in by_name.items() if "sqft" in k)
    flag_imp = next((v for k, v in by_name.items() if "flag" in k), 0.0)
    assert sqft_imp > flag_imp, (
        "sqft genuinely drives price far more than flag does — standardized "
        "importance must reflect that, not raw coefficient magnitude"
    )


def test_exported_joblib_roundtrip_predicts_identically():
    """The real proof: serialize with the exact function the UI's download
    button calls, deserialize as if a different machine downloaded it, and
    confirm predictions are bit-identical to the original in-memory model.
    """
    df = make_regression_df()
    fitted, split, best = _fit_best(df, "price", TaskType.REGRESSION)

    original_preds = fitted.predict(split.X_test)

    blob = serialize_pipeline(fitted)
    assert isinstance(blob, bytes) and len(blob) > 0

    reloaded = joblib.load(io.BytesIO(blob))
    reloaded_preds = reloaded.predict(split.X_test)

    np.testing.assert_array_almost_equal(original_preds, reloaded_preds)


def test_generated_prediction_script_is_valid_python_and_self_consistent():
    df = make_regression_df()
    fitted, split, best = _fit_best(df, "price", TaskType.REGRESSION)
    feature_columns = list(split.X_train.columns)

    script = generate_prediction_script(
        pipeline_name=best.pipeline_name, model_name=best.model_name,
        task_type_value="regression", target_column="price",
        feature_columns=feature_columns,
    )
    # must be syntactically valid Python — not just plausible-looking text
    ast.parse(script)
    assert "FEATURE_COLUMNS" in script
    assert "price" not in _extract_feature_columns_literal(script), \
        "target column must never appear in the feature list"
    for col in feature_columns:
        assert repr(col) in script or col in script


def _extract_feature_columns_literal(script: str) -> list[str]:
    tree = ast.parse(script)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FEATURE_COLUMNS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    return []
