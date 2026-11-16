"""Plain-language summary and the one-example predictor."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from datapilot.evaluation import ClassificationEvaluation, FinalEvaluationResult, RegressionEvaluation
from datapilot.explain import example_fields, plain_summary, predict_example, prep_label, typical_row
from datapilot.interpretation import FeatureImportance
from datapilot.reasoner import TaskType


def _reg_result():
    reg = RegressionEvaluation(r2=0.66, rmse=1.86, mae=1.25, mape=0.127, residual_mean=0, residual_std=1.8)
    return FinalEvaluationResult(pipeline_name="p", model_name="m", task_type=TaskType.REGRESSION, n_test=18123, regression=reg)


def test_regression_summary_uses_measured_numbers():
    text = " ".join(plain_summary(_reg_result(), "Rings"))
    assert "18,123" in text and "1.25" in text and "13%" in text and "66%" in text


def test_classification_summary_counts_real_cases():
    conf = [[70, 10], [20, 100]]            # rows actual, columns predicted
    clf = ClassificationEvaluation(accuracy=170 / 200, precision=0.8, recall=0.8, f1=0.8, f2=0.8, roc_auc=None,
                                   confusion=conf, labels=["stay", "leave"])
    res = FinalEvaluationResult(pipeline_name="p", model_name="m", task_type=TaskType.CLASSIFICATION, n_test=200, classification=clf)
    text = " ".join(plain_summary(res, "Exited"))
    assert "85%" in text                      # accuracy
    assert "60%" in text                      # always guessing the most common answer: 120/200
    assert "100 of the 120 real <b>leave</b>" in text and "right 91% of the time" in text


def test_example_fields_follow_importance_and_prediction_runs():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"size": rng.normal(10, 2, 300), "kind": rng.choice(["a", "b"], 300), "noise": rng.normal(size=300)})
    y = X["size"] * 3 + (X["kind"] == "a") * 5 + rng.normal(0, 0.1, 300)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    pre = ColumnTransformer([("num", "passthrough", ["size", "noise"]), ("cat", OneHotEncoder(), ["kind"])])
    pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(40, random_state=0))]).fit(X, y)
    feats = [FeatureImportance("num__size", 0.9), FeatureImportance("cat__kind_a", 0.08), FeatureImportance("num__noise", 0.02)]
    assert example_fields(X, feats, 2) == ["size", "kind"]
    lo = predict_example(pipe, X, {"size": 6.0, "kind": "b"})["prediction"]
    hi = predict_example(pipe, X, {"size": 14.0, "kind": "b"})["prediction"]
    assert hi > lo + 10                        # the answer moves with the input
    assert typical_row(X)["kind"] in ("a", "b")


def test_classification_example_gives_chances_and_original_labels():
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": rng.normal(size=200)})
    y_text = np.where(X["x"] > 0, "yes", "no")
    enc = LabelEncoder().fit(y_text)
    pipe = Pipeline([("model", RandomForestClassifier(30, random_state=0))]).fit(X, enc.transform(y_text))
    out = predict_example(pipe, X, {"x": 2.5}, enc)
    assert out["prediction"] == "yes" and out["chances"]["yes"] > 0.8


def test_prep_labels_are_plain():
    assert prep_label("aggressive") == "Extra feature building" and prep_label("custom") == "custom"


def test_readable_feature_names():
    from datapilot.explain import readable_feature
    assert readable_feature("num__age") == "age"
    assert readable_feature("cat_low__sex_M") == "sex = M"
    assert readable_feature("plain") == "plain"


def test_accuracy_percent_and_duration():
    from types import SimpleNamespace as NS
    from datapilot.explain import accuracy_percent, fmt_duration, total_seconds
    reg = NS(classification=None, regression=NS(mape=0.127))
    clf = NS(classification=NS(accuracy=0.86), regression=None)
    assert abs(accuracy_percent(reg) - 0.873) < 1e-9 and accuracy_percent(clf) == 0.86
    assert accuracy_percent(NS(classification=None, regression=NS(mape=3.0))) == 0.0
    assert fmt_duration(0.3) == "under 1 s" and fmt_duration(41) == "41 s" and fmt_duration(221) == "3 min 41 s"
    assert total_seconds({"seconds": 1.5}, None, {"seconds": 2.0}, {}) == 3.5


def test_run_summary_bullets_and_timing():
    import numpy as np, pandas as pd
    from datapilot.experiment import display_score
    from datapilot.explain import run_summary
    from datapilot.reasoner import TaskType
    from datapilot.runner import RunConfig, run_all
    rng = np.random.default_rng(0); n = 300
    df = pd.DataFrame({"id": range(n), "a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["y"] = 50 + 10 * df.a + rng.normal(0, 2, n)
    st, lb, tun, fin = run_all(df, RunConfig("y", TaskType.REGRESSION, effort="Quick"))
    assert all(d["seconds"] > 0 for d in (st, lb, tun, fin))
    text = " ".join(run_summary(st, lb, tun, fin, "y", display_score))
    for word in ("Read the data", "Set rows aside", "Compared models", "Accuracy", "Time."):
        assert word in text
    assert "id" in text   # the identifier column is named as left out


def test_live_screen_pieces_are_safe_and_complete():
    from datapilot.live import (RUN_NODES, clock_html, live_columns_visual, live_features_visual, live_models_visual,
                                run_live_html)
    vis = live_columns_visual([("<script>x</script>", "number"), ("churn", "target")])
    html = run_live_html(2, ["a", "b", "c", "", ""], 125, live_models_visual([("XGBoost", 2)], 7, 36, 3))
    assert "<script>" not in vis and "&lt;script&gt;" in vis
    assert all(n in html for n in RUN_NODES) and "XGBoost" in html and "7 of 36" in html
    assert 'style="--e:125"' in clock_html(125.9) or 'style="--e:125"' in clock_html(125)
    assert "working" in live_features_visual([("Simple cleaning", None)], 6)


def test_summary_tab_html_has_numbered_steps_and_time_bar():
    from datapilot.explain import summary_cards, summary_html
    cards = summary_cards(["<b>Read the data.</b> 10 rows.", "<b>Time.</b> 5 s"])
    out = summary_html(cards, [("Compare models", 4.0), ("Final test", 1.0)])
    assert "Read the data" in out and "class='num'>1<" in out and "Where the 5 s went" in out
