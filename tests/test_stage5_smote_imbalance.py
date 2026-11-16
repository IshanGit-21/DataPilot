import numpy as np
import pandas as pd

from datapilot.experiment import ExperimentEngine
from datapilot.imbalance import class_balance_ratio, smote_is_viable
from datapilot.model_zoo import get_models
from datapilot.planner import PipelinePlanner, train_test_split_data
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import Priority, TaskType, UserPreferences
from datapilot.tuning import tune_top_entries


def make_imbalanced_df(n=900, minority_frac=0.08, seed=5):
    """A genuinely imbalanced binary classification dataset — ~8% positive
    class, with real signal (income drives the outcome) so a model can
    plausibly learn something, not pure noise."""
    rng = np.random.default_rng(seed)
    income = rng.normal(50000, 15000, n)
    age = rng.integers(18, 75, n)
    segment = rng.choice(["retail", "sme", "enterprise"], n)

    n_pos = int(n * minority_frac)
    fraud = np.zeros(n, dtype=int)
    pos_idx = rng.choice(n, size=n_pos, replace=False)
    fraud[pos_idx] = 1
    # give the positive class a real (if noisy) income signature
    income = income.copy()
    income[fraud == 1] += rng.normal(20000, 8000, n_pos)

    df = pd.DataFrame({
        "id": range(n),
        "income": income,
        "age": age,
        "segment": segment,
        "fraud": fraud,
    })
    df.loc[df.sample(frac=0.03, random_state=6).index, "income"] = None
    return df


def make_balanced_df(n=600, seed=9):
    """A genuinely balanced binary classification dataset — SMOTE should
    NOT trigger here."""
    rng = np.random.default_rng(seed)
    income = rng.normal(50000, 15000, n)
    label = (income > np.median(income)).astype(int)
    df = pd.DataFrame({
        "id": range(n),
        "income": income,
        "age": rng.integers(18, 75, n),
        "label": label,
    })
    return df


def _leaderboard_for(df, target):
    profiler = DatasetProfiler()
    profile = profiler.profile(df, target=target)
    prefs = UserPreferences(target_column=target, task_type=TaskType.CLASSIFICATION, priority=Priority.BALANCED)
    split = train_test_split_data(df, target, TaskType.CLASSIFICATION)
    planner = PipelinePlanner()
    candidates = planner.plan(list(df.columns), profile, prefs)
    models = get_models(TaskType.CLASSIFICATION)
    engine = ExperimentEngine(task_type=TaskType.CLASSIFICATION, n_splits=4, n_repeats=1)
    entries = engine.run_leaderboard(candidates, models, split.X_train, split.y_train)
    return engine, entries, candidates, models, split


def test_class_balance_ratio_and_viability_detection():
    imbalanced_df = make_imbalanced_df()
    balanced_df = make_balanced_df()

    ratio_imb = class_balance_ratio(imbalanced_df["fraud"])
    ratio_bal = class_balance_ratio(balanced_df["label"])
    assert ratio_imb < 0.15  # ~8% minority by construction
    assert ratio_bal > 0.4   # roughly 50/50 by construction

    viable_imb, reason_imb = smote_is_viable(imbalanced_df["fraud"])
    viable_bal, reason_bal = smote_is_viable(balanced_df["label"])
    assert viable_imb is True
    assert viable_bal is False
    assert "balanced" in reason_bal.lower() or "%" in reason_bal


def test_smote_entries_appear_only_for_genuinely_imbalanced_data():
    engine, entries, candidates, models, split = _leaderboard_for(make_imbalanced_df(), "fraud")
    smote_entries = [e for e in entries if e.uses_smote]
    assert len(smote_entries) > 0, "expected real SMOTE entries on genuinely imbalanced data"
    assert all(e.pipeline_name == "balanced" for e in smote_entries), \
        "SMOTE is scoped to the balanced candidate only, by design"
    assert all("+ SMOTE" in e.model_name for e in smote_entries)
    assert all(not e.is_baseline for e in smote_entries), "baseline should never get a SMOTE variant"
    assert engine.smote_note and "%" in engine.smote_note

    # on balanced data, SMOTE must NOT run
    engine2, entries2, *_ = _leaderboard_for(make_balanced_df(), "label")
    assert not any(e.uses_smote for e in entries2)
    assert "balanced" in engine2.smote_note.lower()


def test_smote_pipeline_never_resamples_at_predict_time():
    """The real safety proof: predicting on the held-out test split must
    return exactly as many predictions as input rows — if SMOTE were
    (incorrectly) applied at predict time too, the row count would change
    or the call would behave unpredictably.
    """
    engine, entries, candidates, models, split = _leaderboard_for(make_imbalanced_df(), "fraud")
    smote_entry = next((e for e in entries if e.uses_smote and e.ok), None)
    assert smote_entry is not None, f"no successful SMOTE entry: {[e.error for e in entries if e.uses_smote]}"

    from datapilot.imbalance import build_smote_pipeline
    candidate = next(c for c in candidates if c.name == smote_entry.pipeline_name)
    base_name = smote_entry.model_name.removesuffix(" + SMOTE")
    spec = next(m for m in models if m.name == base_name)

    pipeline = build_smote_pipeline(candidate.pipeline.steps, spec.build())
    pipeline.fit(split.X_train, split.y_train)
    preds = pipeline.predict(split.X_test)
    assert len(preds) == len(split.X_test), "predict() must return one prediction per input row — no resampling at inference"


def test_smote_is_skipped_honestly_when_minority_class_too_small():
    """A minority class too small for SMOTE's nearest-neighbor synthesis
    must be skipped with a real, specific reason — not silently attempted
    and crashed, and not silently faked as working."""
    df = make_imbalanced_df(n=200, minority_frac=0.02)  # ~4 positive rows
    viable, reason = smote_is_viable(df["fraud"])
    assert viable is False
    assert "few" in reason.lower() or "neighbor" in reason.lower()


def test_tuning_reconstructs_smote_pipeline_correctly():
    engine, entries, candidates, models, split = _leaderboard_for(make_imbalanced_df(), "fraud")
    smote_entries = [e for e in entries if e.uses_smote and e.ok]
    assert smote_entries, "need at least one successful SMOTE entry to test tuning against"

    # Pass ONLY the SMOTE entry — tune_top_entries dedupes/ranks by raw
    # score across whatever list it's given, so mixing in non-SMOTE entries
    # would let a higher-scoring non-SMOTE entry win the ranking instead of
    # actually exercising the SMOTE reconstruction path this test targets.
    results = tune_top_entries(
        smote_entries[:1], candidates, models, split.X_train, split.y_train,
        TaskType.CLASSIFICATION, top_k=1, n_iter=4, cv_splits=3,
    )
    assert len(results) == 1
    result = results[0]
    assert result.uses_smote is True
    assert "+ SMOTE" in result.model_name
    # tuning a SMOTE entry must succeed (or fail with a real error) — it
    # must never silently fall back to tuning a non-resampled pipeline
    if result.tunable:
        assert result.ok, f"SMOTE tuning failed: {result.error}"
        assert np.isfinite(result.tuned_mean)


def test_macro_f1_correctly_penalizes_baseline_on_imbalanced_data():
    """Regression test for a real bug: 'weighted' F1/precision/recall/F2
    averaging is dominated by the majority class's support, so on genuinely
    imbalanced data a baseline that catches 0% of the minority class could
    still score ~0.88 weighted-F1 — deceptively close to real models,
    masking whether SMOTE/class_weight actually helped. Macro averaging
    must correctly show the baseline scoring far worse (near chance-level
    for the class that matters), since it weights both classes equally
    regardless of size.
    """
    engine, entries, candidates, models, split = _leaderboard_for(make_imbalanced_df(), "fraud")
    ok_entries = [e for e in entries if e.ok]
    baseline_entries = [e for e in ok_entries if e.is_baseline]
    real_entries = [e for e in ok_entries if not e.is_baseline]
    assert baseline_entries and real_entries

    best_baseline_f1 = max(e.metrics["f1"] for e in baseline_entries)
    best_real_f1 = max(e.metrics["f1"] for e in real_entries)

    # A baseline that predicts "not fraud" every time gets macro-F1 close
    # to ~0.5 (perfect on the majority class, zero on the minority class,
    # averaged unweighted) — nowhere near a deceptive ~0.88.
    assert best_baseline_f1 < 0.65, (
        f"baseline macro-F1 ({best_baseline_f1:.3f}) is suspiciously high for a classifier "
        f"that never predicts the minority class — weighted-averaging bug may have regressed"
    )
    # And a real model must show a clear, honest gap over that baseline —
    # not the misleading +0.02 the weighted metric used to show.
    assert best_real_f1 - best_baseline_f1 > 0.05
