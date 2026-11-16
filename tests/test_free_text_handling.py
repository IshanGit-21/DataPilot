import numpy as np
import pandas as pd

from datapilot.experiment import ExperimentEngine
from datapilot.model_zoo import get_models
from datapilot.planner import PipelinePlanner, train_test_split_data
from datapilot.profiler import DatasetProfiler
from datapilot.reasoner import Priority, TaskType, UserPreferences
from datapilot.transformers import TextFeaturizer

DISASTER_WORDS = ["fire", "flood", "earthquake", "emergency", "evacuate", "rescue", "storm", "damage"]
CASUAL_WORDS = ["lol", "party", "weekend", "coffee", "movie", "friend", "music", "beach"]


def make_tweet_like_df(n=600, seed=0):
    """A free-text column whose content genuinely predicts the target
    (disaster words vs casual words), alongside short categorical columns
    that must NOT be mistaken for free text — mirrors the real Disaster
    Tweets dataset's text/keyword/location/target shape.
    """
    rng = np.random.default_rng(seed)
    is_disaster = rng.integers(0, 2, n)
    texts = []
    for d in is_disaster:
        vocab = DISASTER_WORDS if d else CASUAL_WORDS
        words = rng.choice(vocab, size=rng.integers(8, 15), replace=True)
        texts.append(" ".join(words))

    df = pd.DataFrame({
        "id": range(n),
        "text": texts,                                          # free text — must survive
        "keyword": rng.choice(["fire", "flood", ""], n),          # short categorical — must NOT be "free text"
        "location": rng.choice(["NY", "LA", "SF", ""], n),        # short categorical — must NOT be "free text"
        "target": is_disaster,
    })
    return df


def test_free_text_column_detected_and_not_misflagged_as_identifier():
    df = make_tweet_like_df()
    profile = DatasetProfiler().profile(df, target="target")

    assert profile.columns["text"].is_free_text is True
    assert profile.columns["text"].is_likely_identifier is False, (
        "a free-text column is routinely ~100% unique from natural language "
        "variety, not identifier-ness — it must not be dropped as an identifier"
    )
    # short categorical columns must NOT be misdetected as free text
    assert profile.columns["keyword"].is_free_text is False
    assert profile.columns["location"].is_free_text is False
    # a real auto-increment id column is still correctly flagged
    assert profile.columns["id"].is_likely_identifier is True


def test_free_text_column_survives_into_pipeline_feature_set():
    df = make_tweet_like_df()
    profile = DatasetProfiler().profile(df, target="target")
    planner = PipelinePlanner()
    groups = planner.split_columns(list(df.columns), profile)

    assert "text" in groups.free_text
    assert "text" not in (groups.numeric + groups.categorical_low + groups.categorical_high + groups.datetime)
    assert "id" not in groups.free_text  # the real id column stays excluded as an identifier


def test_tfidf_transformer_produces_real_discriminative_features():
    df = make_tweet_like_df()
    featurizer = TextFeaturizer(max_features=50)
    featurizer.fit(df["text"])
    out = featurizer.transform(df["text"])
    names = list(featurizer.get_feature_names_out())

    assert out.shape[0] == len(df)
    assert out.shape[1] > 0
    assert all(n.startswith("text_tfidf_") for n in names)
    # at least one disaster-vocabulary word should have made it into the vocabulary
    assert any(any(w in n for w in DISASTER_WORDS) for n in names)


def test_all_three_candidates_use_free_text_with_decision_records():
    """Regression test for a real gap: the aggressive candidate had TF-IDF
    wired into its actual ColumnTransformer but was missing the matching
    DecisionRecord — a silent decision, breaking the project's core
    transparency guarantee even though the feature itself worked.
    """
    df = make_tweet_like_df()
    profile = DatasetProfiler().profile(df, target="target")
    prefs = UserPreferences(target_column="target", task_type=TaskType.CLASSIFICATION, priority=Priority.BALANCED)
    candidates = PipelinePlanner().plan(list(df.columns), profile, prefs)

    for candidate in candidates:
        has_tfidf_step = any(
            step_name.startswith("text_") for step_name, _, _ in candidate.pipeline.steps[0][1].transformers
        )
        assert has_tfidf_step, f"{candidate.name}: expected a text_* ColumnTransformer branch"
        has_decision = any("TF-IDF" in d.what for d in candidate.decisions)
        assert has_decision, f"{candidate.name}: TF-IDF is wired in but has no matching DecisionRecord"


def test_end_to_end_leaderboard_beats_baseline_using_real_text_signal():
    df = make_tweet_like_df()
    profile = DatasetProfiler().profile(df, target="target")
    prefs = UserPreferences(target_column="target", task_type=TaskType.CLASSIFICATION, priority=Priority.BALANCED)
    split = train_test_split_data(df, "target", TaskType.CLASSIFICATION)
    candidates = PipelinePlanner().plan(list(df.columns), profile, prefs)

    for c in candidates:
        c.pipeline.fit(split.X_train, split.y_train)
        Xt_train = c.pipeline.transform(split.X_train)
        Xt_test = c.pipeline.transform(split.X_test)
        assert Xt_train.shape[1] == Xt_test.shape[1]
        assert np.all(np.isfinite(Xt_train))

    models = get_models(TaskType.CLASSIFICATION)
    engine = ExperimentEngine(task_type=TaskType.CLASSIFICATION, n_splits=3, n_repeats=1)
    entries = engine.run_leaderboard(candidates, models, split.X_train, split.y_train)
    ok_entries = [e for e in entries if e.ok]
    assert len(ok_entries) == len(entries), f"failures: {[(e.model_name, e.error) for e in entries if not e.ok]}"

    best = max(e.primary_mean for e in ok_entries if not e.is_baseline)
    baseline = max(e.primary_mean for e in ok_entries if e.is_baseline)
    assert best > baseline + 0.2, (
        "the disaster-word vocabulary is a strong, clean signal — a real model using the "
        "TF-IDF features should beat the baseline by a wide margin, proving the text column "
        "is actually being used, not just silently surviving as a no-op"
    )
