"""
Stage 5 — Imbalance handling (SMOTE).

Real minority-class oversampling via imbalanced-learn, composed with
imblearn's own Pipeline class rather than sklearn's — that distinction is
the entire safety guarantee here. A sampler step in an imblearn Pipeline
only ever runs during fit_resample (i.e. on the training fold), and is
skipped entirely during transform/predict. A plain sklearn Pipeline has no
such concept and would either reject a sampler step outright or (worse)
apply it at predict time too, which would be invalid — you cannot
"resample" data you're trying to predict on.

class_weight='balanced' (already used by every classification model in
model_zoo.py) is a real, cheap, zero-cost answer to imbalance. SMOTE is a
heavier, complementary one: it changes the training distribution itself
rather than just reweighting the loss, and is only worth the extra fits
when the classes are genuinely imbalanced and there's enough minority-class
data for it to synthesize from.
"""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

IMBALANCE_RATIO_THRESHOLD = 0.35  # minority-class share below this triggers SMOTE
MIN_MINORITY_COUNT_FOR_SMOTE = 6  # SMOTE's default k_neighbors=5 needs at least 6 minority samples


def class_balance_ratio(y) -> float:
    """Minority class's share of the total. 0.5 is perfectly balanced for
    a binary target; lower means more imbalanced. Returns 1.0 for a
    single-class y (nothing to balance)."""
    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.sum() == 0:
        return 1.0
    return float(counts.min() / counts.sum())


def smote_is_viable(y, threshold: float = IMBALANCE_RATIO_THRESHOLD) -> tuple[bool, str]:
    """Whether SMOTE is both warranted (real imbalance) and possible (enough
    minority-class rows for it to synthesize neighbors from). Returns
    (viable, reason) so the UI can say exactly why it did or didn't run,
    rather than silently skipping."""
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        return False, "target has fewer than 2 classes"
    ratio = class_balance_ratio(y)
    if ratio >= threshold:
        return False, f"classes are reasonably balanced (minority share {ratio:.1%})"
    if counts.min() < MIN_MINORITY_COUNT_FOR_SMOTE:
        return False, (
            f"minority class has only {int(counts.min())} rows — too few for SMOTE's "
            f"nearest-neighbor synthesis (needs >= {MIN_MINORITY_COUNT_FOR_SMOTE})"
        )
    return True, f"minority class share {ratio:.1%} — SMOTE applies inside training folds only"


def build_smote_pipeline(preprocessing_steps: list[tuple], estimator, random_state: int = 42) -> ImbPipeline:
    """preprocessing_steps: the real steps from a PipelineCandidate.pipeline
    (e.g. [("preprocess", ColumnTransformer(...))]). Returns an imblearn
    Pipeline: preprocessing -> SMOTE -> model. SMOTE sees only the
    preprocessed (fully numeric) training fold, and never touches
    validation/test data — that's enforced by imblearn's Pipeline, not by
    convention here.
    """
    return ImbPipeline(
        list(preprocessing_steps) + [
            ("smote", SMOTE(random_state=random_state, k_neighbors=min(5, MIN_MINORITY_COUNT_FOR_SMOTE - 1))),
            ("model", estimator),
        ]
    )
