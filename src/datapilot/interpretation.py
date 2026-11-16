"""
Stage 6 — Model interpretation.

Extracts real feature importances/coefficients from a fitted pipeline,
mapped back to human-readable feature names via the pipeline's own
get_feature_names_out() chain (including the custom transformers in
transformers.py, which implement it for exactly this reason) — not a
hand-maintained name list that could drift out of sync with the real
preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.pipeline import Pipeline


@dataclass
class FeatureImportance:
    feature: str
    importance: float  # always >= 0 — magnitude, direction noted separately
    direction: str | None = None  # "+" / "-" for linear coefficients, None for tree importances


@dataclass
class InterpretationResult:
    kind: str  # "coefficients" | "tree_importances" | "unavailable"
    features: list[FeatureImportance]
    note: str = ""


def _unwrap(model):
    """The estimator that carries the learned weights, seen through our two wrappers."""
    from sklearn.compose import TransformedTargetRegressor

    from .ensemble import ThresholdTunedClassifier
    while True:
        if isinstance(model, TransformedTargetRegressor):
            model = model.regressor_
        elif isinstance(model, ThresholdTunedClassifier):
            model = model.estimator_
        else:
            return model


def _ensemble_importances(model):
    """(normalised importances, how many members contributed, member count) for a Voting/Stacking
    ensemble: the average of its members' tree importances (stacking: weighted by the meta-model)."""
    members = list(getattr(model, "estimators_", []) or [])
    if not members:
        return None
    weights = np.ones(len(members))
    meta = getattr(model, "final_estimator_", None)
    if meta is not None and hasattr(meta, "coef_") and np.ravel(meta.coef_).shape[0] == len(members):
        weights = np.abs(np.ravel(meta.coef_))
    vecs, used = [], []
    for m, w in zip(members, weights):
        m = _unwrap(m)
        fi = getattr(m, "feature_importances_", None)
        if fi is not None and np.sum(fi) > 0:
            vecs.append(np.asarray(fi, dtype=float) / np.sum(fi))
            used.append(w)
    if not vecs:
        return None
    return np.average(vecs, axis=0, weights=used), len(vecs), len(members)


def interpret(fitted_pipeline: Pipeline, X_train=None, top_n: int = 15) -> InterpretationResult:
    model = _unwrap(fitted_pipeline.steps[-1][1])
    # Exclude the model AND, for a SMOTE pipeline, the "smote" resampling
    # step — it only implements fit_resample (no transform/
    # get_feature_names_out), and it doesn't change feature identity
    # anyway (it adds synthetic rows, not columns). Wrapping it into the
    # preprocessing-only chain would break get_feature_names_out() and
    # silently degrade every SMOTE-selected model to positional feature
    # names instead of real ones.
    preprocessing_steps = [
        (name, step) for name, step in fitted_pipeline.steps[:-1] if name != "smote"
    ]
    preprocessing = Pipeline(preprocessing_steps)

    try:
        feature_names = list(preprocessing.get_feature_names_out())
    except Exception:
        # Fall back to positional names rather than crashing interpretation
        # entirely — a real model still gets *some* honest answer.
        n_features = getattr(model, "n_features_in_", None)
        feature_names = [f"feature_{i}" for i in range(n_features or 0)]

    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)  # multiclass: average magnitude across classes

        note = ""
        if X_train is not None:
            # Raw coefficients are only comparable across features when
            # every feature is on the same scale. This pipeline may not
            # scale numeric features (the conservative/balanced candidates
            # don't) — a coefficient of 1.15 on a $0-500,000 feature and a
            # coefficient of 4000 on a 0/1 flag are NOT comparable as-is;
            # the 0/1 flag would misleadingly look "more important." Multiply
            # by each feature's real training-data spread (std) to get a
            # fair, comparable effect size — same trick as standardizing
            # inputs, applied after the fact so it works regardless of
            # whether the pipeline itself scaled anything.
            try:
                X_transformed = preprocessing.transform(X_train)
                X_transformed = np.asarray(X_transformed)
                feature_std = np.std(X_transformed, axis=0)
                if len(feature_std) == len(coef):
                    coef = coef * feature_std
                    note = "Coefficients scaled by each feature's training-data spread for fair comparison."
            except Exception:
                note = "Could not verify feature scale — raw coefficients may not be directly comparable."

        result = _rank(feature_names, coef, kind="coefficients", signed=True, top_n=top_n)
        if result.kind != "unavailable":  # don't clobber a real mismatch note from _rank
            result.note = note
        return result

    if hasattr(model, "feature_importances_"):
        # Tree-based importances are already scale-invariant by construction
        # (splits are chosen by information gain, not by a feature's raw
        # numeric range), so no standardization is needed here.
        importances = np.asarray(model.feature_importances_)
        return _rank(feature_names, importances, kind="tree_importances", signed=False, top_n=top_n)

    ens = _ensemble_importances(model)
    if ens is not None:
        vec, used, total = ens
        res = _rank(feature_names, vec, kind="tree_importances", signed=False, top_n=top_n)
        if res.kind != "unavailable":
            res.note = f"Average over the ensemble's members ({used} of {total} expose importances)."
        return res

    return InterpretationResult(
        kind="unavailable", features=[],
        note=f"{type(model).__name__} exposes neither coef_ nor feature_importances_ — "
             f"no interpretation available for this model type.",
    )


def _rank(feature_names, values, kind: str, signed: bool, top_n: int) -> InterpretationResult:
    if len(feature_names) != len(values):
        # A real mismatch (e.g. get_feature_names_out() fell back to
        # positional names of the wrong length) — say so, don't pretend.
        return InterpretationResult(
            kind="unavailable", features=[],
            note=f"Feature name count ({len(feature_names)}) didn't match model "
                 f"input count ({len(values)}) — skipped rather than mislabel.",
        )
    order = np.argsort(np.abs(values))[::-1][:top_n]
    features = [
        FeatureImportance(
            feature=feature_names[i],
            importance=float(abs(values[i])),
            direction=("+" if values[i] >= 0 else "-") if signed else None,
        )
        for i in order
    ]
    return InterpretationResult(kind=kind, features=features)
