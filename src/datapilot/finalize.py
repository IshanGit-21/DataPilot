"""Final fit with a fallback chain.

The best leaderboard entry is refit on all training rows and scored on the held-out test
set. If that crashes (an odd dtype, a model that behaves on CV folds but not on the full
set, memory), the next best entries are tried, then a plain rescue pipeline with a simple
model, so a run always ends with a usable model and an honest note about what happened."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from .ensemble import log_target_model, resolve_model_spec
from .evaluation import PrecomputedPredictions, evaluate_on_held_out_test
from .imbalance import build_smote_pipeline
from .model_zoo import use_all_cores
from .reasoner import TaskType
from .tuning import pick_best_entry


@dataclass
class FinalFit:
    pipeline: object
    entry: object                     # LeaderboardEntry that was finalized (None for last resort)
    pipeline_name: str
    model_name: str                   # name as shown (includes " + SMOTE" when used)
    base_model_name: str              # name without " + SMOTE", for rebuilding
    params: dict
    uses_smote: bool
    evaluation: object
    notes: list[str] = field(default_factory=list)


def _build_pipe(steps, model_name: str, uses_smote: bool, params: dict, task: TaskType,
                log_target: bool, all_cores: bool):
    base = model_name.removesuffix(" + SMOTE") if uses_smote else model_name
    est = resolve_model_spec(base, task).build()
    if all_cores:
        est = use_all_cores(est)
    if params:
        est.set_params(**params)
    if log_target:   # the search ran on log(1+y); the delivered model maps back to the real scale
        est = log_target_model(est)
    pipe = (build_smote_pipeline(steps, est) if uses_smote else Pipeline(list(steps) + [("model", est)]))
    return pipe, base


def cross_check(steps, pipeline_name: str, model_name: str, uses_smote: bool, params: dict, task: TaskType,
                X, y, log_target: bool, label_encoder, folds: int = 5):
    """Score the chosen model without hiding any rows: every row is predicted by a model that
    was fitted on the other folds and never saw it. Preparation is re-fitted inside each fold."""
    from joblib import Parallel, delayed
    X = X.reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    classify = task == TaskType.CLASSIFICATION
    if classify and int(y.value_counts().min()) >= folds:
        splits = list(StratifiedKFold(folds, shuffle=True, random_state=42).split(X, y))
    else:
        splits = list(KFold(folds, shuffle=True, random_state=42).split(X))

    def one(tr, va):
        pipe, _ = _build_pipe(steps, model_name, uses_smote, params, task, log_target, all_cores=False)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba = pipe.predict_proba(X.iloc[va]) if classify and hasattr(pipe, "predict_proba") else None
        return va, pipe.predict(X.iloc[va]), proba

    parts = Parallel(n_jobs=-1)(delayed(one)(tr, va) for tr, va in splits)
    pred = np.empty(len(X), dtype=parts[0][1].dtype)
    proba = None
    if all(p[2] is not None and p[2].shape[1] == parts[0][2].shape[1] for p in parts):
        proba = np.zeros((len(X), parts[0][2].shape[1]))
    for va, pr, pb in parts:
        pred[va] = pr
        if proba is not None:
            proba[va] = pb
    ev = evaluate_on_held_out_test(pipeline_name, model_name, PrecomputedPredictions(pred, proba), X, y, task,
                                   label_encoder=label_encoder)
    ev.scored_by = "cv"
    return ev


def _fit_one(candidate, model_name: str, uses_smote: bool, params: dict, task: TaskType,
             X_train, y_train, X_test, y_test, label_encoder, log_target: bool = False,
             score_mode: str = "holdout"):
    pipe, base = _build_pipe(candidate.pipeline.steps, model_name, uses_smote, params, task, log_target, all_cores=True)
    pipe.fit(X_train, y_train)
    if score_mode == "cv":
        ev = cross_check(candidate.pipeline.steps, candidate.name, model_name, uses_smote, params, task, X_test, y_test,
                         log_target, label_encoder)
    else:
        ev = evaluate_on_held_out_test(candidate.name, model_name, pipe, X_test, y_test, task,
                                       label_encoder=label_encoder)
        ev.scored_by = "external" if score_mode == "external" else "holdout"
    if not ev.ok:
        raise RuntimeError(ev.error or "evaluation failed")
    return pipe, ev, base


def fit_best_with_fallback(entries, candidates, tuning, task: TaskType, X_train, y_train, X_test, y_test,
                           label_encoder=None, rescue=None, log_target: bool = False,
                           score_mode: str = "holdout") -> FinalFit:
    by_name = {c.name: c for c in candidates}
    best, best_params, _ = pick_best_entry(entries, tuning)
    tuned = {(r.pipeline_name, r.model_name): dict(r.best_params)
             for r in (tuning or []) if r.ok and r.tunable}
    plan = []
    if best is not None:
        plan.append((best, best_params))
    seen = {(best.pipeline_name, best.model_name)} if best is not None else set()
    for e in sorted((e for e in entries if e.ok and not e.is_baseline),
                    key=lambda e: -e.primary_mean):
        key = (e.pipeline_name, e.model_name)
        if key not in seen:
            seen.add(key)
            plan.append((e, tuned.get(key, {})))

    failures: list[str] = []
    for entry, params in plan:
        cand = by_name.get(entry.pipeline_name)
        if cand is None:
            continue
        try:
            pipe, ev, base = _fit_one(cand, entry.model_name, bool(entry.uses_smote), params, task,
                                      X_train, y_train, X_test, y_test, label_encoder, log_target, score_mode)
        except Exception as e:  # noqa: BLE001 — the whole point is to try something else
            failures.append(f"{entry.model_name} on {entry.pipeline_name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        notes = []
        if failures:
            notes.append(f"The top choice could not be refit on the full training set ({failures[0]}). "
                         f"Fell back to {entry.model_name} on {entry.pipeline_name}, the next best that worked.")
        return FinalFit(pipe, entry, entry.pipeline_name, entry.model_name, base, params,
                        bool(entry.uses_smote), ev, notes)

    # last resort: plain pipeline + the simplest dependable model
    if rescue is not None:
        simple = "Logistic Regression" if task == TaskType.CLASSIFICATION else "Ridge"
        try:
            pipe, ev, base = _fit_one(rescue, simple, False, {}, task, X_train, y_train, X_test, y_test,
                                      label_encoder, log_target, score_mode)
            return FinalFit(pipe, None, rescue.name, simple, base, {}, False, ev,
                            [f"Every leaderboard model failed to refit ({len(failures)} tried); "
                             f"used {simple} on the plain rescue pipeline instead."])
        except Exception as e:  # noqa: BLE001
            failures.append(f"{simple} on rescue: {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError("No model could be fitted on the full training set. Tried: " + " | ".join(failures[:5]))
