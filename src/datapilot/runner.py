"""The whole DataPilot run as plain functions (no Streamlit): build -> race -> tune -> finalize.

The app calls `run_all`, which runs the four stages in order. `progress(stage, done, total, label)` is optional."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .experiment import ExperimentEngine
from .explain import prep_label
from .finalize import fit_best_with_fallback
from .model_zoo import get_models
from .planner import PipelineOptions, PipelinePlanner, train_test_split_data
from .profiler import DatasetProfiler
from .reasoner import Priority, TaskType, UserPreferences
from .scale import subsample_for_search
from .tuning import tune_top_entries


@dataclass(frozen=True)
class Effort:
    cv: int
    backups: bool
    tune_top_k: int
    tune_iter: int
    note: str
    repeats: int = 1   # repeated CV on small data, so the winner is not luck


BIG_DATA_FOLDS = 3
BIG_DATA_FOLDS_ROWS = 15_000

EFFORTS: dict[str, Effort] = {
    "Quick": Effort(3, False, 2, 6, "3-fold CV, core models, light tuning"),
    "Balanced": Effort(5, True, 3, 15, "5-fold CV, all models, moderate tuning", 2),
    "Thorough": Effort(5, True, 4, 40, "5-fold CV, all models, heavy tuning", 3),
}


@dataclass
class RunConfig:
    target: str
    task: TaskType
    priority: Priority = Priority.BALANCED
    options: PipelineOptions = field(default_factory=PipelineOptions)
    effort: str = "Balanced"
    primary: str | None = None
    seed: int = 42            # decides which rows are held out
    test_df: pd.DataFrame | None = None   # the user's own test file (with or without answers)

    @property
    def eff(self) -> Effort:
        return EFFORTS[self.effort]


def engine_metric(primary: str | None) -> str | None:
    """RMSLE is RMSE on log(1+y): the engine ranks by RMSE while it trains on the logged target."""
    return "rmse" if primary == "rmsle" else primary


def _use_log_target(cfg: RunConfig, y_train) -> tuple[bool, str]:
    if cfg.primary != "rmsle" or cfg.task != TaskType.REGRESSION:
        return False, ""
    if float(np.nanmin(np.asarray(y_train, dtype=float))) < 0:
        return False, "RMSLE needs a target that is never negative; ranked by RMSE on the original scale instead."
    return True, "Ranked by RMSLE: models are trained on log(1+target) and predictions are converted back."


def _p(progress, stage, done, total, label=""):
    if progress:
        progress(stage, done, total, label)


def build_state(df: pd.DataFrame, cfg: RunConfig, dataset: str = "", progress=None) -> dict:
    t0 = time.perf_counter()
    _p(progress, "build", 0, 4, "Reading and checking every column")
    profile = DatasetProfiler().profile(df, target=cfg.target)
    prefs = UserPreferences(target_column=cfg.target, task_type=cfg.task, priority=cfg.priority)
    planner = PipelinePlanner()
    opts = cfg.options
    groups = planner.split_columns([c for c in df.columns if c not in opts.excluded_columns], profile)
    candidates = planner.plan(list(df.columns), profile, prefs, opts)
    rescue = planner.plan_rescue(list(df.columns), profile, opts)
    _p(progress, "build", 1, 4, "Setting rows aside for the final test")
    test_df = cfg.test_df.drop(columns=[c for c in opts.excluded_columns if c in cfg.test_df.columns]) \
        if cfg.test_df is not None else None
    split = train_test_split_data(df.drop(columns=list(opts.excluded_columns)), cfg.target, cfg.task,
                                  random_state=cfg.seed, test_df=test_df)
    fit_results = []
    for i, cand in enumerate(candidates):
        _p(progress, "build", 2 + i / max(len(candidates), 1), 4, f"Building features: {prep_label(cand.name)}")
        try:
            cand.pipeline.fit(split.X_train, split.y_train)
            n_out = cand.pipeline.transform(split.X_train).shape[1]
            cand.pipeline.transform(split.X_test)  # prove the held-out side works too
            fit_results.append({"name": cand.name, "ok": True, "n_features": n_out})
            _p(progress, "build", 2 + (i + 1) / max(len(candidates), 1), 4, f"Built {n_out} features: {prep_label(cand.name)}")
        except Exception as e:  # noqa: BLE001 — shown to the user, not hidden
            fit_results.append({"name": cand.name, "ok": False, "error": str(e)})
    _p(progress, "build", 4, 4, "Ready")
    log_target, log_note = _use_log_target(cfg, split.y_train)
    return {
        "seconds": time.perf_counter() - t0,
        "log_target": log_target, "log_note": log_note,
        "rescue": rescue, "dataset": dataset, "target": cfg.target, "task": cfg.task.value,
        "groups": groups, "candidates": candidates, "fit_results": fit_results,
        "n_train": len(split.X_train), "n_test": len(split.X_test),
        "score_mode": split.score_mode, "test_note": split.test_note,
        "n_dropped_missing_target": split.n_dropped_missing_target,
        "label_encoder": split.label_encoder, "stratify_note": split.stratify_note,
        "target_note": split.target_note, "n_dropped_duplicates": split.n_dropped_duplicates,
        "profile": profile, "prefs": prefs, "options": opts, "priority": cfg.priority.value,
        "X_train": split.X_train, "X_test": split.X_test,
        "y_train": split.y_train, "y_test": split.y_test,
    }


def search_data(state: dict):
    """(X, y, note) used for model search and tuning; the winner is refit on everything."""
    Xs, ys, note = subsample_for_search(state["X_train"], state["y_train"], TaskType(state["task"]))
    if state.get("log_target"):
        ys = np.log1p(np.asarray(ys, dtype=float))
    return Xs, ys, note


def race(state: dict, cfg: RunConfig, dataset: str = "", progress=None) -> dict:
    t0 = time.perf_counter()
    task = TaskType(state["task"])
    Xs, ys, _ = search_data(state)
    models = get_models(task, backups=cfg.eff.backups, n_rows=len(Xs))
    n_splits = min(cfg.eff.cv, BIG_DATA_FOLDS) if len(Xs) >= BIG_DATA_FOLDS_ROWS else cfg.eff.cv   # plenty of rows: 3 folds rank models as well as 5
    engine = ExperimentEngine(task_type=task, n_splits=n_splits, n_repeats=(cfg.eff.repeats if len(Xs) <= 3000 else 1),
                              primary=engine_metric(cfg.primary))
    entries = engine.run_leaderboard(
        state["candidates"], models, Xs, ys, rescue=state.get("rescue"),
        progress=lambda d, t, label: _p(progress, "race", d, t, label))
    if state.get("log_target"):   # relabel: the engine's RMSE on log(1+y) is exactly RMSLE
        for e in entries:
            e.primary_metric_name = "rmsle"
            if e.ok:
                e.metrics = {"rmsle": e.metrics.get("rmse", float("nan")), "r2 (log scale)": e.metrics.get("r2", float("nan"))}
    return {
        "seconds": time.perf_counter() - t0,
        "primary": "rmsle" if state.get("log_target") else engine_metric(cfg.primary),
        "dataset": dataset or state["dataset"], "target": state["target"],
        "entries": entries, "smote_note": getattr(engine, "smote_note", ""),
        "cv_fold_note": getattr(engine, "cv_fold_note", ""), "rescue_note": getattr(engine, "rescue_note", ""),
        "models": [m.name for m in models],
    }


def tune(state: dict, lb: dict, cfg: RunConfig, dataset: str = "", progress=None) -> dict:
    t0 = time.perf_counter()
    task = TaskType(state["task"])
    ok = [e for e in lb["entries"] if e.ok]
    Xs, ys, _ = search_data(state)
    models = get_models(task, backups=True, n_rows=len(Xs))
    _p(progress, "tune", 0, 1, "Tuning the top models")
    top_k, n_iter = cfg.eff.tune_top_k, cfg.eff.tune_iter
    results = tune_top_entries(ok, state["candidates"], models, Xs, ys, task,
                               top_k=top_k, n_iter=n_iter, cv_splits=3,
                               primary=engine_metric(lb.get("primary")))
    _p(progress, "tune", 1, 1, "Tuned")
    return {"dataset": dataset or state["dataset"], "results": results, "seconds": time.perf_counter() - t0}


def finalize(state: dict, lb: dict, tuning: dict | None, dataset: str = "", progress=None) -> dict:
    t0 = time.perf_counter()
    task = TaskType(state["task"])
    _p(progress, "final", 0, 1, "Refitting the winner")
    ff = fit_best_with_fallback(
        lb["entries"], state["candidates"], (tuning or {}).get("results", []), task,
        state["X_train"], state["y_train"], state["X_test"], state["y_test"],
        state.get("label_encoder"), rescue=state.get("rescue"), log_target=bool(state.get("log_target")),
        score_mode=state.get("score_mode", "holdout"))
    _p(progress, "final", 1, 1, ff.model_name)
    return {
        "seconds": time.perf_counter() - t0,
        "dataset": dataset or state["dataset"], "result": ff.evaluation,
        "used_tuned_params": bool(ff.params), "best_params": dict(ff.params),
        "uses_smote": bool(ff.uses_smote), "base_model_name": ff.base_model_name,
        "final_pipeline": ff.pipeline, "feature_columns": list(state["X_train"].columns),
        "fallback_notes": list(ff.notes),
    }


def run_all(df: pd.DataFrame, cfg: RunConfig, dataset: str = "", progress=None):
    """Everything, in order. Returns (state, leaderboard, tuning, final)."""
    state = build_state(df, cfg, dataset, progress)
    lb = race(state, cfg, dataset, progress)
    if not any(e.ok for e in lb["entries"]):
        raise RuntimeError("No model could be trained on this data.")
    tun = tune(state, lb, cfg, dataset, progress)
    fin = finalize(state, lb, tun, dataset, progress)
    return state, lb, tun, fin
