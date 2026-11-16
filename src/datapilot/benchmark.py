"""Stage 9 — benchmark harness.

Runs the real Stage 1-6 flow end to end on named datasets and records what
actually happened, so changes to DataPilot can be compared run over run.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from .diagnostics import (
    assess_results,
    detect_leakage_suspects,
    sensitive_input_columns,
)
from .experiment import ExperimentEngine
from .finalize import fit_best_with_fallback
from .model_zoo import get_models
from .planner import PipelineOptions, PipelinePlanner, train_test_split_data
from .profiler import DatasetProfiler
from .reasoner import Priority, UserPreferences
from .scale import DEFAULT_SEARCH_ROWS, subsample_for_search
from .setup import infer_task_type, suggest_input_exclusions, suggest_targets
from .tuning import pick_best_entry, tune_top_entries


@dataclass
class BenchmarkRow:
    dataset: str
    rows: int
    cols: int
    target: str
    task: str
    target_suggested_first: bool | None   # None if no expected-target check was possible
    best_pipeline: str = ""
    best_model: str = ""
    baseline_cv: float | None = None
    best_cv: float | None = None
    heldout_primary: float | None = None
    heldout_secondary: float | None = None
    heldout_roc_auc: float | None = None
    notes: list[str] = field(default_factory=list)
    primary_metric: str = ""
    runtime_seconds: float = 0.0
    leakage_suspects: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def uplift(self) -> float | None:
        if self.baseline_cv is None or self.best_cv is None:
            return None
        return self.best_cv - self.baseline_cv


def run_case(name: str, df: pd.DataFrame, target: str, *, tune: bool = False,
             n_splits: int = 3, options: PipelineOptions | None = None,
             primary: str | None = None, search_rows: int = DEFAULT_SEARCH_ROWS) -> BenchmarkRow:
    t0 = time.time()
    row = BenchmarkRow(name, len(df), df.shape[1], target, "", None)
    try:
        profile = DatasetProfiler().profile(df, target=target)
        sug = suggest_targets(df, profile)
        row.target_suggested_first = bool(sug) and sug[0].column == target
        task = infer_task_type(df[target])
        row.task = task.value
        prefs = UserPreferences(target_column=target, task_type=task, priority=Priority.BALANCED)
        opts = options or PipelineOptions(excluded_columns=list(suggest_input_exclusions(profile, target)))
        df_model = df.drop(columns=list(opts.excluded_columns))
        split = train_test_split_data(df_model, target, task)
        for n in (split.target_note, ):
            if n:
                row.notes.append(n)
        if split.n_dropped_duplicates:
            row.notes.append(f"{split.n_dropped_duplicates} exact duplicate rows dropped before splitting")
        cands = PipelinePlanner().plan(list(df.columns), profile, prefs, opts)
        models = get_models(task, backups=True, n_rows=len(split.X_train))
        Xs, ys, snote = subsample_for_search(split.X_train, split.y_train, task, search_rows)
        if snote:
            row.notes.append(snote)
        engine = ExperimentEngine(task_type=task, n_splits=n_splits, n_repeats=1, primary=primary)
        rescue = PipelinePlanner().plan_rescue(list(df.columns), profile, opts)
        entries = engine.run_leaderboard(cands, models, Xs, ys, rescue=rescue)
        if engine.rescue_note:
            row.notes.append(engine.rescue_note)
        ok = [e for e in entries if e.ok]
        if not ok:
            row.error = "no model trained successfully"
            return row
        tuning = []
        if tune:
            tuning = tune_top_entries(entries, cands, models, Xs, ys, task,
                                      top_k=2, n_iter=12, cv_splits=3, primary=primary)
        best, params, best_score = pick_best_entry(entries, tuning)
        base = [e.primary_mean for e in ok if e.is_baseline]
        row.baseline_cv = max(base) if base else None
        row.best_cv = best_score
        row.primary_metric = best.primary_metric_name
        row.best_pipeline, row.best_model = best.pipeline_name, best.model_name

        ff = fit_best_with_fallback(entries, cands, tuning, task, split.X_train, split.y_train,
                                    split.X_test, split.y_test, split.label_encoder, rescue=rescue)
        ev = ff.evaluation
        row.notes += ff.notes
        row.best_pipeline, row.best_model = ff.pipeline_name, ff.model_name
        if ev.regression:
            row.heldout_primary, row.heldout_secondary = ev.regression.r2, ev.regression.rmse
        elif ev.classification:
            row.heldout_primary, row.heldout_secondary = ev.classification.f1, ev.classification.accuracy
            row.heldout_roc_auc = ev.classification.roc_auc
            if primary == "roc_auc":
                row.heldout_primary = ev.classification.roc_auc
        row.leakage_suspects = [f"{s.column}: {s.evidence}" for s in
                                detect_leakage_suspects(split.X_train, split.y_train, task)]
        row.warnings = assess_results(entries, ev)
        sens = sensitive_input_columns(split.X_train.columns)
        if sens:
            row.warnings.append("Sensitive attributes used as inputs: " + ", ".join(sens))
    except Exception as e:  # noqa: BLE001 — a benchmark must record failures, not die on them
        row.error = f"{type(e).__name__}: {e}"
    finally:
        row.runtime_seconds = time.time() - t0
    return row


def benchmark_to_markdown(rows: list[BenchmarkRow]) -> str:
    def f(v, nd=3):
        return "-" if v is None else f"{v:.{nd}f}"
    L = ["# DataPilot Benchmark", "",
         "| Dataset | Rows x cols | Task | Target found | Best (pipeline / model) | Baseline CV | Best CV | Uplift | Held-out | Runtime (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.error:
            L.append(f"| {r.dataset} | {r.rows} x {r.cols} | {r.task or '-'} | - | FAILED: {r.error[:80]} | - | - | - | - | {r.runtime_seconds:.0f} |")
            continue
        found = "-" if r.target_suggested_first is None else ("yes" if r.target_suggested_first else "no")
        L.append(f"| {r.dataset} | {r.rows} x {r.cols} | {r.task} | {found} | {r.best_pipeline} / {r.best_model} | "
                 f"{f(r.baseline_cv)} | {f(r.best_cv)} | {f(r.uplift)} | {r.primary_metric}={f(r.heldout_primary)} | {r.runtime_seconds:.0f} |")
    notes = [r for r in rows if r.leakage_suspects or r.warnings]
    if notes:
        L += ["", "## Flags"]
        for r in notes:
            for s in r.leakage_suspects:
                L.append(f"- **{r.dataset}** possible leakage — {s}")
            for w in r.warnings:
                L.append(f"- **{r.dataset}** {w}")
    return "\n".join(L) + "\n"
