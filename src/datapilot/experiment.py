"""
Stage 4/5 — Experiment Engine.

Evaluates every (pipeline candidate x model) combination under the *same*
repeated cross-validation protocol, using task-appropriate metrics computed
in one CV pass each (not one CV run per metric — that would be wasteful).

Leakage safety is structural: every transformer (imputer, encoder, scaler, clipper,
selector) is fitted on the training fold of each split only (see parallel.py), and all
model x fold fits run in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer, roc_auc_score
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
)

from .ensemble import extra_specs_for
from .imbalance import build_smote_pipeline, smote_is_viable
from .explain import prep_label
from .parallel import ModelJob, cross_validate_jobs
from .model_zoo import ModelSpec
from .planner import PipelineCandidate
from .reasoner import TaskType


def _safe_roc_auc_ovr_weighted(estimator, X, y):
    """roc_auc_score raises when a CV fold's validation split doesn't
    contain every class the model was trained on — a real failure mode,
    not a hypothetical: it happens routinely with a rare class spread
    thin across folds, and previously took down the ENTIRE leaderboard
    entry (every other metric included) because cross_validate's
    error_score='raise' treats any scorer exception as a hard failure for
    that whole fold. Returning NaN here instead means one fragile metric
    degrades gracefully — nanmean below simply excludes the bad folds —
    rather than silently erasing accuracy/precision/recall/F1/F2 too.
    """
    try:
        proba = estimator.predict_proba(X)
        classes = getattr(estimator, "classes_", None)
        if proba.shape[1] == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        return float(roc_auc_score(y, proba, multi_class="ovr", average="weighted", labels=classes))
    except Exception:
        return np.nan


REGRESSION_METRICS = {
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "mape": "neg_mean_absolute_percentage_error",
}
REGRESSION_PRIMARY = "r2"

CLASSIFICATION_METRICS = {
    "accuracy": "accuracy",
    # "macro" averaging, not "weighted": weighted averaging is dominated by
    # the majority class's support, so on real imbalanced data (fraud,
    # churn, etc.) a baseline that NEVER predicts the minority class can
    # still score ~0.88 weighted-F1 while catching 0% of what actually
    # matters. Macro treats every class equally regardless of size, so a
    # model that ignores the minority class gets correctly penalized
    # (verified: same baseline scores ~0.48 macro-F1 — near chance-level
    # for the class that matters — instead of a deceptively strong 0.88).
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro",
    "f2": make_scorer(fbeta_score, beta=2, average="macro", zero_division=0),
    # a plain scorer string here (e.g. "roc_auc_ovr_weighted") crashes the
    # WHOLE entry whenever one CV fold is missing a rare class — see
    # _safe_roc_auc_ovr_weighted's docstring for why this must be a
    # NaN-on-failure callable instead.
    "roc_auc": _safe_roc_auc_ovr_weighted,
}
CLASSIFICATION_PRIMARY = "f1"

# sklearn reports these as negative (it always maximizes); flip sign so the
# UI shows the metric the way a human reads it (lower RMSE is still "better"
# but the number itself should read positive).
NEGATED_METRICS = {"rmse", "mae", "mape"}


# Metrics a user can rank by. Error metrics are ranked as their negative, so a higher score is
# always better; "rmsle" is RMSE on log(1+y), which the runner arranges by training on log(1+y).
SELECTABLE_PRIMARY = {
    TaskType.CLASSIFICATION: ["f1", "accuracy", "roc_auc", "f2", "precision", "recall"],
    TaskType.REGRESSION: ["r2", "rmse", "mae", "rmsle"],
}


def display_score(primary: str, value: float) -> float:
    """A ranking score as a person reads it (RMSE 1.86, not -1.86)."""
    return -value if primary in NEGATED_METRICS or primary == "rmsle" else value


def metrics_for(task_type: TaskType, primary: str | None = None) -> tuple[dict[str, object], str]:
    """`primary` overrides the metric used to rank models (e.g. roc_auc for a
    competition scored on AUC). Only higher-is-better metrics are selectable."""
    if task_type == TaskType.CLASSIFICATION:
        metrics, default = CLASSIFICATION_METRICS, CLASSIFICATION_PRIMARY
    else:
        metrics, default = REGRESSION_METRICS, REGRESSION_PRIMARY
    if primary is not None and primary != default:
        if primary not in SELECTABLE_PRIMARY.get(task_type, []):
            raise ValueError(f"primary metric {primary!r} is not selectable for {task_type.value}")
        return metrics, primary
    return metrics, default


@dataclass
class LeaderboardEntry:
    pipeline_name: str
    model_name: str
    is_baseline: bool
    primary_metric_name: str
    primary_mean: float
    primary_std: float
    metrics: dict[str, float] = field(default_factory=dict)
    fold_scores: list[float] = field(default_factory=list)
    runtime_seconds: float = 0.0
    error: str | None = None
    uses_smote: bool = False
    stopped_early: bool = False   # trailed the best after fold 1 and was not run on the rest

    @property
    def ok(self) -> bool:
        return self.error is None


class ExperimentEngine:
    """
    Runs repeated CV for every (pipeline candidate x model) combination via
    `run_leaderboard()` — the real Stage 4/5 entry point.
    """

    def __init__(self, task_type: TaskType, n_splits: int = 5, n_repeats: int = 1, random_state: int = 42,
                 primary: str | None = None):
        self.task_type = task_type
        self.primary = primary
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def _cv_strategy(self):
        if self.task_type == TaskType.CLASSIFICATION:
            return RepeatedStratifiedKFold(
                n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state
            )
        return RepeatedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state
        )

    def run_leaderboard(
        self,
        candidates: list[PipelineCandidate],
        models: list[ModelSpec],
        X, y,
        rescue: PipelineCandidate | None = None,
        progress=None,
    ) -> list[LeaderboardEntry]:
        """`rescue`: a plain fallback pipeline; any model that crashes on every main pipeline
        is retried on it, and it is appended to `candidates` when used so later stages can
        find it by name. `progress(done, total, label)` is called after each scored fit."""
        scorers, primary = metrics_for(self.task_type, self.primary)
        cv = self._cv_strategy()
        entries: list[LeaderboardEntry] = []

        # Non-fatal but real: StratifiedKFold only warns (doesn't raise) when
        # a class has fewer rows than n_splits — some folds then simply never
        # see that class in their test portion, quietly degrading per-fold
        # reliability for it. That warning is easy to miss in a UI context
        # (Python warnings aren't shown to the user), so surface it as a real
        # note instead of letting it go unseen.
        self.cv_fold_note = ""
        if self.task_type == TaskType.CLASSIFICATION:
            min_class_count = int(pd.Series(y).value_counts().min())
            if 0 < min_class_count < self.n_splits:
                self.cv_fold_note = (
                    f"The smallest class has only {min_class_count} row(s), fewer than the "
                    f"{self.n_splits} CV folds — some folds won't see it in their test portion, "
                    f"so its fold-level metrics are less reliable than other classes'."
                )

        self.rescue_note = ""
        y_arr = np.asarray(y)
        splits = list(cv.split(X, y_arr))
        n_models = len(models)
        total = len(candidates) * n_models
        counter = {"done": 0}
        if progress:
            progress(0, total, "Starting the practice rounds")

        def tick(label_fn):
            def _cb(i):
                counter["done"] += 1
                if progress:
                    progress(min(counter["done"], total), total, label_fn(i))
            return _cb

        def score(name, steps, jobs, on_done=None, prune=False):
            res = cross_validate_jobs(steps, jobs, X, y_arr, splits, scorers, on_done, primary if prune else None)
            return [self._entry(name, j, r, primary, len(splits)) for j, r in zip(jobs, res)]

        for candidate in list(candidates):
            jobs = [ModelJob(m.name, m.build(), m.is_baseline) for m in models]
            entries += score(candidate.name, candidate.pipeline.steps, jobs,
                             tick(lambda i, c=candidate, j=jobs: f"{j[i].model_name} on {prep_label(c.name)}"), prune=True)

        # Second approach for models that crashed on every main pipeline.
        if rescue is not None:
            ok_models = {e.model_name for e in entries if e.ok}
            broken = [m for m in models if m.name not in ok_models and not m.is_baseline]
            if broken:
                new = score(rescue.name, rescue.pipeline.steps, [ModelJob(m.name, m.build()) for m in broken])
                entries += new
                retried = [e.model_name for e in new if e.ok]
                if retried and rescue not in candidates:
                    candidates.append(rescue)
                self.rescue_note = (
                    f"{len(broken)} model(s) failed on every main pipeline and were retried on the plain "
                    f"rescue pipeline; {len(retried)} recovered" + (f" ({', '.join(retried)})." if retried else "."))

        # SMOTE variants: classification only, only when genuinely imbalanced (and enough
        # minority-class rows to synthesize from), scoped to the "balanced" candidate +
        # non-baseline models so the combination count stays bounded. `smote_note` says
        # exactly why SMOTE did or didn't run.
        if self.task_type == TaskType.CLASSIFICATION:
            viable, reason = smote_is_viable(y)
            self.smote_note = reason
            if viable:
                balanced_candidate = next((c for c in candidates if c.name == "balanced"), None)
                if balanced_candidate is not None:
                    specs = [m for m in models if not m.is_baseline]  # resampling a dummy changes nothing
                    jobs = [ModelJob(f"{m.name} + SMOTE",
                                     build_smote_pipeline(balanced_candidate.pipeline.steps, m.build(), self.random_state),
                                     uses_smote=True) for m in specs]
                    entries += score(balanced_candidate.name, None, jobs)
        else:
            self.smote_note = "SMOTE applies to classification tasks only"

        # Accuracy boosters (ensemble of the top model types; tuned threshold for binary
        # targets; log target for skewed ones), scored under the same CV as everything else.
        is_binary = self.task_type == TaskType.CLASSIFICATION and pd.Series(y).nunique() == 2
        by_name = {c.name: c for c in candidates}
        extras = extra_specs_for(entries, self.task_type, is_binary, y, primary, n_rows=len(y))
        for pname in dict.fromkeys(p for p, _ in extras):
            specs = [s for p, s in extras if p == pname]
            entries += score(pname, by_name[pname].pipeline.steps, [ModelJob(s.name, s.build()) for s in specs])

        entries.sort(key=lambda e: (not e.ok, -e.primary_mean if e.ok else 0.0))
        return entries

    def _entry(self, pipeline_name: str, job: ModelJob, result, primary: str, n_folds: int) -> LeaderboardEntry:
        folds, info = result
        if folds is None:  # surfaced in the leaderboard row, not hidden
            return LeaderboardEntry(
                pipeline_name=pipeline_name, model_name=job.model_name,
                is_baseline=job.is_baseline, primary_metric_name=primary,
                primary_mean=float("nan"), primary_std=float("nan"),
                error=info, uses_smote=job.uses_smote,
            )

        metrics_mean = {}
        for key in folds[0]:
            vals = np.array([f[key] for f in folds], dtype=float)
            if key in NEGATED_METRICS:
                vals = -vals
            # nanmean: a scorer like roc_auc can legitimately return NaN for a fold missing a
            # rare class (see _safe_roc_auc_ovr_weighted); excluding just those folds from that
            # one metric is correct, while a plain mean would poison it back to NaN.
            with np.errstate(invalid="ignore"):
                metrics_mean[key] = float(np.nanmean(vals)) if np.any(~np.isnan(vals)) else float("nan")

        # The ranking score is always "higher is better" (an error metric is kept negative).
        primary_vals = np.array([f[primary] for f in folds], dtype=float)

        return LeaderboardEntry(
            pipeline_name=pipeline_name, model_name=job.model_name,
            is_baseline=job.is_baseline, primary_metric_name=primary,
            primary_mean=float(np.mean(primary_vals)),
            primary_std=float(np.std(primary_vals)),
            metrics=metrics_mean,
            fold_scores=primary_vals.tolist(),
            runtime_seconds=float(info),
            uses_smote=job.uses_smote, stopped_early=len(folds) < n_folds,
        )
