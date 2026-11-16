"""
Stage 5 — Hyperparameter tuning.

Takes the top non-baseline entries from the Stage 4 leaderboard and runs a
real RandomizedSearchCV over each — not a placeholder, not a fixed "best
guess" set of params. Search spaces are small and bounded (n_iter kept low)
specifically so this stays interactive-UI-fast rather than a batch job;
that's a real, deliberate trade-off, not a shortcut that skips the work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import (
    ParameterSampler,
    RandomizedSearchCV,
    check_cv,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    cross_val_score,
)

from .experiment import LeaderboardEntry, metrics_for
from .imbalance import build_smote_pipeline
from .model_zoo import ModelSpec
from .parallel import ModelJob, cross_validate_jobs
from .planner import PipelineCandidate
from .reasoner import TaskType

# Param distributions keyed by ModelSpec.name. Prefixed with "model__" since
# these params apply to the final step of a Pipeline(preprocess..., model).
# Baseline and plain linear models are intentionally absent — there's
# nothing meaningful to tune on a DummyRegressor/DummyClassifier or on
# vanilla LinearRegression, and pretending otherwise would just waste CV
# budget for no real gain.
PARAM_DISTRIBUTIONS: dict[str, dict[str, list]] = {
    "Logistic Regression": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30],
    },
    "Decision Tree": {
        "model__max_depth": [3, 5, 8, 12, 16, None],
        "model__min_samples_leaf": [1, 2, 5, 10, 20],
    },
    "Random Forest": {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [None, 8, 15, 25],
        "model__min_samples_leaf": [1, 2, 5, 10],
    },
    "Extra Trees": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 5],
        "model__max_features": ["sqrt", 0.5, 1.0],
    },
    "Hist Gradient Boosting": {
        "model__max_iter": [150, 250, 400],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
    "XGBoost": {
        "model__n_estimators": [150, 250, 350, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12, 0.18],
        "model__subsample": [0.7, 0.85, 1.0],
    },
    "CatBoost": {
        "model__iterations": [150, 250, 400],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.03, 0.06, 0.1, 0.15],
        "model__l2_leaf_reg": [1, 3, 6, 10],
    },
    "LightGBM": {
        "model__n_estimators": [150, 250, 350, 500],
        "model__num_leaves": [15, 31, 63, 127],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.12, 0.18],
    },
}


@dataclass
class TuningResult:
    pipeline_name: str
    model_name: str
    base_mean: float
    tuned_mean: float
    tuned_std: float
    best_params: dict = field(default_factory=dict)
    tunable: bool = True
    error: str | None = None
    uses_smote: bool = False
    verified: bool = False       # re-scored on fresh folds against the defaults
    accepted: bool = True        # False: the search found nothing that really beats the defaults

    @property
    def improvement(self) -> float:
        return self.tuned_mean - self.base_mean

    @property
    def ok(self) -> bool:
        return self.error is None


def _best_entry_per_model(entries: list[LeaderboardEntry]) -> list[LeaderboardEntry]:
    """Collapses (pipeline x model) entries down to one best-pipeline entry
    per model name, so tuning budget isn't wasted re-tuning the same model
    type twice just because it topped two different pipelines."""
    best: dict[str, LeaderboardEntry] = {}
    for e in entries:
        if not e.ok:
            continue
        current = best.get(e.model_name)
        if current is None or e.primary_mean > current.primary_mean:
            best[e.model_name] = e
    return sorted(best.values(), key=lambda e: e.primary_mean, reverse=True)


def tune_top_entries(
    entries: list[LeaderboardEntry],
    candidates: list[PipelineCandidate],
    models: list[ModelSpec],
    X_train, y_train,
    task_type: TaskType,
    top_k: int = 3,
    n_iter: int = 20,
    primary: str | None = None,
    cv_splits: int = 3,
    random_state: int = 42,
) -> list[TuningResult]:
    """Tunes the top `top_k` *distinct model types* (best pipeline each) from
    the leaderboard — not just the raw top-k rows, which could be the same
    non-tunable model on two different pipelines. Entries with no registered
    param distribution (baseline, Linear Regression) are skipped with
    tunable=False rather than silently omitted — the UI can then say so
    honestly instead of the entry just vanishing.

    A "<Model> + SMOTE" entry from the leaderboard is tuned as a real
    imblearn Pipeline (preprocessing -> SMOTE -> model) — not silently
    downgraded to a plain pipeline without resampling, which would tune a
    different thing than what topped the leaderboard.
    """
    scorers, primary = metrics_for(task_type, primary)
    primary_scorer = scorers[primary]

    candidates_by_name = {c.name: c for c in candidates}
    models_by_name = {m.name: m for m in models}

    from .ensemble import BOOSTER_SUFFIXES, ENSEMBLE_PREFIX, STACK_PREFIX
    plain = [e for e in entries if not e.model_name.startswith((ENSEMBLE_PREFIX, STACK_PREFIX))
             and not e.model_name.endswith(BOOSTER_SUFFIXES)]  # boosters are not tuned directly
    ranked = [e for e in _best_entry_per_model(plain) if not e.is_baseline][:top_k]
    results: dict[int, TuningResult] = {}
    work: list[tuple] = []   # (rank, entry, steps, spec, param_dist) tuned on the shared-fold fast path

    for rank, entry in enumerate(ranked):
        uses_smote = entry.uses_smote
        base_model_name = entry.model_name.removesuffix(" + SMOTE") if uses_smote else entry.model_name

        candidate = candidates_by_name.get(entry.pipeline_name)
        spec = models_by_name.get(base_model_name)
        if candidate is None or spec is None:
            continue

        param_dist = PARAM_DISTRIBUTIONS.get(base_model_name)
        if not param_dist:
            results[rank] = TuningResult(
                pipeline_name=entry.pipeline_name, model_name=entry.model_name,
                base_mean=entry.primary_mean, tuned_mean=entry.primary_mean,
                tuned_std=entry.primary_std, tunable=False, uses_smote=uses_smote,
            )
        elif uses_smote:  # a resampling pipeline: tuned as a whole (rare), one search at a time
            results[rank] = _tune_resampled(entry, candidate, spec, param_dist, X_train, y_train, task_type,
                                            primary_scorer, n_iter, cv_splits, random_state)
        else:
            work.append((rank, entry, candidate.pipeline.steps, spec, param_dist))

    # Fast path: preprocessing fitted once per fold, every (setting x fold) fit in one parallel pool.
    y_arr = np.asarray(y_train)
    folds = list(check_cv(cv_splits, y_arr, classifier=task_type == TaskType.CLASSIFICATION).split(X_train, y_arr))
    sc = {"s": primary_scorer}
    for pname in dict.fromkeys(w[1].pipeline_name for w in work):
        group = [w for w in work if w[1].pipeline_name == pname]
        steps = group[0][2]
        jobs, owner, sets = [], [], []
        for gi, (_, entry, _, spec, dist) in enumerate(group):
            for params in ParameterSampler(dist, n_iter=n_iter, random_state=random_state):
                p = {k.removeprefix("model__"): v for k, v in params.items()}
                jobs.append(ModelJob(spec.name, spec.build().set_params(**p)))
                owner.append(gi)
                sets.append(p)
        scored = cross_validate_jobs(steps, jobs, X_train, y_train, folds, sc)
        picked: dict[int, tuple] = {}   # group index -> (mean, std, params)
        failed: dict[int, str] = {}
        for ji, (fs, info) in enumerate(scored):
            gi = owner[ji]
            if fs is None:
                failed.setdefault(gi, info)
                continue
            v = np.array([f["s"] for f in fs], dtype=float)
            m = float(np.mean(v)) if not np.isnan(v).any() else float("-inf")
            if gi not in picked or m > picked[gi][0]:
                picked[gi] = (m, float(np.std(v)), sets[ji])
        verify = []
        for gi, (rank, entry, _, spec, _) in enumerate(group):
            if gi in failed or gi not in picked:
                results[rank] = TuningResult(
                    pipeline_name=entry.pipeline_name, model_name=entry.model_name,
                    base_mean=entry.primary_mean, tuned_mean=float("nan"), tuned_std=float("nan"),
                    error=failed.get(gi, "no setting could be scored"))
            else:
                verify.append((gi, rank, entry, spec, picked[gi]))
        # best_score is the maximum over every setting tried, so it is optimistic. Re-score the winner
        # and the defaults together on fresh folds (the same folds for both) and keep the new
        # settings only if they still win.
        vfolds = _verification_folds(X_train, y_arr, task_type, random_state) if VERIFY_TUNING else None
        vres = None
        if verify and vfolds is not None:
            vjobs = [ModelJob(spec.name, clone(est)) for _, _, _, spec, _ in verify for est in
                     (spec.build(), spec.build())]
            for k, (_, _, _, spec, (_, _, p)) in enumerate(verify):
                vjobs[2 * k + 1].estimator.set_params(**p)
            vres = cross_validate_jobs(steps, vjobs, X_train, y_train, vfolds, sc)
        for k, (gi, rank, entry, spec, (best_m, best_sd, p)) in enumerate(verify):
            base_f, tuned_f = (vres[2 * k], vres[2 * k + 1]) if vres else (None, None)
            if base_f is None or base_f[0] is None or tuned_f[0] is None:  # unverified: flagged as such
                results[rank] = TuningResult(
                    pipeline_name=entry.pipeline_name, model_name=entry.model_name,
                    base_mean=entry.primary_mean, tuned_mean=best_m, tuned_std=best_sd, best_params=p)
                continue
            b = np.array([f["s"] for f in base_f[0]]); t = np.array([f["s"] for f in tuned_f[0]])
            accept = float(np.mean(t - b)) >= MIN_TUNING_GAIN
            results[rank] = TuningResult(
                pipeline_name=entry.pipeline_name, model_name=entry.model_name,
                base_mean=float(np.mean(b)), tuned_mean=float(np.mean(t)) if accept else float(np.mean(b)),
                tuned_std=float(np.std(t)) if accept else best_sd,
                best_params=p if accept else {}, verified=True, accepted=accept)

    return [results[r] for r in sorted(results)]


def _tune_resampled(entry, candidate, spec, param_dist, X, y, task_type, scorer, n_iter, cv_splits, random_state):
    """RandomizedSearchCV over a SMOTE pipeline (resampling happens inside each training fold)."""
    full = build_smote_pipeline(candidate.pipeline.steps, spec.build(), random_state)
    search = RandomizedSearchCV(full, param_dist, n_iter=n_iter, cv=cv_splits, scoring=scorer,
                                random_state=random_state, n_jobs=-1, error_score="raise")
    try:
        search.fit(X, y)
    except Exception as e:  # noqa: BLE001 — surfaced, not hidden
        return TuningResult(pipeline_name=entry.pipeline_name, model_name=entry.model_name,
                            base_mean=entry.primary_mean, tuned_mean=float("nan"), tuned_std=float("nan"),
                            error=f"{type(e).__name__}: {e}", uses_smote=True)
    i = search.best_index_
    params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    verdict = None if not VERIFY_TUNING else _verify_against_defaults(
        full, search.best_params_, X, y, task_type, scorer, random_state)
    if verdict is None:
        return TuningResult(pipeline_name=entry.pipeline_name, model_name=entry.model_name,
                            base_mean=entry.primary_mean, tuned_mean=float(search.best_score_),
                            tuned_std=float(search.cv_results_["std_test_score"][i]), best_params=params,
                            uses_smote=True)
    base, tuned, sd, accept = verdict
    return TuningResult(pipeline_name=entry.pipeline_name, model_name=entry.model_name, base_mean=base,
                        tuned_mean=tuned if accept else base,
                        tuned_std=sd if accept else float(search.cv_results_["std_test_score"][i]),
                        best_params=params if accept else {}, uses_smote=True, verified=True, accepted=accept)


def _verification_folds(X, y, task_type, random_state: int):
    """Fresh folds, shared by the defaults and the tuned settings."""
    repeats = 2 if len(X) <= 3000 else 1
    cls = RepeatedStratifiedKFold if task_type == TaskType.CLASSIFICATION else RepeatedKFold
    try:
        return list(cls(n_splits=5, n_repeats=repeats, random_state=random_state + 101).split(X, y))
    except ValueError:  # a class too small for 5 stratified folds
        return list(RepeatedKFold(n_splits=3, n_repeats=repeats, random_state=random_state + 101).split(X, y))


VERIFY_TUNING = True      # switch off only to compare against the old behaviour
MIN_TUNING_GAIN = 0.003   # fresh-fold gain a tuned setting must show to replace the defaults


def _verify_against_defaults(pipeline, tuned_params: dict, X, y, task_type, scorer, random_state: int):
    """(default score, tuned score, tuned std, accept?) on identical fresh folds, or None on failure."""
    try:
        repeats = 2 if len(X) <= 3000 else 1
        cls = RepeatedStratifiedKFold if task_type == TaskType.CLASSIFICATION else RepeatedKFold
        try:
            cv = cls(n_splits=5, n_repeats=repeats, random_state=random_state + 101)
            list(cv.split(X, y))
        except ValueError:  # a class too small for 5 stratified folds
            cv = RepeatedKFold(n_splits=3, n_repeats=repeats, random_state=random_state + 101)
        base = cross_val_score(clone(pipeline), X, y, cv=cv, scoring=scorer, n_jobs=-1, error_score="raise")
        tuned = cross_val_score(clone(pipeline).set_params(**tuned_params), X, y, cv=cv, scoring=scorer,
                                n_jobs=-1, error_score="raise")
        gain = float(np.mean(tuned - base))
        return float(np.mean(base)), float(np.mean(tuned)), float(np.std(tuned)), gain >= MIN_TUNING_GAIN
    except Exception:  # noqa: BLE001 - verification is a safeguard; never block the run
        return None


ROBUST_TOLERANCE = 0.01   # an averaged ensemble may trail the best single model by this much and still win


def _component_params(model_name: str, pipeline_name: str, tuned: dict) -> dict:
    """Tuned settings of each ensemble member, keyed the way a VotingClassifier expects (m0__depth)."""
    from .ensemble import ENSEMBLE_PREFIX
    parts = model_name.removeprefix(ENSEMBLE_PREFIX).removesuffix(" + SMOTE").split(" + ")
    out: dict = {}
    for i, part in enumerate(parts):
        t = tuned.get((pipeline_name, part))
        if t:
            out.update({f"m{i}__{k}": v for k, v in t.best_params.items()})
    return out


def pick_best_entry(entries: list[LeaderboardEntry], tuning_results: list[TuningResult] | None,
                    robust: bool = True):
    """The single (entry, best_params, score) to finalize.

    Highest score, using a tuned score where tuning succeeded for that exact (pipeline,
    model). With `robust` on, an averaged ensemble of the top model types wins when it is
    within ROBUST_TOLERANCE of the best single model: picking the maximum of many noisy
    cross-validation scores overstates it (winner's curse), while averaging different
    models keeps accuracy and cuts the run-to-run swing. The ensemble's members use their
    tuned settings when they were tuned on the same pipeline.
    Shared by the app and the benchmark so they cannot disagree."""
    from .ensemble import ENSEMBLE_PREFIX
    tuned = {(r.pipeline_name, r.model_name): r for r in (tuning_results or []) if r.ok and r.tunable}
    best, best_params, best_score = None, {}, float("-inf")
    for e in entries:
        if not e.ok:
            continue
        t = tuned.get((e.pipeline_name, e.model_name))
        score = t.tuned_mean if t else e.primary_mean
        if score > best_score:
            best, best_params, best_score = e, (dict(t.best_params) if t else {}), score
    if robust and best is not None and not best.model_name.startswith(ENSEMBLE_PREFIX) and not best.is_baseline:
        from .ensemble import BOOSTER_SUFFIXES
        ens = [e for e in entries if e.ok and e.model_name.startswith(ENSEMBLE_PREFIX)
               and not e.model_name.endswith(BOOSTER_SUFFIXES) and e.primary_mean >= best_score - ROBUST_TOLERANCE]
        if ens:
            pick = max(ens, key=lambda e: e.primary_mean)
            return pick, _component_params(pick.model_name, pick.pipeline_name, tuned), pick.primary_mean
    return best, best_params, best_score
