"""Parallel cross-validation: every model x fold fit runs at once, preprocessing runs once.

The old engine called `cross_validate` once per (pipeline, model): at most `cv` fits were
alive at a time, and the same preprocessing was refit for every model (11 models x 5 folds =
55 identical refits per pipeline). Here, for one pipeline:

1. `_prep_fold` fits the preprocessing on each training fold once and stores the transformed
   train/test arrays in a temporary folder (memory-mapped by the workers, never held twice).
2. `_fit_fold` fits ONE model on ONE stored fold. All model x fold tasks go to a process pool
   together, heaviest first, so every core stays busy until the last task.

The numbers are the ones `cross_validate` would give: the preprocessing is fitted on the
training fold only (`fit_transform`, exactly as Pipeline does) and applied to the test fold.
"""
from __future__ import annotations

import os
import shutil
import warnings
import tempfile
import time
from dataclasses import dataclass

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.pipeline import Pipeline


def worker_count(n_tasks: int) -> int:
    """Cores to use. DATAPILOT_JOBS overrides (1 = fully sequential)."""
    env = os.environ.get("DATAPILOT_JOBS", "").strip()
    cores = int(env) if env.isdigit() and int(env) > 0 else (os.cpu_count() or 1)
    return max(1, min(cores, n_tasks))


# Rough relative cost of one fit, used only to start the slowest tasks first.
_COST = {"CatBoost": 9, "Neural Net": 8, "Random Forest": 7, "Extra Trees": 7, "KNN": 6, "AdaBoost": 5,
         "XGBoost": 5, "LightGBM": 3, "Hist Gradient": 3, "Decision Tree": 1, "Linear": 1, "Ridge": 1,
         "Elastic": 1, "Logistic": 2, "Baseline": 0}


def fit_cost(model_name: str) -> float:
    cost = sum(v for k, v in _COST.items() if k in model_name) or 4
    if model_name.startswith("Stacking"):
        cost *= 4
    if model_name.endswith("+ threshold"):
        cost *= 4
    return cost


@dataclass
class ModelJob:
    """One thing to cross-validate: a built model, or (steps=None) a ready full pipeline."""
    model_name: str
    estimator: object
    is_baseline: bool = False
    uses_smote: bool = False


_DATA: dict = {}   # per-worker memo of the full X, y (only for uncached full-pipeline jobs)


def _load_data(path):
    if path not in _DATA:
        _DATA.clear()
        _DATA[path] = joblib.load(path)
    return _DATA[path]


def _score(est, Xte, yte, scorers: dict) -> dict:
    out = {}
    for key, sc in scorers.items():
        fn = get_scorer(sc) if isinstance(sc, str) else sc
        out[key] = float(fn(est, Xte, yte))
    return out


def _prep_fold(steps, X, y, tr, te, path):
    warnings.simplefilter("ignore")
    try:
        pre = clone(Pipeline(list(steps)))
        t = time.perf_counter()
        Xt = pre.fit_transform(X.iloc[tr], y[tr])
        Xe = pre.transform(X.iloc[te])
        joblib.dump((Xt, Xe, y[tr], y[te]), path)
        return None, time.perf_counter() - t
    except Exception as e:  # noqa: BLE001 - reported on every model of this pipeline
        return f"{type(e).__name__}: {e}", 0.0


def _fit_fold(idx, fold, est, path, scorers, data_path=None, tr=None, te=None):
    warnings.simplefilter("ignore")
    try:
        t = time.perf_counter()
        est = clone(est)
        if data_path is None:
            Xt, Xe, yt, ye = joblib.load(path, mmap_mode="r")
        else:  # a full pipeline (e.g. SMOTE) fitted on the raw rows of this fold
            X, y = _load_data(data_path)
            Xt, yt, Xe, ye = X.iloc[tr], y[tr], X.iloc[te], y[te]
        est.fit(Xt, yt)
        fit = time.perf_counter() - t
        t = time.perf_counter()
        sc = _score(est, Xe, ye, scorers)
        return idx, fold, sc, fit + (time.perf_counter() - t), None
    except Exception as e:  # noqa: BLE001
        return idx, fold, None, 0.0, f"{type(e).__name__}: {e}"


def _run(tasks, workers: int, ordered: bool):
    """Run delayed tasks on the pool; if the pool itself breaks (memory, pickling), redo in-process once."""
    try:
        return list(Parallel(n_jobs=workers, backend="loky",
                             return_as="list" if ordered else "generator_unordered")(tasks))
    except Exception:  # noqa: BLE001 - infrastructure failure, not a model error (those are returned)
        return list(Parallel(n_jobs=1)(tasks))


PRUNE_MIN_COST = 5      # only fits at least this expensive are ever stopped early
PRUNE_MARGIN = 0.02     # ...and only when fold 1 trails the best by more than 2% of its score
PRUNE_MIN_ROWS = 5000   # on small data every model is cheap, so every model gets every fold


def cross_validate_jobs(steps, jobs: list[ModelJob], X, y, splits, scorers: dict, on_done=None,
                        primary: str | None = None):
    """Cross-validate every job over `splits`. `steps` are the unfitted preprocessing steps shared by
    all jobs (None: each job's estimator is already a full pipeline). Returns, per job in order,
    (list of per-fold score dicts, seconds) or (None, error text). `on_done(job_index)` fires as each
    job completes.

    With `primary` given and enough rows, fold 1 is scored for every model first; an expensive model
    (forests, neural nets) that trails the best by more than PRUNE_MARGIN is not run on the other
    folds: its single-fold score still ranks it, and the time goes to models that can win."""
    n_folds = len(splits)
    results: list = [None] * len(jobs)
    X = X.reset_index(drop=True)
    y = np.asarray(y)
    tmp = tempfile.mkdtemp(prefix="datapilot_folds_")
    try:
        data_path = None
        paths = [os.path.join(tmp, f"fold{i}.pkl") for i in range(n_folds)]
        if steps is not None:
            prep = _run([delayed(_prep_fold)(steps, X, y, tr, te, p) for (tr, te), p in zip(splits, paths)],
                        worker_count(n_folds), ordered=True)
            failed = next((err for err, _ in prep if err), None)
            if failed:
                return [(None, failed)] * len(jobs)
            prep_secs = sum(s for _, s in prep) / max(len(jobs), 1)
        else:
            prep_secs = 0.0
            data_path = os.path.join(tmp, "data.pkl")
            joblib.dump((X, y), data_path)

        got: dict[int, dict] = {i: {} for i in range(len(jobs))}
        errs: dict[int, str] = {}

        def run(pairs, need):
            """Fit the (job, fold) pairs, heaviest first; finish a job once it has `need` folds."""
            pairs = sorted(pairs, key=lambda p: (-fit_cost(jobs[p[0]].model_name), p[1]))
            tasks = [delayed(_fit_fold)(i, f, jobs[i].estimator, paths[f], scorers, data_path, splits[f][0], splits[f][1])
                     for i, f in pairs]
            for idx, fold, sc, secs, err in _run(tasks, worker_count(len(tasks)), ordered=False):
                if err:
                    errs.setdefault(idx, err)
                else:
                    got[idx][fold] = (sc, secs)
                if results[idx] is None and (idx in errs or len(got[idx]) >= need[idx]):
                    finish(idx)

        def finish(idx):
            folds = sorted(got[idx])
            results[idx] = ((None, errs[idx]) if idx in errs else
                            ([got[idx][f][0] for f in folds], sum(got[idx][f][1] for f in folds) + prep_secs))
            if on_done:
                on_done(idx)

        everyone = range(len(jobs))
        if primary and len(X) >= PRUNE_MIN_ROWS and n_folds > 1:
            run([(i, 0) for i in everyone], {i: n_folds + 1 for i in everyone})   # nobody finishes on fold 1 alone
            scores = {i: got[i][0][0][primary] for i in everyone if 0 in got[i] and not np.isnan(got[i][0][0][primary])}
            best = max(scores.values(), default=0.0)
            slack = PRUNE_MARGIN * max(abs(best), 1e-9)
            stop = {i for i, v in scores.items() if v < best - slack and fit_cost(jobs[i].model_name) >= PRUNE_MIN_COST
                    and not jobs[i].is_baseline}
            rest = [i for i in everyone if i not in stop and i not in errs]
            for i in stop:
                finish(i)
            run([(i, f) for i in rest for f in range(1, n_folds)], {i: n_folds for i in everyone})
        else:
            run([(i, f) for i in everyone for f in range(n_folds)], {i: n_folds for i in everyone})
        return results
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
