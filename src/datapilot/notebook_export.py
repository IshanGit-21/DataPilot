"""One readable script for the whole run.

`generate_train_script` writes `train.py` as a single file in the order people normally
write a model: imports, settings, load, clean and split, features, model, train, tune,
evaluate, save. The feature section is the real preprocessing of the winning pipeline
written out as scikit-learn code, so editing it changes what the model learns.
Sections start with `# %%`, which VS Code shows as runnable cells."""
from __future__ import annotations

import importlib
import re

from sklearn.pipeline import Pipeline

_REGISTRY = ["sklearn.pipeline", "sklearn.compose", "sklearn.impute", "sklearn.preprocessing",
             "sklearn.decomposition", "sklearn.feature_selection", "sklearn.model_selection",
             "sklearn.feature_extraction.text", "datapilot.transformers", "datapilot.feature_engineering",
             "datapilot.cleaning"]


def _walk(obj, found: dict, depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _walk(item, found, depth + 1)
        return
    if hasattr(obj, "get_params"):
        found[type(obj).__name__] = type(obj)
        try:
            params = obj.get_params(deep=False)
        except Exception:
            return
        for v in params.values():
            _walk(v, found, depth + 1)


def _public_module(cls) -> str:
    mod = cls.__module__
    if mod.startswith("datapilot"):
        return mod
    parts = mod.split(".")
    while len(parts) > 1 and (parts[-1].startswith("_") or not hasattr(importlib.import_module(".".join(parts[:-1])), cls.__name__)):
        if not parts[-1].startswith("_") and not hasattr(importlib.import_module(".".join(parts[:-1])), cls.__name__):
            break
        parts.pop()
    return ".".join(parts)


def _imports_for(source: str, found: dict) -> list[str]:
    lines: dict[str, set[str]] = {}
    for name in sorted(set(re.findall(r"\b([A-Z][A-Za-z0-9]+)\(", source))):
        cls = found.get(name)
        if cls is None:
            for modname in _REGISTRY:
                try:
                    cls = getattr(importlib.import_module(modname), name, None)
                except Exception:
                    cls = None
                if cls is not None:
                    break
        if cls is None:
            raise ValueError(f"cannot place import for {name}")
        lines.setdefault(_public_module(cls), set()).add(name)
    return [f"from {m} import {', '.join(sorted(n))}" for m, n in sorted(lines.items())]


def feature_source(fitted_pipeline) -> tuple[str, list[str]]:
    """(source of the preprocessing Pipeline, import lines it needs)."""
    steps = [(n, s) for n, s in fitted_pipeline.steps if n not in ("model", "smote")]
    pre = Pipeline(steps)
    found: dict = {}
    _walk(pre, found)
    src = pre.__repr__(N_CHAR_MAX=10**9)
    src = re.sub(r"<TaskType\.(\w+): '[^']*'>", r"TaskType.\1", src)
    imports = [i for i in _imports_for(src, found) if i != "from sklearn.pipeline import Pipeline"]
    return src, imports


_TEMPLATE = '''"""
DataPilot project: one script, top to bottom.

    python train.py your_data.csv

Task: {task} on "{target}"     Best model: {model}{smote_note}   (data preparation: "{pipeline}")

Each "# %%" block is a step you can edit and rerun (VS Code shows them as cells).
Change SETTINGS, the FEATURES block or the MODEL block, then run again.
"""

# %% 1. Imports
import sys

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
{imports}
from datapilot.io import read_any
from datapilot.finalize import cross_check
from datapilot import (
    TaskType, build_smote_pipeline, evaluate_on_held_out_test, log_target_model, resolve_model_spec,
    train_test_split_data, use_all_cores,
)

# %% 2. Settings
TARGET = {target!r}
TASK = TaskType({task!r})
EXCLUDED_COLUMNS = {excluded!r}  # columns left out of the inputs
TEST_SIZE = 0.2                      # share of rows kept aside for the final test (when there is no TEST_FILE)
TEST_FILE = {test_file!r}            # your own test file, as added in the app; None keeps TEST_SIZE of the rows aside
RANDOM_STATE = 42
MODEL = {model!r}
MODEL_PARAMS = {params!r}  # tuned values; edit freely
USE_SMOTE = {smote!r}  # rebalance classes inside training only
LOG_TARGET = {log_target!r}  # learn log(1+target) and convert back (for an RMSLE-style score)
RUN_TUNING = False                   # True: search for better MODEL_PARAMS (slower)
CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "data.csv"

# %% 3. Load data
df = read_any(CSV_PATH)   # a .csv or an Excel file (first sheet)
print(f"{{len(df):,}} rows, {{df.shape[1]}} columns")

# %% 4. Clean and split
# Drops exact duplicate rows and rows with no target, fixes messy target labels,
# then keeps TEST_SIZE of the rows aside (or uses TEST_FILE). The test rows are used once, at step 9.
# A TEST_FILE without answers keeps no rows aside: step 9 then cross-checks the model on the training rows.
df_model = df.drop(columns=[c for c in EXCLUDED_COLUMNS if c in df.columns])
test_df = read_any(TEST_FILE) if TEST_FILE else None
split = train_test_split_data(df_model, TARGET, TASK, test_size=TEST_SIZE, random_state=RANDOM_STATE, test_df=test_df)
X_train, X_test, y_train, y_test = split.X_train, split.X_test, split.y_train, split.y_test
print(f"train {{len(X_train):,}} rows, test {{len(X_test):,}} rows")

# %% 5. Features (cleaning, missing values, encoding, scaling, new columns)
# Fitted on training rows only, so nothing from the test rows leaks in.
preprocess = {features}

# %% 6. Model
estimator = use_all_cores(resolve_model_spec(MODEL, TASK).build())
if MODEL_PARAMS:
    estimator.set_params(**MODEL_PARAMS)
if LOG_TARGET:
    estimator = log_target_model(estimator)
if USE_SMOTE:
    pipe = build_smote_pipeline(preprocess.steps, estimator)
else:
    pipe = Pipeline(preprocess.steps + [("model", estimator)])

# %% 7. Tune (optional)
if RUN_TUNING:
    from sklearn.model_selection import RandomizedSearchCV
    from datapilot.tuning import PARAM_DISTRIBUTIONS
    space = PARAM_DISTRIBUTIONS.get(MODEL.removesuffix(" + SMOTE"))
    if space and LOG_TARGET:
        space = {{k.replace("model__", "model__regressor__"): v for k, v in space.items()}}
    if space:
        search = RandomizedSearchCV(pipe, space, n_iter=20, cv=3, random_state=RANDOM_STATE, n_jobs=-1,
                                    scoring="f1_macro" if TASK == TaskType.CLASSIFICATION else "r2")
        search.fit(X_train, y_train)
        pipe = search.best_estimator_
        print("best settings:", search.best_params_)
    else:
        print("No search space for this model; skipping.")

# %% 8. Train
if not RUN_TUNING:
    pipe.fit(X_train, y_train)

# %% 9. Evaluate on the held-out rows
if split.score_mode == "cv":   # no rows were hidden: every row predicted by a model that did not see it
    result = cross_check(preprocess.steps, "{pipeline}", MODEL, USE_SMOTE, MODEL_PARAMS, TASK, X_test, y_test,
                         LOG_TARGET, split.label_encoder)
else:
    result = evaluate_on_held_out_test("{pipeline}", MODEL, pipe, X_test, y_test, TASK,
                                       label_encoder=split.label_encoder)
if result.error:
    raise SystemExit(f"Evaluation failed: {{result.error}}")
metrics = result.regression or result.classification
print(f"Held-out rows: {{result.n_test}}")
for name, value in vars(metrics).items():
    if isinstance(value, (int, float)) or value is None:
        print(f"  {{name}}: {{value}}")

# %% 10. Save
joblib.dump(pipe, "model.joblib")
joblib.dump(split.label_encoder, "label_encoder.joblib")
print("Saved model.joblib and label_encoder.joblib (use predict.py for new rows)")
'''


def generate_train_script(*, fitted_pipeline, target: str, task: str, pipeline_name: str, model_name: str,
                          best_params: dict, uses_smote: bool, excluded: list[str], log_target: bool = False,
                          test_file: str | None = None) -> str:
    features, imports = feature_source(fitted_pipeline)
    indented = features.replace("\n", "\n" + " " * 0)
    return _TEMPLATE.format(
        task=task, target=target, pipeline=pipeline_name, model=model_name,
        smote_note=" with SMOTE" if uses_smote else "", imports="\n".join(imports),
        excluded=list(excluded), params=dict(best_params or {}), smote=uses_smote, features=indented, log_target=log_target,
        test_file=test_file,
    )


_SUBMISSION_TEMPLATE = '''"""
Make the competition file: train on ALL your rows, predict the test file.

    python make_submission.py your_train.csv

Needs test.csv (and sample_submission.csv if you have it) next to this script.
Writes submission.csv. The model and features are the same as train.py; edit the
FEATURES or MODEL blocks to try something else.
"""

# %% 1. Imports
import sys

import pandas as pd
from sklearn.pipeline import Pipeline
{imports}
from datapilot.io import read_any
from datapilot import TaskType, build_smote_pipeline, log_target_model, resolve_model_spec, train_test_split_data, use_all_cores
from datapilot.submission import check_submission, make_submission

# %% 2. Settings
TARGET = {target!r}
TASK = TaskType({task!r})
EXCLUDED_COLUMNS = {excluded!r}
FEATURE_COLUMNS = {features_list!r}
MODEL = {model!r}
MODEL_PARAMS = {params!r}
USE_SMOTE = {smote!r}
LOG_TARGET = {log_target!r}
TRAIN_CSV = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
TEST_CSV = "test.csv"
SAMPLE_CSV = "sample_submission.csv"   # optional
RANDOM_STATE = 42

# %% 3. Load
train = read_any(TRAIN_CSV)
test = read_any(TEST_CSV)
try:
    sample = read_any(SAMPLE_CSV)
except FileNotFoundError:
    sample = None

# %% 4. Clean and use every row for training
df_model = train.drop(columns=[c for c in EXCLUDED_COLUMNS if c in train.columns])
split = train_test_split_data(df_model, TARGET, TASK, test_size=0.2, random_state=RANDOM_STATE)
X = pd.concat([split.X_train, split.X_test])
y = pd.concat([split.y_train, split.y_test])
print(f"training on {{len(X):,}} rows")

# %% 5. Features
preprocess = {features}

# %% 6. Model and fit
estimator = use_all_cores(resolve_model_spec(MODEL, TASK).build())
if MODEL_PARAMS:
    estimator.set_params(**MODEL_PARAMS)
if LOG_TARGET:
    estimator = log_target_model(estimator)
if USE_SMOTE:
    pipe = build_smote_pipeline(preprocess.steps, estimator)
else:
    pipe = Pipeline(preprocess.steps + [("model", estimator)])
pipe.fit(X, y)

# %% 7. Predict and write submission.csv
submission, note = make_submission(pipe, test, sample, FEATURE_COLUMNS, split.label_encoder)
print(note)
for level, message in check_submission(submission, sample, test):
    print(f"[{{level}}] {{message}}")
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
'''


def generate_submission_script(*, fitted_pipeline, target: str, task: str, model_name: str, best_params: dict,
                               uses_smote: bool, excluded: list[str], feature_columns: list[str], log_target: bool = False) -> str:
    features, imports = feature_source(fitted_pipeline)
    return _SUBMISSION_TEMPLATE.format(
        target=target, task=task, model=model_name, imports="\n".join(imports), excluded=list(excluded),
        features_list=list(feature_columns), params=dict(best_params or {}), smote=uses_smote, features=features,
        log_target=log_target)
