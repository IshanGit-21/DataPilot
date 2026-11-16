# DataPilot — Project Phases & Progress Tracker

Status legend: `[ ]` Not started · `[~]` In progress · `[x]` Done

This file is rewritten each pass to match what's actually in the code —
not aspirational status.

---

## Phase 1 — Frontend: UI/UX (Streamlit, real)
**Status:** Real end-to-end for all 6 stages, including SMOTE-aware leaderboard rows and Stage 3 engineering.

## Phase 2 — Backend: Dataset Profiler (Stage 1)
**Status:** Done, real, tested.

## Phase 3 — Backend: Contextual Question Engine
**Status:** Functionally covered by UI selectors.

## Phase 4 — Backend: Pipeline Planner (Stage 2)
**Status:** Done, real, tested.

## Phase 4b — Feature Engineering (Stage 3) — NOW COMPLETE

- [x] `feature_engineering.py`: `SkewCorrector` (Yeo-Johnson on columns whose
      *training-fold* skew > 1.0) and `TopKInteractions` (up to 3 pairwise
      product features from the numeric columns most associated with the
      target, scored by f_regression/f_classif on the training fold)
- [x] Both decide *what to do* inside `fit()` on the data they receive, not
      from the pre-split Stage 1 profile — so the decision itself is
      leakage-safe, not just the fitted parameters
- [x] Wired into the balanced and aggressive candidates; conservative
      intentionally untouched as the minimal baseline
- [x] Missingness indicators moved to their own `MissingIndicator` branch so
      the engineered numeric branch keeps real feature names
- [x] Decision records document Stage 3 on both candidates; UI text and
      stage rail updated (no more "not built yet")

**Status:** Done, real, tested (6 dedicated tests).

## Phase 4c — Stage 1-6 Robustness Pass (edge-case hardening) — COMPLETE

Before moving to Stage 7, every stage was probed with the inputs real
datasets actually have — string/non-standard class labels, singleton and
rare classes, missing categoricals, all-NaN/constant columns, no-numeric /
no-categorical schemas, odd column names, tiny row counts. 11 scenarios
run end-to-end (Stage 1 through Stage 6) against every combination. Found
and fixed 3 more real bugs (#9, #10, #11 in the bug log below) — one of
which (#11) was silently taking an entire leaderboard from 18/18 working
entries down to 0/18 on nothing more exotic than a 3-row rare class.

**Status:** All 11 probed scenarios now run clean end-to-end. 5 dedicated
regression tests (`test_stage4_edge_cases.py`).

## Phase 5 — Backend: Model Zoo + Experiment Engine (Stage 4)
**Status:** Done, real, tested.

## Phase 5b — Model Improvement (Stage 5) — NOW COMPLETE

- [x] RandomizedSearchCV tuning, deduped by model type
- [x] class_weight='balanced' for all classification models
- [x] **SMOTE-based resampling — real, via imbalanced-learn.** Scoped to the
      "balanced" pipeline candidate + non-baseline models (bounded combo
      count), gated on genuine imbalance + enough minority-class rows to
      synthesize from (`smote_is_viable()`), composed via imblearn's own
      Pipeline so resampling only ever touches training folds — never
      validation/test/predict data. Shown as real, separate leaderboard
      rows ("Model + SMOTE"), tunable, and correctly reconstructed at
      finalize time.
- [x] Error analysis (confusion matrix / residuals) — lives in Stage 6
      since it needs real held-out predictions

**Status:** Stage 5 is now fully done and real — nothing deferred.

## Phase 6 — Final System (Stage 6)
**Status:** Done and real — held-out eval, error analysis, interpretation, export.

## Phase 7 — AI Reasoning Layer
**Status:** Not started.

## Phase 8 — Explainability & Audit Trail
**Status:** Partial — pieces real, not bundled into one exportable report.

## Phase 9 — Benchmarking
**Status:** Not run against the named Kaggle House Prices dataset yet.

## Phase 10 — Polish & Handoff
**Status:** Not started.

---

## Quick status snapshot

| Phase | Area | Status |
|---|---|---|
| 1 | Frontend | Real end-to-end, all 6 stages |
| 2 | Profiler (Stage 1) | Done, real, tested |
| 3 | Question Engine | Covered by UI |
| 4 | Pipeline Planner (Stage 2) | Done, real, tested |
| 4b | Feature Engineering (Stage 3) | **Done, real, tested** |
| 4c | Stage 1-6 robustness pass | **Complete — 11 edge cases, 3 bugs fixed** |
| 5 | Model Zoo + Leaderboard (Stage 4) | Done, real, tested |
| 5b | Tuning + SMOTE (Stage 5) | **Done — nothing deferred** |
| 6 | Final System (Stage 6) | Done, real, tested |
| 7 | AI Reasoning Layer | Not started |
| 8 | Explainability report | Partial (pieces real, not bundled) |
| 9 | Benchmarking | Not run against named dataset yet |
| 10 | Polish & Handoff | Not started |

## Bugs found and fixed across all passes (real-data testing, not toy CSVs)

1. **pandas 3.0 StringDtype**: `dtype == "object"` silently failed on every
   string column under this environment's pandas version. Fixed with
   `pd.api.types` checks throughout.
2. **Identifier over-flagging**: a continuous numeric feature (~100% unique
   purely from measurement precision) was flagged as an identifier and
   silently dropped — every model scored *worse than baseline*. Fixed by
   requiring sequential-integer or id-name evidence.
3. **Missing-target crash**: NaN in the target column crashed every model.
   Fixed by dropping those rows pre-split and reporting the count.
4. **Tuning budget waste**: raw top-K leaderboard rows could mean tuning
   the *same* non-tunable model twice. Fixed by deduping to the best
   pipeline per distinct model type first.
5. **Misleading feature importance**: raw linear coefficients aren't
   comparable across differently-scaled features — a 0/1 flag outranked
   the actual dominant driver. Fixed by scaling coefficients by each
   feature's real training-data spread before ranking.
6. **Target leaking into feature↔feature correlations**: `top_correlations`
   included the target column in its input list, duplicating (and
   mislabeling) what `target_correlations` already reported. Fixed by
   excluding the target before computing feature↔feature pairs. Also
   vectorized the target-correlation computation (one matrix call instead
   of one `.corr()` call per column).
7. **Weighted-average metrics masking real performance on imbalanced data**:
   `precision/recall/f1/f2` used "weighted" averaging, which is dominated
   by the majority class — a baseline that predicted "not fraud" every
   time (0% recall on the class that matters) scored ~0.88 F1, only ~0.02
   below real models, making it look like nothing helped. Switched to
   "macro" averaging (equal weight per class regardless of size); the same
   baseline now correctly scores ~0.48 — near chance-level for the class
   that matters — and real models show an honest, meaningful gap. This
   was found specifically *while verifying SMOTE worked* — the metric bug
   was hiding whether SMOTE/class_weight were doing anything at all.

8. **Column names lost before Stage 3**: `SimpleImputer` returns a bare numpy
   array by default, so `SkewCorrector` saw auto-generated integer column
   names and the pipeline emitted `num__0`, `num__2_x_3` instead of
   `num__factor_a_x_factor_b` — which would have silently broken feature
   interpretation for the balanced pipeline. Caught by the very first
   integration test; fixed with `SimpleImputer(...).set_output(transform="pandas")`.
9. **XGBoost crashed on ordinary label formats**: string labels
   ("yes"/"no") or non-zero-indexed integers (1,2,3) raised inside
   XGBoost's sklearn API specifically — every other model in the zoo
   handles arbitrary labels natively. Fixed centrally, not by
   special-casing XGBoost: `train_test_split_data()` now label-encodes the
   classification target once, consistently across train/test, and the
   fitted encoder is threaded through to the final evaluation so the
   confusion matrix still displays real class names, not 0/1.
10. **Singleton class crashed the entire run**: a class with only 1 row made
    `train_test_split(..., stratify=y)` raise immediately — a hard crash
    before anything downstream could run, not a graceful per-model
    failure. Fixed by checking each class's count first and falling back
    to a non-stratified split when any class has fewer than 2 rows, with
    the fallback reported in `stratify_note` rather than happening silently.
11. **One fragile metric took down an entire leaderboard entry**: `roc_auc`
    raises when a CV fold's validation split doesn't contain every class —
    routine with a rare class spread thin across folds. Because it shared
    one `cross_validate()` call with every other metric and
    `error_score="raise"`, a single bad fold erased accuracy, precision,
    recall, and F1 too, not just roc_auc — with a 3-row rare class, this
    took the WHOLE leaderboard from 18/18 entries down to 0/18. Fixed with
    a self-guarding scorer that returns NaN instead of raising, aggregated
    with `nanmean` so a few bad folds degrade just that one metric instead
    of the whole entry. A separate, real `cv_fold_note` also surfaces when
    a class is smaller than the fold count, since Python warnings aren't
    visible in a UI.

12. **PCA crashed on wide, heavily-categorical real data**: found on the
    actual Ames Housing benchmark, not synthetic data. `OneHotEncoder`
    defaults to sparse output, and `ColumnTransformer` propagates that
    sparsity to the WHOLE combined matrix once categoricals make up enough
    of it (`sparse_threshold=0.3`) — even though the numeric branch
    (StandardScaler, IQRClipper) produced dense output on its own. PCA's
    `svd_solver='arpack'` (auto-selected for sparse input) then crashes
    outright whenever explaining 95% variance needs close to the full
    component count — routine on wide, heavily one-hot-encoded real data
    (81 raw columns, ~288 after encoding, 95% variance needing all 288).
    Fixed with `OneHotEncoder(sparse_output=False)`, keeping the whole
    pipeline consistently dense like every other custom transformer in
    this codebase already is — not a workaround scoped to PCA alone.

13. **Free-text column silently dropped as a false-positive identifier**:
    found via a real Kaggle dataset (NLP with Disaster Tweets) the user
    uploaded. The exact same failure mode as bug #2 (`price`), but on the
    string side: a genuine free-text column (tweet text, ~98.6% unique
    purely from natural-language variety, not identifier-ness) was flagged
    `is_likely_identifier` and dropped entirely — silently discarding the
    single most predictive column in the whole dataset. DataPilot also had
    zero text-handling at all before this, so even fixing the flag alone
    would have left it falling into frequency-encoding (near-zero real
    signal for free text). Fixed with: (1) a real `is_free_text` detector
    (average word count — 14.9 for tweet text vs 1.0-2.1 for short
    categorical columns like `keyword`/`location`, a clean separation),
    excluded from identifier/high-cardinality consideration; (2) a real
    `TextFeaturizer` (scikit-learn's own `TfidfVectorizer`, bounded
    vocabulary) wired into all three candidates, including conservative —
    the minimal baseline shouldn't mean discarding the main signal.
    Verified on the real dataset: F1 jumped from a 0.363 baseline to 0.722
    on the leaderboard, 0.734 on the true held-out test set, with actual
    text terms (`text_tfidf_http`, `text_tfidf_just`) showing up as top
    interpreted features — proof the signal is real, not just present.
    Caught a second gap during this same pass: the `aggressive` candidate
    had TF-IDF wired into its real ColumnTransformer but was missing the
    matching `DecisionRecord` — a silent decision, fixed immediately.

14. **MissingIndicator crashed on rare missingness under normal sampling
    variance**: found on a real, large (699,635-row) Kaggle dataset
    (Airline Satisfaction). A column with very rare missingness (0.04% in
    the real case) can, purely by chance, land zero missing rows in the
    training split while the test split still has one. sklearn's
    `MissingIndicator` defaults to `error_on_new=True` and crashes outright
    in exactly that situation — not a one-off, this is routine for ANY
    rare-missingness column on ANY dataset under ordinary train/test
    splitting or cross-validation, not specific to this competition. Fixed
    with `error_on_new=False`: the indicator now reports "not missing" for
    a feature on rows the model never learned could be missing, rather
    than crashing the whole candidate.

Every one of these was caught by a test written against realistic (messy,
correlated, high-cardinality, imbalanced) synthetic data, OR by running
the actual pipeline against real, named benchmark datasets — not by
inspection of clean toy examples. Each fix has a dedicated regression test.

## Second real-dataset validation — a genuinely different kind of data

The 5-dataset validation above was entirely structured/tabular data. To
test a meaningfully different shape of real-world problem, the user
uploaded the actual Kaggle "NLP with Disaster Tweets" competition data
(train.csv: 7,613 rows, free-text tweets + short categorical columns,
binary target) — a text-classification problem, something DataPilot had
never been run against before. This is exactly how bug #13 above was
found. After the fix, all 6 stages ran clean end-to-end: profiling
correctly separated the free-text column from the short categorical ones,
all 3 candidate pipelines built and fit (102/152/129 features), all 18
leaderboard entries succeeded, tuning ran, held-out evaluation scored
F1=0.734/ROC-AUC=0.79, interpretation surfaced real text terms, and the
exported model round-tripped identically.

## Real-dataset validation pass — COMPLETE

Before calling Stage 1-6 finished, the full pipeline was run end-to-end
(all 6 stages, nothing skipped) against 5 real datasets — not synthetic
generators:

| Dataset | Rows x Cols | Task | Real messiness | Result |
|---|---|---|---|---|
| Breast Cancer (sklearn, real diagnostic data) | 569 x 31 | Binary classification | None (clean) | Held-out F1=0.981, beats baseline by +0.59 |
| Wine (sklearn, real chemistry data) | 178 x 14 | 3-class classification | None (clean) | Held-out F1=0.920, beats baseline by +0.80 |
| Diabetes (sklearn, real progression data) | 442 x 11 | Regression | None (clean) | Held-out R²=0.453 |
| Titanic (real, via GitHub) | 891 x 12 | Binary classification | 866 missing values, mixed types, 3 high-cardinality columns | Held-out F1=0.739, beats baseline by +0.42 |
| **House Prices / Ames (real, via GitHub)** — the actual named Kaggle benchmark from the original project proposal | 1460 x 81 | Regression | 7,829 missing values, 43 categorical columns | **Held-out R²=0.857, RMSE=$33,131, MAPE=11.3%** |

All 5 passed every stage — profiling, all 3 candidate pipelines, the full
model leaderboard (beating baseline on every dataset), tuning, held-out
evaluation, interpretation with real feature names, and an export whose
reloaded predictions matched the original pipeline's bit-for-bit. Bug #12
above was found and fixed during this pass, specifically by the real Ames
Housing data — no synthetic test had wide enough one-hot-encoded
dimensionality to trigger it.

## Universal real-dataset benchmark — 8 datasets, 14 bugs found and fixed

Every dataset below was run through the full, real pipeline — Stage 1
through Stage 6, nothing mocked, nothing skipped. This is the standing
benchmark: any future change to Stages 1-6 should be re-validated against
this table before being considered safe.

| # | Dataset | Rows x Cols | Task | What made it a genuinely different test | Held-out result |
|---|---|---|---|---|---|
| 1 | Breast Cancer (sklearn) | 569 x 31 | Binary classification | Clean baseline | F1=0.981 |
| 2 | Wine (sklearn) | 178 x 14 | 3-class classification | Multi-class | F1=0.920 |
| 3 | Diabetes (sklearn) | 442 x 11 | Regression | Clean baseline | R²=0.453 |
| 4 | Titanic (Kaggle, via GitHub) | 891 x 12 | Binary classification | Real missing values, mixed types | F1=0.739 |
| 5 | **House Prices / Ames** (Kaggle, via GitHub) | 1,460 x 81 | Regression | 7,829 missing values, 43 categoricals, wide one-hot (found bug #12) | **R²=0.857, MAPE=11.3%** |
| 6 | **NLP Disaster Tweets** (Kaggle, user-uploaded) | 7,613 x 5 | Binary classification | Genuine free text (found bugs #13) | F1=0.734, ROC-AUC=0.79 |
| 7 | Bike Sharing Demand (Kaggle, via GitHub) | 10,886 x 10 | Regression | Real datetime strings, target-leakage trap (casual+registered=count) correctly avoided | R²=0.484 |
| 8 | **Airline Satisfaction** (Kaggle, user-uploaded) | 699,635 x 23 | Binary classification | Largest scale yet (700k rows), boolean-dtype target, rare missingness (found bug #14) | **F1=0.900, ROC-AUC=0.946** |

Scale check: Stage 1 profiling ran on the full 699,635-row dataset in 3.1
seconds — confirms profiling itself scales to real production-size data,
not just the samples used for the full CV sweep (a full 700k-row x
18-combination x 5-fold leaderboard sweep was deliberately sampled down
for sandbox runtime, not because anything breaks at full scale).

Every dataset beat its own dummy baseline by a wide, honest margin — the
earlier bugs (#2, #7, #13) are exactly the ones that would have silently
prevented that from being true.

---

## Stage 7 — Guided Setup (done; replaces the dropped AI reasoning layer)
Decision: the LLM narration layer was removed. It added no accuracy, needed an API key and internet, and was not reproducible.
- `src/datapilot/setup.py`: after Stage 1, `suggest_targets()` ranks likely output columns (name hints, class-like cardinality, position) and infers the task type. `suggest_input_exclusions()` pre-unticks constant and identifier-like inputs, with reasons.
- `PipelineOptions` (in `planner.py`): the user can pick which pipelines to build, numeric imputation, skew correction, interactions, outlier clipping, PCA variance, text vocabulary size and missing-value indicators. They can also remove input columns. Defaults are a no-op, overrides are logged as a DecisionRecord, and invalid values raise `ValueError`.
- UI: the output and input pickers sit in "Configure this run". An "Edit these pipelines" panel under the presented pipelines lets the user change settings and rebuild; stale results are cleared.

## Stage 8 — Audit report + downloadable project (done)
- `audit_report.py`: Markdown of the real computed objects (profile, decisions, leaderboard including failed and SMOTE rows, tuning, held-out evaluation with decoded labels, interpretation).
- `export.build_project_zip()`: a runnable project made of `train.py` (configuration block plus the same planner, options and model), `predict.py` (decodes labels), the library source, the fitted `model.joblib`, a README and requirements.
- Tests: `tests/test_audit_report_cross_stage.py` covers target suggestion, options, the audit report, and a test that unzips the project and runs `train.py` and `predict.py` standalone.

## Stage 9 — Benchmark + result diagnostics (done)
- `benchmark.py` and `scripts/run_benchmark.py [--tune] [--online]`: run the real flow end to end and write `reports/benchmark.md`. Each row records target-suggestion success, baseline vs best CV, held-out score, runtime, leakage suspects and warnings. Failures are recorded, not raised.
- `diagnostics.py`: `detect_leakage_suspects()` (training data only: a numeric column with rank correlation >= 0.98 with the target, or a categorical column that maps to one class >= 99% of the time) and `assess_results()` (barely beats baseline, unstable folds, small test set, failures). Both appear in the UI after Finalize and in the audit report (section 7).
- Bug #15: nested parallelism. Models used `n_jobs=-1` inside a cross-validation that also runs in parallel, so threads oversubscribed the CPU (LightGBM on 142 rows took 67 s). Zoo models are now `n_jobs=1`, and `use_all_cores()` is applied only for single final fits. The benchmark went from about 7 min to about 70 s and the full test suite from 25 min to 2.5 min, with the same scores.
- Latest results (tuned, 6 datasets) are in `reports/benchmark.md`: the right target was suggested 6/6, and the best model beat the baseline by +0.32 to +0.80 CV.

### Real-world evaluation (6 more datasets)
Churn, Iris, Social Network Ads, Data Science Job, German Credit (biased training) and a headerless Wine file. The right target was found on 6/6 and every best model beat the baseline by +0.25 to +0.80 CV (`reports/real_world_evaluation.md`). Four fixes came out of it: headerless CSV detection (`io.py`), a first-column class-label cue for target suggestion, id-name normalisation (`User ID`, `CustomerId`), and a sensitive-attribute check. A LogisticRegression n_jobs warning was also removed. 56 tests pass.

### Accuracy upgrade round
Measured in `reports/accuracy_improvements.md` and `reports/benchmark_all_datasets.md` (14 datasets, 3 of them new regression sets). Added: Extra Trees and Hist Gradient Boosting, a top-3 ensemble, a tuned decision threshold (binary), a log-target model (skewed regression), cross-fitted target encoding, and a larger tuning budget. Best gains: +0.032 CV on the imbalanced set, +0.024 on Data Science Job, +0.011 on Churn. Models are rebuilt from their names via `resolve_model_spec`, including in exported projects. 62 tests pass.

### Kaggle-readiness round
- `cleaning.py`: unicode-aware spelling-variant merging (`category_key`), `RawCleaner` (stateless first pipeline step; also reads text like "$114.17" or "1,200" as numbers), boolean-label target cleaning (only when the target is genuinely messy; clean yes/no stays readable), exact-duplicate dropping before the split.
- `scale.py`: model search and tuning run on a stratified sample (default 40k rows); the winner is refit on every training row.
- `submission.py`: builds `submission.csv` in the sample submission's format (probabilities when the sample holds probabilities). App: optional test.csv + sample_submission.csv upload after Finalize.
- Metric choice (f1, accuracy, roc_auc, f2, precision, recall, r2) for ranking and tuning.
- `clustering.py` + app panel: no output column -> KMeans, Ward and Gaussian mixture compared by silhouette, cluster profiles in original units, PCA plot, labelled CSV.
- Bugs found by the new datasets and fixed: category merging erased every non-Latin label (Japanese Pokemon types collapsed into one class); XGBoost crashed when a CV fold lacked a rare class (now wrapped, with a fitted-check fix).
- Results: `reports/benchmark_kaggle_round.md` and `reports/kaggle_readiness.md`. Tests: 77 passing.

### Robustness + interface round
- Crash handling in three layers: a plain rescue pipeline retries any model that fails on all main pipelines (`plan_rescue`); the final fit walks down the leaderboard if the winner cannot be refit, ending on a simple model on the rescue pipeline (`finalize.fit_best_with_fallback`, shared by app and benchmark); a clear error only if nothing at all can be fitted.
- Model bench: added KNN, AdaBoost, Neural Net (classification), Ridge, Elastic Net, KNN, AdaBoost, Neural Net (regression) as opt-in backups with row limits; CatBoost and stacking ensembles are in the zoo. Search-effort slider (Fast / Balanced / Thorough) for tuning.
- Interface: wine / warm-paper / sand palette, glass nav and tab controls, live status panel on upload (read, profile, find output), progress bar and floating status pill while models train, staggered chips, reduced-motion support. `.streamlit/config.toml` carries the theme colours; `live.py` holds the HTML builders (unit-tested).
- Results: `reports/benchmark_robust_round.md`. Tests: 86 passing.

## Next
- Stage 10: polish and handoff (final README, run guide, cleanup, final zip).

## Stage 10 - Polish and handoff

- Interface reworked around how people actually work: **Fast** mode (one button, live progress) and **Step by step** (one "Next" button, or "Finish the rest"). Copy cut to the minimum.
- The hero's dead nav became a real progress rail (Upload / Configure / Train / Results).
- Results now live in tabs: Results, Models, Pipelines, Program, Competition.
- **Program** tab (workbench): view, edit, run `train.py` / `predict.py` on your data, reset, download the edited project (`workbench.py`).
- **Competition** tab is optional and outside the main flow: metric choice, test/sample upload, submission checks (`submission.check_submission`), download. No rank is ever claimed.
- `runner.py` holds the pure orchestration shared by Fast and Step by step; `tests/test_workbench_runner.py` covers it.
- Handoff: new README with VS Code setup, `.vscode/` launch + settings, `streamlit>=1.50` pin.

### Stage 10, second round
- **One-file program.** `notebook_export.py` writes `train.py` as ten `# %%` steps with the winning preprocessing written out as scikit-learn code (verified: rerunning it reproduces the app's held-out score exactly). Program tab shows it step by step with Edit / Run.
- **Competition flow.** Test and sample upload, retrain on all rows, submission checks, plus an editable `make_submission.py` that can be run from the app.
- **Brain.** `StringShape` mines titles, prefixes, lengths and missing flags from code-like text (names, tickets, cabins) that used to be dropped; repeated cross-validation on small data (2 repeats Balanced, 3 Thorough, up to 3,000 rows) so the winner is less likely to be luck. A/B on quick runs: Titanic 0.812 to 0.820, Data Science Job 0.706 to 0.707, Churn and German credit unchanged (no harm).
- **Interface.** Drifting-wave footer band that mirrors the hero; honours reduced-motion.

### Stage 10, third round (accuracy audit)
- **Finding 1: single-split noise.** A 176-row test moves ~3 points from the choice of rows alone. Same code gave 0.784-0.818 accuracy on Titanic's default split; on six other splits it gave 0.827-0.866 (mean 0.846), about 2 points above fixed single-model baselines on the same splits.
- **Finding 2: the features help.** 5x4 repeated CV, balanced pipeline: bare 0.797-0.824 -> all features 0.826-0.834 accuracy (HGB, Random Forest, CatBoost). Text-structure and target encoding each add 0.5-1.5 points; interactions are neutral.
- **Finding 3 (bug): tuning bias.** The tuned score was the maximum over up to 40 search tries (3-fold), compared with the untuned mean (5-fold). Tuned settings nearly always "won", even by luck. Now the tuned settings and the defaults are re-scored on identical fresh folds and the tuned settings are kept only if they gain at least 0.003 (`tuning._verify_against_defaults`).
- **Reliability tools.** Results show a 95% range for the held-out score next to the cross-validated score; "Check stability" refits the winner on 5 other splits; optional model trained on all rows for real use.
- **UI.** Footer back to the plain line with a slow light sweep; KPI cards and importance bars animate in.

### Stage 10, fourth round (speed on a large Kaggle table)
Test file: Playground S4E4 Abalone (90,615 rows, target Rings), Balanced depth, 2-core machine.
- **Finding: where the time went.** Total 776 s: model race 682 s, tuning 91 s, final fit 3 s. Random Forest and Extra Trees (300 deep trees, 5 folds, 3 pipelines) used about 950 of the 1,300 CPU-seconds and finished behind the boosters (R2 0.64 vs 0.66). Each model also refit the same preprocessing on every fold, and models ran one after another with at most 5 fits alive at once.
- **Fix 1 (`parallel.py`).** Preprocessing is fitted once per pipeline and fold; every model x fold fit then runs in one process pool, slowest first. Scores match `cross_validate` exactly (tested).
- **Fix 2.** On 5,000+ rows, a slow model that trails the best by more than 2% after fold 1 is not run on the other folds (shown as "stopped early"). Forests use 100 trees above 10,000 rows (tuning can raise it).
- **Fix 3.** Tuning shares the same cached folds, so each setting no longer refits the preprocessing.
- **Result.** 776 s -> about 180 s on 2 cores; same winning ensemble and the same held-out R2 (0.6622). More cores help further; not measured here.
- **RMSLE.** New ranking choice for regression; held-out RMSLE 0.1518 (ranked by R2) -> 0.1508 (ranked by RMSLE). Kaggle's own score is not visible to DataPilot.
- **Ensembles explained.** Voting/Stacking models now report averaged feature importances (members that expose them).
- **Cleanup.** Removed unused `ContextualQuestionEngine`, `DatasetProfile.summary`, two unused preference fields and a dead UI line.

## Excel input and plain-words screen (after HOD feedback)
- `io.py`: `.xlsx/.xlsm/.xls` are read like CSV (`read_any`, `excel_sheets`, `read_table(source, sheet)`); generated `train.py` / `predict.py` read either kind. `openpyxl` and `xlrd` added to requirements. Tests: `tests/test_excel_input.py`.
- `explain.py`: the three data-preparation options are shown as Simple cleaning / Standard cleaning / Extra feature building (`prep_label`); a plain-words reading of the held-out score (`plain_summary`), readable column names, and a one-example predictor (`predict_example`). Tests: `tests/test_explain.py`.
- App: Results shows the plain reading and a "Try it on one example" box; the former Pipelines tab is now an *Advanced* expander; the Models tab lists the best result per model with the full table under an expander; check messages rewritten in everyday words.
