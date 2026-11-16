# DataPilot

Upload a CSV or Excel file, say what to predict (will this customer leave? what will this house sell for?),
and DataPilot cleans the data, tries many models, tunes the best, tests it once on rows it has never seen,
and gives you the model, the evidence, and a program you can read and edit inside the app.

It also finds groups when there is nothing to predict (clustering), and has an optional
Competition tab for Kaggle-style files. Everything runs on your machine; nothing is uploaded.

## Run it (VS Code, Windows / macOS / Linux)

1. Install **Python 3.11 or newer** (python.org; on Windows tick "Add Python to PATH") and **VS Code** with the *Python* extension.
2. Unzip the project, then in VS Code: **File > Open Folder** > the `datapilot` folder.
3. Open a terminal in VS Code (**Terminal > New Terminal**) and create the environment:

   | | Windows (PowerShell) | macOS / Linux |
   |---|---|---|
   | create | `py -m venv .venv` | `python3 -m venv .venv` |
   | activate | `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

   If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
4. Install: `pip install -r requirements.txt` then `pip install -e .`
5. When VS Code asks, pick the `.venv` interpreter (or **Ctrl/Cmd+Shift+P > Python: Select Interpreter**).
6. Start the app: `streamlit run src/datapilot/app.py` (or press **F5** and choose *DataPilot app*). It opens at http://localhost:8501.

Check your install any time with `pip install -r requirements-dev.txt` then `pytest -q` (about 10 minutes; the quick part is `pytest tests/test_workbench_runner.py`).

### Troubleshooting
- **`pip` fails on catboost / lightgbm / xgboost**: DataPilot still works without them; it skips the model it cannot import. Try `pip install --upgrade pip` first, and use Python 3.11-3.13.
- **`streamlit` not found**: the environment is not active. Re-run the activate command, or use `python -m streamlit run src/datapilot/app.py`.
- **Port busy**: add `--server.port 8600`.
- **Large file is slow**: pick *Depth: Quick*. Search and tuning run on a 20,000-row sample; the winner is refit on all rows.

## Using it

1. **Upload** a CSV or Excel file (.xlsx, .xls, .xlsm; pick the sheet if there are several). DataPilot reads it and suggests what to predict.
2. **Test file** (next to the upload box): add one if you have it, like Kaggle's `test.csv`.
   - With the answers in it, it becomes the final test, so no training rows are set aside.
   - Without answers (Kaggle style), every training row is used to learn; the score comes from cross-checking on the training rows (each row is predicted by a model that did not see it), and the Competition tab already has your test file ready for `submission.csv`.
   - No test file: 20% of the rows are set aside, as before.
3. **Predict**: pick the output column. Open *Options* to change inputs, task, depth, the metric to rank by, or priority.
   One button runs everything and shows a live picture of what DataPilot is doing (reading columns, building features, models finishing, tuning, final test) with a running clock.
4. Read the results in tabs:
   - **Results**: an Accuracy card beside the other scores (for number targets: 100% minus the average percentage error), what the prediction depends on most, a "try it on one example" box, checks, downloads (project zip, audit report). An *Advanced* section shows how the data was prepared (Simple, Standard or Extra feature building) and lets you change settings.
   - **Summary**: what DataPilot did and how, in numbered plain steps, with the real time each stage took.
   - **Models**: the best result per model, tuning outcome; every model with every data-preparation option is under an expander.
   - **Program**: your whole program as one file, `train.py`, in ten labelled steps (imports, settings, load, clean and split, features, model, train, tune, evaluate, save). Pick a step, switch on *Edit*, change it, then *Run program*. Library files and the fitted model sit under *Other project files* if you ever want them.
   - **Competition** (optional): choose the competition metric, add `test.csv` (and `sample_submission.csv` if you have it). DataPilot retrains on all rows, predicts, checks the file and gives you `submission.csv`, plus an editable `make_submission.py` you can change and run on the same files. DataPilot cannot submit for you or see a leaderboard, so it never claims a rank.

## What is inside

```
src/datapilot/
  app.py            Streamlit interface        runner.py     one-shot / staged orchestration
  profiler.py       understand the data        cleaning.py   safe cleaning (text numbers, spellings, duplicates)
  planner.py        3 pipelines + rescue       model_zoo.py  models incl. backups
  experiment.py     model race                 ensemble.py   voting / stacking
  parallel.py       runs every model x fold fit at once, preprocessing once per fold
  tuning.py         randomized search          imbalance.py  SMOTE, thresholds
  finalize.py       final fit with fallbacks   evaluation.py held-out metrics
  interpretation.py what the model relies on   audit_report.py  plain-language audit
  export.py         runnable project zip       notebook_export.py  writes the one-file train.py
  workbench.py      edit / run the project
  submission.py     Kaggle files + checks      clustering.py find groups
  scale.py          big-data sampling          live.py, theme.py  interface pieces
tests/              pytest suite
reports/            benchmark write-ups on real datasets
PROJECT_PHASES.md   stage-by-stage history
```

## Guarantees and limits
- Preprocessing lives inside the sklearn pipeline, so it is fit on training folds only (no leakage); the test rows are touched once.
- Speed: all model x fold fits run on every CPU core (set `DATAPILOT_JOBS=1` to run one at a time). Every model and every data preparation is still tried. Big tables are searched on a 20,000-row sample with 3 folds; slow models (forests, neural nets) that trail the best after the first fold on 5,000+ rows are stopped early and marked so in the leaderboard.
- Big tables: the final boosted models (LightGBM, XGBoost, CatBoost) are fitted on every row and pick their own number of trees, using 10% of their own training rows to know when to stop.
- Regression can be ranked by R2, RMSE, MAE or RMSLE. RMSLE trains on log(1+target) and converts predictions back; it needs a target that is never negative.
- If a model or pipeline fails, DataPilot falls back to a simpler one and says so.
- Scores come from your data; nothing is simulated. Small or noisy data gives modest scores, and the audit says so.
- No cloud deployment or monitoring is included; the exported project is plain scikit-learn and runs anywhere Python does.
