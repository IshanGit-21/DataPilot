# DataPilot Benchmark

| Dataset | Rows x cols | Task | Target found | Best (pipeline / model) | Baseline CV | Best CV | Uplift | Held-out | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| churn | 10000 x 14 | classification | yes | balanced / XGBoost + SMOTE | 0.443 | 0.750 | 0.307 | f1=0.768 | 50 |
| iris | 150 x 6 | classification | yes | conservative / Logistic Regression | 0.164 | 0.958 | 0.795 | f1=1.000 | 6 |
| social_ads | 400 x 5 | classification | yes | aggressive / Random Forest | 0.392 | 0.893 | 0.502 | f1=0.907 | 8 |
| ds_job | 19158 x 13 | classification | yes | conservative / Logistic Regression | 0.429 | 0.682 | 0.253 | f1=0.695 | 98 |
| german_credit | 5000 x 21 | classification | yes | balanced / Random Forest + SMOTE | 0.400 | 0.753 | 0.353 | f1=0.759 | 45 |
| wine_noheader | 178 x 14 | classification | yes | aggressive / LightGBM | 0.191 | 0.986 | 0.795 | f1=0.920 | 10 |

## Flags
- **churn** Sensitive attributes used as inputs: Gender
- **iris** Only 30 held-out rows: test metrics are noisy.
- **social_ads** Only 80 held-out rows: test metrics are noisy.
- **social_ads** Sensitive attributes used as inputs: Gender
- **ds_job** Sensitive attributes used as inputs: gender
- **german_credit** Sensitive attributes used as inputs: Sex, ForeignWorker
- **wine_noheader** Only 36 held-out rows: test metrics are noisy.

## What these datasets exposed (and what changed)
| Finding | Fix |
|---|---|
| `wine_data.csv` has no header row, so pandas used the first data row as column names | `read_table()` detects an all-numeric first row and names columns `col_0...`; the app tells the user |
| Wine target was suggested as the last continuous column (`col_13`) instead of the class label (`col_0`) | Target suggestion now credits a first column with few integer values (UCI class-label convention) |
| `User ID` and `CustomerId` were not flagged as identifiers (name check only matched `user_id` style) | Names are normalised (spaces, camelCase, hyphens) before the id check |
| `german_credit` uses `Sex` and `ForeignWorker`; churn, ads and job data use gender | New check lists sensitive attributes used as inputs, in the app and the audit report |
