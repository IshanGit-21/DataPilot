# DataPilot Benchmark

| Dataset | Rows x cols | Task | Target found | Best (pipeline / model) | Baseline CV | Best CV | Uplift | Held-out | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| heart (Kaggle/UCI) | 303 x 14 | classification | yes | aggressive / Logistic Regression | 0.352 | 0.829 | 0.477 | f1=0.831 | 42 |
| play_tennis (14 rows) | 14 x 6 | classification | yes | conservative / Baseline (most frequent) | 0.387 | 0.387 | 0.000 | f1=0.400 | 13 |
| messy_customer_orders | 51000 x 12 | classification | no | balanced / LightGBM | 0.334 | 0.505 | 0.171 | f1=0.496 | 854 |
| tweets disaster (text) | 7613 x 5 | classification | yes | balanced / XGBoost | 0.364 | 0.756 | 0.392 | f1=0.761 | 152 |
| pokemon_EN (Type) | 2273 x 18 | classification | no | aggressive / XGBoost | 0.021 | 0.821 | 0.800 | f1=0.978 | 316 |
| pokemon_JP (タイプ) | 2273 x 17 | classification | no | balanced / LightGBM | 0.021 | 0.807 | 0.786 | f1=0.968 | 226 |
| mall (country-data regression: gdpp) | 167 x 10 | regression | yes | conservative / Extra Trees | -0.017 | 0.932 | 0.949 | r2=0.647 | 16 |

## Flags
- **heart (Kaggle/UCI)** Only 61 held-out rows: test metrics are noisy.
- **heart (Kaggle/UCI)** Sensitive attributes used as inputs: sex
- **play_tennis (14 rows)** The best model (0.387) barely beats the naive baseline (0.387) on cross-validation; the inputs may carry little signal for this target.
- **play_tennis (14 rows)** Only 3 held-out rows: test metrics are noisy.
- **pokemon_EN (Type)** ROC-AUC could not be computed on the held-out set (a class is missing there).
- **pokemon_JP (タイプ)** ROC-AUC could not be computed on the held-out set (a class is missing there).
- **mall (country-data regression: gdpp)** Only 34 held-out rows: test metrics are noisy.
