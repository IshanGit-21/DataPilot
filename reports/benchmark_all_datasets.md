# DataPilot Benchmark

| Dataset | Rows x cols | Task | Target found | Best (pipeline / model) | Baseline CV | Best CV | Uplift | Held-out | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | 569 x 31 | classification | yes | conservative / Extra Trees | 0.385 | 0.972 | 0.587 | f1=0.953 | 49 |
| wine_sk | 178 x 14 | classification | yes | conservative / Extra Trees | 0.191 | 0.987 | 0.796 | f1=1.000 | 25 |
| diabetes | 442 x 11 | regression | yes | balanced / Linear Regression | -0.015 | 0.488 | 0.503 | r2=0.481 | 17 |
| synthetic_imb | 3000 x 13 | classification | yes | balanced / Extra Trees + SMOTE | 0.489 | 0.839 | 0.350 | f1=0.862 | 92 |
| titanic | 891 x 12 | classification | yes | balanced / Hist Gradient Boosting | 0.381 | 0.817 | 0.436 | f1=0.777 | 47 |
| churn | 10000 x 14 | classification | yes | balanced / Random Forest | 0.443 | 0.761 | 0.318 | f1=0.764 | 149 |
| social_ads | 400 x 5 | classification | yes | aggressive / Random Forest | 0.392 | 0.908 | 0.516 | f1=0.895 | 26 |
| ds_job | 19158 x 13 | classification | yes | balanced / Ensemble: Hist Gradient Boosting + Random Forest + Logistic Regression | 0.429 | 0.706 | 0.277 | f1=0.720 | 261 |
| german_credit | 5000 x 21 | classification | yes | balanced / Random Forest + SMOTE | 0.400 | 0.757 | 0.357 | f1=0.759 | 140 |
| wine_noheader | 178 x 14 | classification | yes | conservative / Extra Trees | 0.191 | 0.987 | 0.796 | f1=1.000 | 26 |
| iris | 150 x 6 | classification | yes | conservative / Ensemble: Logistic Regression + Hist Gradient Boosting + LightGBM | 0.164 | 0.967 | 0.803 | f1=0.900 | 15 |

## Flags
- **wine_sk** Only 36 held-out rows: test metrics are noisy.
- **diabetes** Only 89 held-out rows: test metrics are noisy.
- **diabetes** Sensitive attributes used as inputs: sex
- **titanic** Sensitive attributes used as inputs: Sex
- **churn** Sensitive attributes used as inputs: Gender
- **social_ads** Only 80 held-out rows: test metrics are noisy.
- **social_ads** Sensitive attributes used as inputs: Gender
- **ds_job** Sensitive attributes used as inputs: gender
- **german_credit** Sensitive attributes used as inputs: Sex, ForeignWorker
- **wine_noheader** Only 36 held-out rows: test metrics are noisy.
- **iris** Only 30 held-out rows: test metrics are noisy.
