# DataPilot Benchmark

| Dataset | Rows x cols | Task | Target found | Best (pipeline / model) | Baseline CV | Best CV | Uplift | Held-out | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| heart | 303 x 14 | classification | yes | aggressive / Logistic Regression | 0.352 | 0.829 | 0.477 | f1=0.831 | 34 |
| Churn | 10000 x 14 | classification | yes | conservative / Ensemble: CatBoost + Hist Gradient Boosting + Random Forest | 0.443 | 0.757 | 0.314 | f1=0.771 | 156 |
| Data Science Job | 19158 x 13 | classification | yes | conservative / Hist Gradient Boosting + threshold | 0.429 | 0.701 | 0.273 | f1=0.720 | 403 |
| German credit | 5000 x 21 | classification | yes | balanced / CatBoost + SMOTE | 0.399 | 0.765 | 0.365 | f1=0.763 | 238 |
| Social Network Ads | 400 x 5 | classification | yes | balanced / Random Forest | 0.387 | 0.914 | 0.527 | f1=0.864 | 27 |
| Iris | 150 x 6 | classification | yes | conservative / KNN | 0.163 | 0.983 | 0.820 | f1=0.967 | 19 |
| tweets | 7613 x 5 | classification | yes | balanced / XGBoost | 0.364 | 0.756 | 0.392 | f1=0.761 | 240 |
| Country gdpp | 167 x 10 | regression | yes | conservative / Extra Trees | -0.017 | 0.932 | 0.949 | r2=0.647 | 24 |

## Flags
- **heart** Only 61 held-out rows: test metrics are noisy.
- **heart** Sensitive attributes used as inputs: sex
- **Churn** Sensitive attributes used as inputs: Gender
- **Data Science Job** Sensitive attributes used as inputs: gender
- **German credit** Sensitive attributes used as inputs: Sex, ForeignWorker
- **Social Network Ads** Only 76 held-out rows: test metrics are noisy.
- **Social Network Ads** Sensitive attributes used as inputs: Gender
- **Iris** Only 30 held-out rows: test metrics are noisy.
- **Country gdpp** Only 34 held-out rows: test metrics are noisy.
