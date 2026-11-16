# DataPilot Benchmark

| Dataset | Rows x cols | Task | Target found | Best (pipeline / model) | Baseline CV | Best CV | Uplift | Held-out | Runtime (s) |
|---|---|---|---|---|---|---|---|---|---|
| breast_cancer | 569 x 31 | classification | yes | balanced / Logistic Regression | 0.385 | 0.974 | 0.589 | f1=0.981 | 19 |
| wine | 178 x 14 | classification | yes | aggressive / LightGBM | 0.191 | 0.986 | 0.795 | f1=0.920 | 10 |
| iris | 150 x 5 | classification | yes | conservative / Logistic Regression | 0.164 | 0.958 | 0.795 | f1=1.000 | 6 |
| diabetes | 442 x 11 | regression | yes | balanced / Linear Regression | -0.015 | 0.488 | 0.503 | r2=0.481 | 6 |
| synthetic_imbalanced_3000 | 3000 x 13 | classification | yes | balanced / LightGBM | 0.489 | 0.807 | 0.319 | f1=0.861 | 21 |
| titanic | 891 x 12 | classification | yes | balanced / LightGBM | 0.381 | 0.805 | 0.424 | f1=0.793 | 10 |

## Flags
- **wine** Only 36 held-out rows: test metrics are noisy.
- **iris** Only 30 held-out rows: test metrics are noisy.
- **diabetes** Only 89 held-out rows: test metrics are noisy.
