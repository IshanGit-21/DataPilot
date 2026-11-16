# Accuracy improvements — measured before vs after

Cross-validated best score (macro-F1 for classification, R² for regression). Held-out scores are shown for reference but several test sets are only 30-90 rows, so CV is the steadier signal. "After" CV includes tuning-selection optimism, same as "before".

| Dataset | Before CV | After CV | Change | Held-out before → after |
|---|---|---|---|---|
| synthetic imbalanced (4% positive) | 0.807 | 0.839 | +0.032 | 0.861 → 0.862 |
| Data Science Job | 0.682 | 0.706 | +0.024 | 0.695 → 0.720 |
| Titanic | 0.805 | 0.817 | +0.012 | 0.793 → 0.777 |
| Churn | 0.750 | 0.761 | +0.011 | 0.768 → 0.764 |
| German credit | 0.753 | 0.757 | +0.004 | 0.759 → 0.759 |
| Iris | 0.958 | 0.967 | +0.009 | 1.000 → 0.900 (30 rows) |
| Wine | 0.986 | 0.987 | +0.001 | 0.920 → 1.000 (36 rows) |
| Breast cancer | 0.974 | 0.972 | -0.002 | 0.981 → 0.953 (114 rows) |
| Social network ads | 0.893-0.914 | 0.908 | within run-to-run range | 0.893-0.907 → 0.895 |
| Diabetes (regression) | 0.488 | 0.488 | 0 (linear is best) | 0.481 → 0.481 |

New regression sets (first run with the new system): California housing R² 0.836 (held-out 0.838), Abalone 0.559 (0.564), Boston 0.866 (0.874).

## What changed
1. Two more model types: Extra Trees and Hist Gradient Boosting (handles missing values natively). Extra Trees won on 4 datasets, Hist Gradient Boosting on Churn/Titanic variants.
2. Ensemble entry: soft-voting / averaging of the top 3 distinct model types on the best pipeline, scored under the same CV. It won on Titanic, Data Science Job and Iris.
3. Tuned decision threshold for binary targets, chosen on out-of-fold predictions inside each training fold, so CV stays honest. It won on Data Science Job in one run.
4. Log-target model for right-skewed non-negative regression targets. It is scored, but CV did not pick it on any of the datasets tried.
5. Cross-fitted target encoding for high-cardinality categoricals (balanced and aggressive). Effect on the datasets tried: neutral.
6. Tuning budget: top 3 model types (was 2), 20 search iterations (was 8), with search spaces for the two new models.
7. App and benchmark now share one best-model selector, and every model, including boosters, is rebuilt from its name (exported projects too).

## Honest limits
- Gains are largest on harder or imbalanced data; on datasets already near their ceiling (Iris, Wine, Breast cancer) there is nothing left to gain and differences are noise.
- Runtime rose (largest case, 19k rows: about 4 minutes with tuning) because more models and a bigger search run.
- Target encoding and the log-target model are included because CV decides whether they are used; they did not win here.
