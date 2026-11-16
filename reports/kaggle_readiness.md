# DataPilot on Kaggle-style problems: what was measured

All numbers are local: a held-out 20% split DataPilot makes itself. No Kaggle leaderboard was available, so no rank is claimed.

| Dataset | Problem | Result (held-out) | Reading |
|---|---|---|---|
| Airline Satisfaction (558k train rows) | binary, ROC-AUC | AUC 0.955, accuracy 0.921 | Search on a 20k sample, winner refit on all rows. A larger booster reached ~0.957, so the data, not model size, is the limit. submission.csv produced (299,844 rows, probabilities). |
| Disaster tweets (text) | binary, F1 | F1 0.761 | TF-IDF + boosted trees. Transformer models on this competition score higher; DataPilot has no pretrained language models. |
| heart | binary | F1 0.831 (CV 0.829) | Only 61 test rows, so noisy. |
| Pokemon cards EN / JP (Type) | 13-class | F1 0.978 / 0.968 (CV ~0.82) | Held-out is far above CV: reprints of the same card sit on both sides of the split. Trust the CV number. |
| messy_customer_orders | binary, noisy | F1 0.496 (baseline 0.334) | Cleaning worked (12 label spellings, 1000 duplicates, "$/£" prices). The table carries little signal. |
| play_tennis (14 rows) | binary | no better than baseline | Correctly reported as "barely beats baseline". |
| Country-data (gdpp) | regression | R2 0.647 (CV 0.932) | 34 test rows, noisy. |
| Mall customers / Country-data | clustering | 8 groups, silhouette 0.33 / 2 groups, 0.32 | Groups described in original units. |

## Bugs the new data exposed (fixed, tested)
- Spelling-merge erased all non-Latin labels (Japanese target collapsed to one class).
- XGBoost crashed whenever a CV fold missed a rare class.
- Prices stored as text ("$114.17") were treated as categories.
- A clean yes/no target was being turned into 0/1.

## Fit with the competitions in the screenshots
- Tabular supervised competitions (like Airline Satisfaction): direct fit.
- Pokemon battle (reinforcement learning) and NFL Big Data Bowl (tracking-data analytics): not tabular prediction, out of scope.

## Limits
- Strong baseline and solid mid-table results are realistic; winning a leaderboard needs hand-made features, long tuning and large ensembles that DataPilot does not do.
- No near-duplicate detection (the Pokemon reprint effect).
- The Kaggle submission can only be checked for format here, not scored.
