# E0 — IBM Churn Score baseline

The `Churn Score` column ships with the dataset and is IBM's own churn
propensity estimate. It is post-hoc derived, so it can never be a
feature — but it is the yardstick the federated model has to beat.

| Metric | Value |
|---|---|
| AUC | **0.9421** |
| Samples | 6,589 |
| Churned | 1,869 |
| Score range | 5 – 96 |
| Score mean | 59.19 |

> **How to read this.** No model was trained. The score is used directly
> as a ranking, so this is the ceiling a single free column achieves.
> The federated model reaches AUC 0.8949 (E1 fold 0), which is below this
> baseline — expected, because `Churn Score` is derived from information
> the model is not allowed to see.
