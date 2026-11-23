# E5 — Fairness

Partition `label_skew_a0.1`, scored on `test_real` (1,318 rows).

## E5.1 — Per-client fairness

| Strategy | Overall AUC | Client AUC mean | Client AUC std | Worst client | Best client | Spread |
|---|---|---|---|---|---|---|
| fedavg | 0.8889 | 0.7950 | 0.0045 | 0.7883 | 0.8049 | 0.0166 |
| fedprox | 0.8835 | 0.7914 | 0.0040 | 0.7837 | 0.7955 | 0.0117 |

## E5.2 — Recall by churn reason

Share of churners flagged by the global top-10% cut-off, per reason.

| Strategy | Recall mean | Recall std | Worst reason | Best reason | Spread |
|---|---|---|---|---|---|
| fedavg | 0.2940 | 0.0648 | 0.2000 | 0.3647 | 0.1647 |
| fedprox | 0.3088 | 0.0570 | 0.2200 | 0.3750 | 0.1550 |

### Per-reason detail — fedavg

| Reason | Churned | Flagged | Recall |
|---|---|---|---|
| Competitor | 170 | 62 | 0.3647 |
| Attitude | 58 | 15 | 0.2586 |
| Dissatisfaction | 56 | 18 | 0.3214 |
| Price | 50 | 10 | 0.2000 |
| Other | 40 | 13 | 0.3250 |

**Success criterion #6 (fedprox reduces unfairness): PASS** — client AUC std +10.34%, reason recall std +12.04% (positive = less dispersion = fairer)

> **Caveat (E5.2).** With 374 churned customers spread across 5 reasons, each reason holds 40-170 positives. Recall on 50 positives has a 95% CI of roughly ±0.13, so differences smaller than that between strategies should not be interpreted as meaningful.

> **Why recall and not AUC.** `Churn Category` is only populated for customers who churned, so every reason stratum is 100% positive. AUC needs both classes and is undefined there.

> **How to read the verdict.** The two fairness views can disagree, and neither is decisive on its own. A PASS here means the alternative strategy reduced dispersion on at least one view; it does **not** mean the model is fair. With 40-170 positives per reason, a difference of a few percentage points is inside the noise band, so the honest reading is *no detectable difference* rather than *FedProx is fairer*.
