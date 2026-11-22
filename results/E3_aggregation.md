# E3 — Aggregation robustness

Partition `label_skew_a0.1`, 50 rounds, scored on `test_real` (1,318 real rows).

| Strategy | AUC | 95% CI | PR-AUC | Recall@10% | Converged at round | Per-client AUC std | Per-client AUC min | Gain vs FedAvg |
|---|---|---|---|---|---|---|---|---|
| fedavg | 0.8889 | [0.8695, 0.9068] | 0.7876 | 0.3155 | 14 | 0.0045 | 0.7883 | +0.00% |
| fedprox | 0.8835 | [0.8628, 0.9020] | 0.7743 | 0.3155 | 30 | 0.0040 | 0.7837 | -0.60% |
| fedavgm | 0.8875 | [0.8677, 0.9056] | 0.7827 | 0.3182 | 20 | 0.0037 | 0.7871 | -0.16% |

**Success criterion #4 (FedProx/FedAvgM beat FedAvg): FAIL** — best alternative gains -0.16%
