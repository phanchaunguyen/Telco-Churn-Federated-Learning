# E2 — Partition impact

FedAvg, 50 rounds, scored on `test_real` (1,318 real rows).

| Partition | AUC | 95% CI | PR-AUC | Recall@10% | Converged at round | Per-client AUC std | Gap vs IID |
|---|---|---|---|---|---|---|---|
| iid | 0.8949 | [0.8755, 0.9123] | 0.8121 | 0.3289 | 14 | 0.0018 | +0.00% |
| label_skew_a0.5 | 0.8928 | [0.8737, 0.9111] | 0.7985 | 0.3289 | 21 | 0.0070 | +0.24% |
| label_skew_a0.1 | 0.8889 | [0.8695, 0.9068] | 0.7876 | 0.3155 | 14 | 0.0045 | +0.67% |

**Success criterion #3 (non-IID degrades AUC): PASS** — alpha=0.1 costs 0.67% vs IID
