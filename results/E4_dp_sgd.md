# E4 — Differential privacy utility cost

Partition `iid`, FedAvg with DP-SGD, scored on `test_real` (1,318 rows).

| Noise sigma | Epsilon (delta=1e-5) | AUC | 95% CI | PR-AUC | Recall@10% | AUC loss vs non-DP |
|---|---|---|---|---|---|---|
| 0.0 | inf | 0.8949 | [0.8755, 0.9123] | 0.8121 | 0.3289 | +0.00% |
| 0.5 | 12.37 | 0.8941 | [0.8749, 0.9108] | 0.7877 | 0.3262 | +0.09% |
| 1.0 | 1.55 | 0.8894 | [0.8697, 0.9067] | 0.7500 | 0.3128 | +0.62% |
| 2.0 | 0.57 | 0.8865 | [0.8663, 0.9047] | 0.7799 | 0.3128 | +0.94% |

**Success criterion #5 (privacy costs utility, monotonically): PASS** — worst-case AUC loss +0.94% at the highest noise level

Non-DP reference AUC: 0.8949
