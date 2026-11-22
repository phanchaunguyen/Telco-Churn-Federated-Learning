# E1 — FL gap

Folds: [0] · Test set: `test_real` (1,318 real rows)

| Model | AUC (mean ± std) | AUC min | AUC max | F1 | Recall@10% | Gap vs centralized |
|---|---|---|---|---|---|---|
| centralized | 0.8966 ± nan | 0.8966 | 0.8966 | 0.6420 | 0.3369 | 0.00% |
| federated | 0.8949 ± nan | 0.8949 | 0.8949 | 0.6667 | 0.3289 | 0.19% |
| local_only | 0.8879 ± nan | 0.8879 | 0.8879 | 0.6592 | 0.3275 | 0.98% |

**Success criterion #2 (FL gap < 5%): PASS** — measured 0.19%
