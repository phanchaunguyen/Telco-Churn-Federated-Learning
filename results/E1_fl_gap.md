# E1 — FL gap

Folds: [0, 1, 2, 3, 4] · Test set: `test_real` (1,318 real rows)

| Model | AUC (mean ± std) | AUC min | AUC max | F1 | Recall@10% | Gap vs centralized |
|---|---|---|---|---|---|---|
| centralized | 0.8954 ± 0.0020 | 0.8931 | 0.8981 | 0.6285 | 0.3326 | 0.00% |
| federated | 0.8980 ± 0.0140 | 0.8841 | 0.9159 | 0.6659 | 0.3337 | -0.30% |
| local_only | 0.8880 ± 0.0004 | 0.8876 | 0.8885 | 0.6613 | 0.3251 | 0.82% |

**Success criterion #2 (FL gap < 5%): PASS** — measured -0.30%
