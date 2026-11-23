# E7 — Communication cost

FedAvg, IID, K=10, 50 rounds. Scored on `test_real` (1,318 rows).

`bytes_per_round = 2 x 18,049 params x 4 bytes x n_selected`

| Fraction | Clients/round | MB/round | Cumulative MB | AUC (real) | PR-AUC (real) | Recall@10% | AUC (synth) |
|---|---|---|---|---|---|---|---|
| 0.25 | 2 | 0.289 | 14.4 | 0.8942 | 0.8099 | 0.3289 | 0.7994 |
| 0.50 | 5 | 0.722 | 36.1 | 0.8949 | 0.8121 | 0.3289 | 0.8007 |
| 1.00 | 10 | 1.444 | 72.2 | 0.8958 | 0.8152 | 0.3316 | 0.8008 |

**Success criterion (cheapest setting stays within 2% AUC): PASS** — fraction=0.25 costs 0.289 MB/round and loses 0.18% AUC vs the best setting

> **What is counted.** Model weights only: no features, no labels, no
> gradients. `num_examples` is sent for the weighted average but is a
> single integer per client, so it is excluded as negligible.

> **Why the factor of 2.** Every round downloads the global model to each
> selected client and uploads the updated weights back, so the wire cost
> is double the model size per client.

> **Note on fraction=0.25.** With K=10 the selection is
> `max(2, round(10 * 0.25)) = 2` clients, not the 2-3 the protocol
> sketched. The floor of 2 exists because FedAvg needs at least two
> clients to average.
