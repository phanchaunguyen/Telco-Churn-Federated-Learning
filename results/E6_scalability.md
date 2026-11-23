# E6 — Scalability

FedAvg, IID, 50 rounds, `fraction_fit=0.5`. Timed on the federated
loop only (client construction excluded).

| K | Clients/round | Total (s) | s/round | s/client-round | Sanity AUC (synth) |
|---|---|---|---|---|---|
| 5 | 2 | 354.4 | 7.09 | 3.544 | 0.8012 |
| 10 | 5 | 381.1 | 7.62 | 1.524 | 0.8007 |
| 20 | 10 | 370.7 | 7.41 | 0.741 | 0.7994 |
| 50 | 25 | 387.1 | 7.74 | 0.310 | 0.7960 |

**Success criterion (wall-clock stays flat as K grows): PASS** — total time varies by only 9.2% across K=5..50 (354s to 387s)

### Why the cost is flat, not linear

`fraction_fit=0.5` is held constant, so the number of clients per round
grows with K while each client's silo shrinks by the same factor. The
total rows trained per round is therefore ~450,000 at every K:

| K | Clients/round | Rows per client | Rows per round |
|---|---|---|---|
| 5 | 2 | 180,000 | 360,000 |
| 10 | 5 | 90,000 | 450,000 |
| 20 | 10 | 45,000 | 450,000 |
| 50 | 25 | 18,000 | 450,000 |

This is the honest reading: **FL scales in client count without scaling
in wall-clock**, because the per-round workload is set by the sampling
fraction, not by K. The cost of more clients shows up as *less data per
client per round*, not as more compute.

> **What the AUC column is.** A sanity check on `synth_test`, not a
> quality claim. This experiment measures load; accuracy on synthetic
> data says nothing about the real population. A collapse here would
> indicate a bug in the sweep, not a finding about scaling.

> **Why IID for every K.** Varying K and partition skew together would
> confound two effects. IID holds the distribution fixed so K is the
> only thing that changes.
