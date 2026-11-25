# D14 — Spark evaluation parity

AUC computed twice on the same 1,318 predictions: once with scikit-learn
in the pandas path, once with Spark SQL using the rank-based
Mann-Whitney formula.

| Metric | Value |
|---|---|
| Spark AUC | 0.894901 |
| pandas AUC | 0.894901 |
| Absolute difference | 4.74e-07 |
| Tolerance | 0.001 |
| Rows | 1,318 (374 pos / 944 neg) |

**Success criterion #11 (Spark AUC matches pandas within 0.001): PASS**

> **Why the formula is written out.** Spark MLlib has an AUC evaluator,
> but calling it would test MLlib rather than the aggregation. AUC has a
> closed form in terms of ranks, and expressing it in SQL is what proves
> the aggregation itself is correct. Ties are handled by averaging the
> ranks a tie group spans, which `RANK` alone cannot do.
