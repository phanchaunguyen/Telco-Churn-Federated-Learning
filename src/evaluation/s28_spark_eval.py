"""D14 — compute AUC with Spark SQL and check it against the pandas number.

Pipeline position: **PREDICT → EVALUATE (Spark)**

Layer 3 of the architecture says evaluation aggregation runs on Spark. Until
now every metric came from scikit-learn in a single Python process, so the
Spark half of that claim was untested.

This module reads the per-row predictions with Spark, computes AUC using only
Spark SQL, and asserts the result matches the pandas value to within 0.001.
That tolerance is success criterion #11.

Why AUC is computed by hand instead of calling a library
--------------------------------------------------------
Spark MLlib has ``BinaryClassificationEvaluator``, but using it would test
MLlib rather than the aggregation logic. AUC has a closed form in terms of
ranks — the Mann-Whitney U statistic — and expressing it in SQL is what proves
the aggregation is correct:

    AUC = (sum of ranks of positives - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

Ties are handled by assigning every tied row the average of the ranks it spans,
which is what ``RANK`` cannot do and ``AVG(rank)`` over a tie group can.

Usage
-----
>>> from src.evaluation.s28_spark_eval import run_spark_eval
>>> result = run_spark_eval()
>>> result["abs_diff"] <= 0.001
True
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import get_config
from src.data.s00_spark_session import get_spark, stop_spark

logger = logging.getLogger(__name__)

# Success criterion #11
MAX_ABS_DIFF: float = 0.001


# =============================================================================
# Paths
# =============================================================================
def predictions_path() -> Path:
    return get_config().paths.results_dir / "predictions" / "fl_fedavg.parquet"


def metadata_path() -> Path:
    return get_config().paths.results_dir / "predictions" / "predictions_metadata.json"


# =============================================================================
# Spark AUC
# =============================================================================
# Rank-based AUC with tie handling. `rank` gives the position in ascending
# order; averaging over a tie group turns it into the mid-rank the
# Mann-Whitney formula expects.
AUC_SQL = """
WITH ranked AS (
    SELECT
        y_true,
        y_score,
        RANK() OVER (ORDER BY y_score) AS r,
        COUNT(*) OVER (PARTITION BY y_score) AS tie_count
    FROM predictions
),
midrank AS (
    SELECT
        y_true,
        -- Mid-rank of a tie group: the mean of the ranks it spans.
        r + (tie_count - 1) / 2.0 AS rank_mid
    FROM ranked
),
totals AS (
    SELECT
        SUM(CASE WHEN y_true = 1 THEN 1 ELSE 0 END) AS n_pos,
        SUM(CASE WHEN y_true = 0 THEN 1 ELSE 0 END) AS n_neg,
        SUM(CASE WHEN y_true = 1 THEN rank_mid ELSE 0 END) AS sum_pos_ranks
    FROM midrank
)
SELECT
    n_pos,
    n_neg,
    (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg) AS auc
FROM totals
"""


def compute_auc_spark(spark: Any, path: Path) -> dict[str, Any]:
    """Compute AUC over a predictions parquet using Spark SQL only."""
    df = spark.read.parquet(str(path))
    df.createOrReplaceTempView("predictions")

    row = spark.sql(AUC_SQL).collect()[0]
    return {
        "auc": float(row["auc"]),
        "n_pos": int(row["n_pos"]),
        "n_neg": int(row["n_neg"]),
        "n_rows": int(row["n_pos"] + row["n_neg"]),
    }


# =============================================================================
# D14
# =============================================================================
def run_spark_eval(*, write: bool = True) -> dict[str, Any]:
    """Compute AUC on Spark and compare it with the pandas value.

    Returns
    -------
    dict
        ``{spark_auc, pandas_auc, abs_diff, passed, n_rows}``
    """
    cfg = get_config()
    path = predictions_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {path}. "
            "Run `python -m src.serving.s27_export_predictions` first."
        )

    meta_file = metadata_path()
    if not meta_file.exists():
        raise FileNotFoundError(
            f"Predictions metadata not found: {meta_file}. "
            "Re-run the export step."
        )
    pandas_auc = float(json.loads(meta_file.read_text(encoding="utf-8"))["auc"])

    logger.info("=" * 70)
    logger.info("D14 — Spark evaluation")
    logger.info("=" * 70)

    spark = get_spark(app_suffix="spark-eval")
    try:
        spark_result = compute_auc_spark(spark, path)
    finally:
        stop_spark(spark)

    abs_diff = abs(spark_result["auc"] - pandas_auc)
    passed = abs_diff <= MAX_ABS_DIFF

    logger.info(
        "Spark AUC %.6f vs pandas AUC %.6f — |diff| %.2e (%s)",
        spark_result["auc"],
        pandas_auc,
        abs_diff,
        "PASS" if passed else "FAIL",
    )

    result: dict[str, Any] = {
        "experiment": "D14",
        "description": "Spark SQL AUC vs pandas AUC on the same predictions",
        "spark_auc": spark_result["auc"],
        "pandas_auc": pandas_auc,
        "abs_diff": abs_diff,
        "tolerance": MAX_ABS_DIFF,
        "passed": passed,
        "n_rows": spark_result["n_rows"],
        "n_pos": spark_result["n_pos"],
        "n_neg": spark_result["n_neg"],
    }

    if write:
        out = cfg.paths.results_dir / "spark_eval.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out)

        md = cfg.paths.results_dir / "spark_eval.md"
        md.write_text(_render(result), encoding="utf-8")
        logger.info("Wrote %s", md)

        result["json"] = out
        result["markdown"] = md

    return result


def _render(result: dict[str, Any]) -> str:
    verdict = "PASS" if result["passed"] else "FAIL"
    return "\n".join([
        "# D14 — Spark evaluation parity",
        "",
        "AUC computed twice on the same 1,318 predictions: once with scikit-learn",
        "in the pandas path, once with Spark SQL using the rank-based",
        "Mann-Whitney formula.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Spark AUC | {result['spark_auc']:.6f} |",
        f"| pandas AUC | {result['pandas_auc']:.6f} |",
        f"| Absolute difference | {result['abs_diff']:.2e} |",
        f"| Tolerance | {result['tolerance']:.3f} |",
        f"| Rows | {result['n_rows']:,} ({result['n_pos']:,} pos / {result['n_neg']:,} neg) |",
        "",
        f"**Success criterion #11 (Spark AUC matches pandas within 0.001): {verdict}**",
        "",
        "> **Why the formula is written out.** Spark MLlib has an AUC evaluator,",
        "> but calling it would test MLlib rather than the aggregation. AUC has a",
        "> closed form in terms of ranks, and expressing it in SQL is what proves",
        "> the aggregation itself is correct. Ties are handled by averaging the",
        "> ranks a tie group spans, which `RANK` alone cannot do.",
        "",
    ])


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.evaluation.s28_spark_eval``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    result = run_spark_eval()

    print("\n" + "=" * 70)
    print("D14 — SPARK EVALUATION")
    print("=" * 70)
    print(f"  Spark AUC  : {result['spark_auc']:.6f}")
    print(f"  pandas AUC : {result['pandas_auc']:.6f}")
    print(f"  |diff|     : {result['abs_diff']:.2e}")
    print(f"  verdict    : {'PASS' if result['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
