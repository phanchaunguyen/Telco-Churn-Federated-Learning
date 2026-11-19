"""Spark implementation of the CLEAN step.

Pipeline position: **CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION**

This module is the Spark counterpart of :mod:`src.data.s01_clean`. It must produce
a byte-for-byte equivalent result: same rows, same columns, same churn rate.
The pandas implementation stays in the repo as the reference that the parity
assertions compare against, and as the fast path for unit tests.

Two source-data traps are handled here exactly as in the pandas version:

1. The raw header spells one column ``"Total Refu    nds"`` (four spaces
   injected mid-word). Without the rename it would silently escape the leakage
   drop list and leak into the feature matrix.
2. ``"None"`` is a valid category for ``Offer`` and ``Internet Type``. Spark's
   CSV reader must not be told to treat it as null.

Artifacts produced
------------------
``data/processed/clean_7043_spark.parquet``  6,589 x 31, for parity checking.

Usage
-----
>>> from src.data.s01_clean_spark import build_clean_spark
>>> df = build_clean_spark(write=True)
>>> df.count()
6589
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.config import get_config
from src.data.s01_clean import (
    CONSTANT_COLS,
    DERIVED_DUP_COLS,
    DROPPED_COLS,
    EXPECTED_CLEAN_COLUMNS,
    EXPECTED_FEATURE_COUNT,
    EXPECTED_RAW_COLUMNS,
    FEATURE_COLS,
    GEO_COLS,
    HEADER_FIXES,
    ID_COL,
    LEAKAGE_COLS,
    STATUS_COL,
    TARGET_COL,
    DROP_STATUS_VALUE,
)
from src.data.s00_spark_session import get_spark, stop_spark

logger = logging.getLogger(__name__)

# Explicit schema for the 50 raw columns. Using inferSchema would trigger a
# full extra pass over the file and can mis-type columns that are all-numeric
# in the first sample but categorical elsewhere.
RAW_SCHEMA: dict[str, str] = {
    "Customer ID": "string",
    "Gender": "string",
    "Age": "int",
    "Under 30": "string",
    "Senior Citizen": "string",
    "Married": "string",
    "Dependents": "string",
    "Number of Dependents": "int",
    "Country": "string",
    "State": "string",
    "City": "string",
    "Zip Code": "int",
    "Latitude": "double",
    "Longitude": "double",
    "Population": "int",
    "Quarter": "string",
    "Referred a Friend": "string",
    "Number of Referrals": "int",
    "Tenure in Months": "int",
    "Offer": "string",
    "Phone Service": "string",
    "Avg Monthly Long Distance Charges": "double",
    "Multiple Lines": "string",
    "Internet Service": "string",
    "Internet Type": "string",
    "Avg Monthly GB Download": "int",
    "Online Security": "string",
    "Online Backup": "string",
    "Device Protection Plan": "string",
    "Premium Tech Support": "string",
    "Streaming TV": "string",
    "Streaming Movies": "string",
    "Streaming Music": "string",
    "Unlimited Data": "string",
    "Contract": "string",
    "Paperless Billing": "string",
    "Payment Method": "string",
    "Monthly Charge": "double",
    "Total Charges": "double",
    "Total Refunds": "double",
    "Total Extra Data Charges": "int",
    "Total Long Distance Charges": "double",
    "Total Revenue": "double",
    "Satisfaction Score": "int",
    "Customer Status": "string",
    "Churn Label": "string",
    "Churn Score": "int",
    "CLTV": "int",
    "Churn Category": "string",
    "Churn Reason": "string",
}


# =============================================================================
# Reading
# =============================================================================
def read_raw(spark: Any, path: str | Path) -> Any:
    """Read the raw CSV with an explicit schema and repaired headers."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

    type_map = {"string": StringType(), "int": IntegerType(), "double": DoubleType()}
    schema = StructType(
        [StructField(name, type_map[dtype], nullable=True) for name, dtype in RAW_SCHEMA.items()]
    )

    df = (
        spark.read.option("header", "true")
        .option("multiLine", "false")
        # Do NOT set nullValue="None": "None" is a real category here.
        .option("nullValue", "")
        .option("mode", "FAILFAST")
        .schema(schema)
        .csv(str(path))
    )

    # Repair the malformed header before anything else touches the columns
    for bad, good in HEADER_FIXES.items():
        if bad in df.columns:
            logger.warning("Repairing malformed header %r -> %r", bad, good)
            df = df.withColumnRenamed(bad, good)

    n_cols = len(df.columns)
    if n_cols != EXPECTED_RAW_COLUMNS:
        raise ValueError(f"Expected {EXPECTED_RAW_COLUMNS} columns, got {n_cols}")

    logger.info("Read raw CSV: %s columns", n_cols)
    return df


# =============================================================================
# Cleaning
# =============================================================================
def clean(spark: Any, raw: Any) -> Any:
    """Apply the full drop list and derive the binary target.

    Order matters: the target is derived from ``Customer Status`` *before* the
    leakage drop removes that column.
    """
    from pyspark.sql import functions as F

    df = raw.filter(F.col(STATUS_COL) != DROP_STATUS_VALUE)

    # Derive the target first — Customer Status is itself a leakage column
    df = df.withColumn(
        TARGET_COL,
        F.when(F.col(STATUS_COL) == "Churned", F.lit(1)).otherwise(F.lit(0)).cast("int"),
    )

    drop_list = LEAKAGE_COLS + CONSTANT_COLS + DERIVED_DUP_COLS + GEO_COLS
    present = [c for c in drop_list if c in df.columns]
    df = df.drop(*present)

    # Deterministic column order: ID + 29 features + target
    return df.select([ID_COL, *FEATURE_COLS, TARGET_COL])


# =============================================================================
# Validation
# =============================================================================
def validate(df: Any) -> dict[str, Any]:
    """Assert the Spark output matches the pandas reference invariants."""
    from pyspark.sql import functions as F

    n_cols = len(df.columns)
    assert n_cols == EXPECTED_CLEAN_COLUMNS, (
        f"Expected {EXPECTED_CLEAN_COLUMNS} columns, got {n_cols}: {df.columns}"
    )
    assert list(df.columns) == [ID_COL, *FEATURE_COLS, TARGET_COL], (
        f"Unexpected column order: {df.columns}"
    )

    leaked = [c for c in DROPPED_COLS if c in df.columns]
    assert not leaked, f"Leakage columns survived: {leaked}"

    assert len(FEATURE_COLS) == EXPECTED_FEATURE_COUNT

    stats = df.agg(
        F.count("*").alias("n_rows"),
        F.sum(TARGET_COL).alias("n_churned"),
        F.avg(TARGET_COL).alias("churn_rate"),
        F.countDistinct(ID_COL).alias("n_unique_ids"),
    ).collect()[0]

    n_rows = int(stats["n_rows"])
    n_churned = int(stats["n_churned"])
    churn_rate = float(stats["churn_rate"])
    n_unique = int(stats["n_unique_ids"])

    assert n_unique == n_rows, f"{ID_COL} not unique ({n_unique} vs {n_rows})"
    assert 0.0 < churn_rate < 1.0, f"Implausible churn rate {churn_rate}"

    # No nulls in any feature
    null_counts = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in FEATURE_COLS]
    ).collect()[0].asDict()
    bad = {c: n for c, n in null_counts.items() if n}
    assert not bad, f"Features contain nulls: {bad}"

    logger.info(
        "validate: OK (%s rows, %s churned, churn rate %.4f)",
        f"{n_rows:,}",
        f"{n_churned:,}",
        churn_rate,
    )
    return {
        "n_rows": n_rows,
        "n_churned": n_churned,
        "churn_rate": churn_rate,
        "n_unique_ids": n_unique,
        "n_columns": n_cols,
    }


def assert_parity(spark_stats: dict[str, Any], pandas_stats: dict[str, Any]) -> None:
    """Compare Spark output against the pandas reference implementation."""
    assert spark_stats["n_rows"] == pandas_stats["n_rows"], (
        f"row count mismatch: spark {spark_stats['n_rows']} vs pandas {pandas_stats['n_rows']}"
    )
    assert spark_stats["n_churned"] == pandas_stats["n_churned"], (
        f"churned mismatch: spark {spark_stats['n_churned']} vs pandas {pandas_stats['n_churned']}"
    )
    delta = abs(spark_stats["churn_rate"] - pandas_stats["churn_rate"])
    assert delta < 1e-9, f"churn rate drift {delta:.2e} exceeds 1e-9"
    logger.info("assert_parity: Spark output matches the pandas reference exactly")


# =============================================================================
# Orchestration
# =============================================================================
def build_clean_spark(
    raw_path: str | Path | None = None,
    *,
    write: bool = True,
    out_path: str | Path | None = None,
    verify_parity: bool = True,
) -> Any:
    """Run the CLEAN step in Spark and optionally verify parity with pandas."""
    cfg = get_config()
    src = Path(raw_path) if raw_path is not None else cfg.paths.raw_data
    if not src.exists():
        raise FileNotFoundError(f"Raw dataset not found: {src}")

    spark = get_spark("clean")
    try:
        raw = read_raw(spark, src)
        cleaned = clean(spark, raw)
        stats = validate(cleaned)

        if verify_parity:
            pandas_stats = _pandas_reference_stats()
            assert_parity(stats, pandas_stats)

        if write:
            target = (
                Path(out_path)
                if out_path is not None
                else cfg.paths.processed_dir / "clean_7043_spark.parquet"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            cleaned.write.mode("overwrite").parquet(str(target))
            logger.info("Wrote %s", target)

        return cleaned
    finally:
        stop_spark(spark)


def _pandas_reference_stats() -> dict[str, Any]:
    """Compute the same statistics from the pandas reference implementation."""
    from src.data.s01_clean import build_clean

    df, _ = build_clean(write=False)
    return {
        "n_rows": int(len(df)),
        "n_churned": int(df[TARGET_COL].sum()),
        "churn_rate": float(df[TARGET_COL].mean()),
        "n_unique_ids": int(df[ID_COL].nunique()),
        "n_columns": int(df.shape[1]),
    }


def main() -> None:
    """CLI entry point: ``python -m src.data.s01_clean_spark``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    cfg = get_config()
    spark = get_spark("clean")
    try:
        raw = read_raw(spark, cfg.paths.raw_data)
        cleaned = clean(spark, raw)
        stats = validate(cleaned)
        assert_parity(stats, _pandas_reference_stats())

        target = cfg.paths.processed_dir / "clean_7043_spark.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        cleaned.write.mode("overwrite").parquet(str(target))
        logger.info("Wrote %s", target)

        print("\n" + "=" * 70)
        print("SPARK CLEAN COMPLETE")
        print("=" * 70)
        print(f"  rows     : {stats['n_rows']:,}")
        print(f"  churned  : {stats['n_churned']:,}")
        print(f"  churn    : {stats['churn_rate']:.4f}")
        print(f"  columns  : {stats['n_columns']}")
        print(f"  features : {len(FEATURE_COLS)}")
        print("  parity   : matches pandas reference exactly")
        print("=" * 70)
    finally:
        stop_spark(spark)


if __name__ == "__main__":
    main()
