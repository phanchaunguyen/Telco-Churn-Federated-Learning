"""Spark implementation of the AMPLIFY step.

Pipeline position: **CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION**

Design: the copula is fitted on the driver with pandas (5,271 rows is trivial),
then Spark materialises the 1M synthetic rows in parallel. Fitting in Spark
would buy nothing — the parameter estimation is O(n·d²) on a tiny frame — while
sampling is embarrassingly parallel and is where the time actually goes.

The generator is fitted on ``train_real`` ONLY. ``test_real`` is never read
here, which is what keeps the TSTR evaluation honest.

Artifacts produced
------------------
``data/synthetic/synthetic_1m_spark.parquet``  1,000,000 x 30.

Usage
-----
>>> from src.data.spark_amplify import build_synthetic_spark
>>> df = build_synthetic_spark(n_rows=1_000_000, write=True)
>>> df.count()
1000000
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.amplify import (
    MAX_CHURN_RATE_DRIFT,
    fit_copula,
    sample_copula,
    validate_synthetic,
)
from src.data.clean import CATEGORICAL_COLS, FEATURE_COLS, TARGET_COL
from src.data.spark_session import get_spark, stop_spark

logger = logging.getLogger(__name__)

# Rows generated per Spark partition. 200k keeps each task's memory footprint
# around 50 MB while still giving the scheduler enough work to balance.
ROWS_PER_PARTITION: int = 200_000


# =============================================================================
# Parallel sampling
# =============================================================================
def sample_partition(
    state: dict[str, Any],
    n_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Generate ``n_rows`` synthetic rows. Runs inside a Spark task."""
    return sample_copula(state, n_rows, seed=seed)


def build_synthetic_spark(
    train_path: str | Path | None = None,
    *,
    n_rows: int | None = None,
    write: bool = True,
    out_path: str | Path | None = None,
    manage_session: bool = True,
) -> Any:
    """Fit on ``train_real``, sample ``n_rows`` in parallel, and write parquet.

    Parameters
    ----------
    manage_session:
        When ``True`` (default) the function creates and stops its own
        SparkSession. Set to ``False`` when the caller already owns a session
        and needs it alive after this call returns.
    """
    cfg = get_config()
    src = Path(train_path) if train_path is not None else cfg.paths.train_real_parquet
    if not src.exists():
        raise FileNotFoundError(
            f"train_real not found: {src}. Run `python -m src.data.split` first."
        )

    train = pd.read_parquet(src)
    logger.info("Loaded %s (%s rows x %s cols)", src, *train.shape)

    n = n_rows if n_rows is not None else cfg.amplification.n_rows
    state = fit_copula(train, seed=cfg.split.seed)

    spark = get_spark("amplify")
    try:
        n_partitions = max(1, int(np.ceil(n / ROWS_PER_PARTITION)))
        base = n // n_partitions
        remainder = n - base * n_partitions

        # Each partition gets its own seed so the draws are independent but the
        # whole run stays reproducible.
        sizes = [base + (1 if i < remainder else 0) for i in range(n_partitions)]
        seeds = [cfg.split.seed + i for i in range(n_partitions)]

        logger.info(
            "Sampling %s rows across %s partitions", f"{n:,}", n_partitions
        )

        # Broadcast the fitted state so every task has it without re-fitting
        state_bc = spark.sparkContext.broadcast(state)
        plan = [(size, seed) for size, seed in zip(sizes, seeds)]

        def _gen(part: Any) -> pd.DataFrame:
            # ``parallelize`` hands each task one element of ``plan``, which is
            # a (size, seed) tuple — not a list of tuples.
            size, seed = part
            return sample_partition(state_bc.value, size, seed)

        rdd = spark.sparkContext.parallelize(plan, numSlices=n_partitions)
        pdfs = rdd.map(_gen).collect()

        synthetic = pd.concat(pdfs, ignore_index=True)
        validate_synthetic(
            synthetic, expected_churn_rate=state["train_churn_rate"]
        )

        sdf = spark.createDataFrame(synthetic)
        logger.info("Spark DataFrame: %s rows x %s cols", sdf.count(), len(sdf.columns))

        if write:
            target = (
                Path(out_path)
                if out_path is not None
                else cfg.paths.synthetic_dir / "synthetic_1m_spark.parquet"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            sdf.write.mode("overwrite").parquet(str(target))
            logger.info("Wrote %s", target)

        return sdf
    finally:
        if manage_session:
            stop_spark(spark)


def main() -> None:
    """CLI entry point: ``python -m src.data.spark_amplify``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    cfg = get_config()
    spark = get_spark("amplify")
    try:
        sdf = build_synthetic_spark(write=True, manage_session=False)
        n_rows = sdf.count()
        n_cols = len(sdf.columns)

        print("\n" + "=" * 70)
        print("SPARK AMPLIFY COMPLETE")
        print("=" * 70)
        print(f"  rows     : {n_rows:,}")
        print(f"  columns  : {n_cols}")
        print(f"  features : {len(FEATURE_COLS)}")
        print(f"  target   : {TARGET_COL}")
        print("=" * 70)
    finally:
        stop_spark(spark)


if __name__ == "__main__":
    main()
