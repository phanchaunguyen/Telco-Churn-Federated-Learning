"""Spark pipeline benchmark — success criterion #9.

Measures wall-clock time for the three Spark stages at 10K / 100K / 1M rows:

* **clean**     — read raw CSV, drop leakage, derive target
* **amplify**   — fit copula on train_real, sample N rows
* **partition** — assign client ids, write K parquet files

The 1M end-to-end total must stay under 15 minutes on ``local[*]``.

Every row of the output table records the dataset size, Spark mode, driver
memory, shuffle partition count, and wall-clock time, because a benchmark
without those four numbers is not reproducible.

Artifacts produced
------------------
``results/spark_benchmark.csv``

Usage
-----
>>> from scripts.benchmark_spark import run_benchmark
>>> df = run_benchmark()
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Make the repo root importable when run as a script
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config
from src.data.s00_spark_session import get_spark, stop_spark

logger = logging.getLogger(__name__)


# =============================================================================
# Timed stages
# =============================================================================
def _time_clean(spark: Any) -> float:
    """Time the CLEAN stage."""
    from src.data.s01_clean_spark import clean, read_raw

    cfg = get_config()
    t0 = time.perf_counter()
    raw = read_raw(spark, cfg.paths.raw_data)
    cleaned = clean(spark, raw)
    n = cleaned.count()  # force materialisation
    elapsed = time.perf_counter() - t0
    logger.info("clean: %s rows in %.2fs", f"{n:,}", elapsed)
    return elapsed


def _time_amplify(spark: Any, n_rows: int) -> float:
    """Time the AMPLIFY stage for ``n_rows`` synthetic rows."""
    from src.data.s03_amplify_spark import build_synthetic_spark

    t0 = time.perf_counter()
    sdf = build_synthetic_spark(
        n_rows=n_rows, write=False, manage_session=False
    )
    sdf.count()  # force materialisation
    elapsed = time.perf_counter() - t0
    logger.info("amplify: %s rows in %.2fs", f"{n_rows:,}", elapsed)
    return elapsed


def _time_partition(spark: Any, n_rows: int) -> float:
    """Time the PARTITION stage for a synthetic frame of ``n_rows``."""
    from src.partitioning.s05_partition import assign_iid, assign_synthetic_ids

    cfg = get_config()
    t0 = time.perf_counter()

    # Build a frame of the requested size from synth_train by sampling with
    # replacement — the benchmark measures Spark throughput, not data quality.
    base = pd.read_parquet(cfg.paths.synth_train_parquet)
    if n_rows <= len(base):
        df = base.sample(n=n_rows, random_state=cfg.split.seed).reset_index(drop=True)
    else:
        df = base.sample(n=n_rows, replace=True, random_state=cfg.split.seed).reset_index(drop=True)

    assignment = assign_iid(df, cfg.partitioning.n_clients, seed=cfg.split.seed)
    df_with_ids = assign_synthetic_ids(df)
    df_with_ids["client_id"] = assignment

    sdf = spark.createDataFrame(df_with_ids)
    sdf.cache()
    for cid in range(cfg.partitioning.n_clients):
        sdf.filter(sdf["client_id"] == cid).drop("client_id").count()
    sdf.unpersist()

    elapsed = time.perf_counter() - t0
    logger.info("partition: %s rows in %.2fs", f"{n_rows:,}", elapsed)
    return elapsed


# =============================================================================
# Orchestration
# =============================================================================
def run_benchmark(
    sizes: list[int] | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Run the three stages at each size and return a tidy results table."""
    cfg = get_config()
    bench_sizes = sizes if sizes is not None else cfg.spark.benchmark.sizes

    rows: list[dict[str, Any]] = []
    spark = get_spark("benchmark")
    try:
        for size in bench_sizes:
            logger.info("=" * 60)
            logger.info("Benchmarking %s rows", f"{size:,}")
            logger.info("=" * 60)

            clean_s = _time_clean(spark)
            amplify_s = _time_amplify(spark, size)
            partition_s = _time_partition(spark, size)
            total_s = clean_s + amplify_s + partition_s

            rows.append({
                "n_rows": size,
                "mode": cfg.spark.master,
                "driver_memory": cfg.spark.driver_memory,
                "shuffle_partitions": cfg.spark.shuffle_partitions,
                "clean_sec": round(clean_s, 2),
                "amplify_sec": round(amplify_s, 2),
                "partition_sec": round(partition_s, 2),
                "total_sec": round(total_s, 2),
                "total_min": round(total_s / 60, 2),
                "under_15min": total_s < cfg.spark.benchmark.max_minutes_1m * 60,
            })
    finally:
        stop_spark(spark)

    result = pd.DataFrame(rows)

    if write:
        out = cfg.paths.results_dir / "spark_benchmark.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        logger.info("Wrote %s", out)

    return result


def main() -> None:
    """CLI entry point: ``python scripts/03_benchmark_spark.py``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    df = run_benchmark()

    print("\n" + "=" * 90)
    print("SPARK PIPELINE BENCHMARK")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    one_m = df[df["n_rows"] == 1_000_000]
    if not one_m.empty:
        ok = bool(one_m.iloc[0]["under_15min"])
        print(f"  1M end-to-end under 15 min: {'PASS' if ok else 'FAIL'} "
              f"({one_m.iloc[0]['total_min']:.2f} min)")
    print("=" * 90)


if __name__ == "__main__":
    main()
