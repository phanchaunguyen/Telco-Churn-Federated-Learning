"""Spark implementation of the PARTITION step, writing client silos to MinIO.

Pipeline position: **CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION → VERIFY**

The partitioning logic itself lives in :mod:`src.partitioning.partitioner` and
is reused here unchanged — this module only adds the Spark write path. Keeping
one implementation of the assignment algorithm means the pandas and Spark
outputs cannot drift apart.

Two write targets are supported:

* **MinIO** (``s3a://clients/client_{i}/features.parquet``) when the service is
  reachable and ``use_minio=True``.
* **Local parquet** (``data/synthetic/partitions_{tag}/client_{i}.parquet``)
  otherwise. This is the fallback that keeps the pipeline runnable on a laptop
  without Docker.

Artifacts produced
------------------
``s3a://clients/client_{0..9}/features.parquet``  (or the local equivalent)
``results/partition_metadata.json``               per-client size and churn rate

Usage
-----
>>> from src.partitioning.spark_partition import build_partitions_spark
>>> build_partitions_spark(strategy="iid", use_minio=False)
"""

from __future__ import annotations

import json
import logging
import socket
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.config import get_config
from src.data.clean import FEATURE_COLS, ID_COL, TARGET_COL
from src.data.spark_session import get_spark, stop_spark
from src.partitioning.partitioner import (
    assign_dirichlet,
    assign_iid,
    assign_synthetic_ids,
    validate_partitions,
)

logger = logging.getLogger(__name__)

Strategy = Literal["iid", "label_skew"]


# =============================================================================
# MinIO availability
# =============================================================================
def minio_reachable(host: str = "localhost", port: int = 9000, timeout: float = 1.0) -> bool:
    """Return ``True`` when the MinIO S3 endpoint accepts connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _configure_s3a(spark: Any, endpoint: str, access_key: str, secret_key: str) -> None:
    """Point Spark's ``s3a://`` filesystem at MinIO."""
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()  # noqa: SLF001
    hadoop_conf.set("fs.s3a.endpoint", f"http://{endpoint}")
    hadoop_conf.set("fs.s3a.access.key", access_key)
    hadoop_conf.set("fs.s3a.secret.key", secret_key)
    hadoop_conf.set("fs.s3a.path.style.access", "true")
    hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
    hadoop_conf.set(
        "fs.s3a.aws.credentials.provider",
        "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
    )


# =============================================================================
# Orchestration
# =============================================================================
def build_partitions_spark(
    strategy: Strategy,
    *,
    alpha: float | None = None,
    k: int | None = None,
    seed: int | None = None,
    synthetic_path: str | Path | None = None,
    use_minio: bool = False,
    write: bool = True,
    manage_session: bool = True,
) -> dict[str, Any]:
    """Partition ``synth_train`` and write one parquet per client.

    Parameters
    ----------
    use_minio:
        Defaults to ``False``. Writing to MinIO requires the ``hadoop-aws``
        bundle plus the AWS SDK v2 (533 MB), and loading those jars adds
        minutes to every Spark start-up. The local parquet layout is
        byte-identical, so the default path keeps the pipeline fast. Set to
        ``True`` when demonstrating the MinIO silo layout end to end.

    Returns
    -------
    dict
        The partition metadata (sizes, churn rates, target location).
    """
    cfg = get_config()
    n_clients = k if k is not None else cfg.partitioning.n_clients
    random_seed = seed if seed is not None else cfg.split.seed

    src = (
        Path(synthetic_path)
        if synthetic_path is not None
        else cfg.paths.synth_train_parquet
    )
    if not src.exists():
        raise FileNotFoundError(
            f"synth_train not found: {src}. Run the synthetic split first."
        )

    df = pd.read_parquet(src)
    logger.info("Loaded %s (%s rows x %s cols)", src, *df.shape)

    # --- Assignment (shared with the pandas partitioner) -------------------
    if strategy == "iid":
        assignment = assign_iid(df, n_clients, seed=random_seed)
    elif strategy == "label_skew":
        if alpha is None:
            raise ValueError("alpha is required for strategy='label_skew'")
        assignment = assign_dirichlet(df, n_clients, alpha, seed=random_seed)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    metadata = validate_partitions(df, assignment, n_clients, strategy=strategy)
    if alpha is not None:
        metadata["alpha"] = alpha
    metadata["seed"] = random_seed

    df_with_ids = assign_synthetic_ids(df)
    df_with_ids["client_id"] = assignment

    tag = strategy if alpha is None else f"{strategy}_a{alpha}"

    # --- Decide the write target ------------------------------------------
    target_minio = use_minio and minio_reachable()
    if use_minio and not target_minio:
        logger.warning(
            "MinIO not reachable on localhost:9000 — falling back to local parquet"
        )

    spark = get_spark(f"partition-{tag}", s3=target_minio)
    try:
        if target_minio:
            _configure_s3a(
                spark,
                cfg.minio.endpoint,
                cfg.minio.access_key,
                cfg.minio.secret_key,
            )

        sdf = spark.createDataFrame(df_with_ids)
        sdf.cache()

        if write:
            for cid in range(n_clients):
                client_df = sdf.filter(sdf["client_id"] == cid).drop("client_id")
                if target_minio:
                    uri = f"s3a://{cfg.minio.bucket_prefix}/client_{cid}/features.parquet"
                    client_df.write.mode("overwrite").parquet(uri)
                else:
                    local = cfg.paths.synthetic_dir / f"partitions_{tag}" / f"client_{cid}.parquet"
                    local.parent.mkdir(parents=True, exist_ok=True)
                    client_df.write.mode("overwrite").parquet(str(local))
            logger.info(
                "Wrote %s client silos to %s",
                n_clients,
                "MinIO" if target_minio else "local parquet",
            )

        metadata["target"] = "minio" if target_minio else "local"
        metadata["tag"] = tag

        if write:
            meta_path = cfg.paths.partition_metadata
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            existing: dict[str, Any] = {}
            if meta_path.exists():
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            existing[f"{tag}_spark"] = metadata
            meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            logger.info("Updated %s", meta_path)

        sdf.unpersist()
        return metadata
    finally:
        if manage_session:
            stop_spark(spark)


def main() -> None:
    """CLI entry point: ``python -m src.partitioning.spark_partition``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    cfg = get_config()

    print("\n" + "=" * 74)
    print("SPARK PARTITIONING")
    print("=" * 74)
    print(f"  MinIO reachable : {minio_reachable()}")

    for spec in cfg.partitioning.strategies:
        meta = build_partitions_spark(spec.name, alpha=spec.alpha, write=True)
        tag = spec.name if spec.alpha is None else f"{spec.name} a={spec.alpha}"
        print(f"\n  {tag}  -> {meta['target']}")
        print(f"    sizes       : {meta['sizes']}")
        print(f"    churn rates : {[round(r, 4) for r in meta['churn_rates']]}")
        print(f"    total rows  : {sum(meta['sizes']):,}")

    print("\n" + "=" * 74)


if __name__ == "__main__":
    main()
