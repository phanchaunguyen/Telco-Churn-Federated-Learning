"""Spark session factory shared by every Spark module.

Keeping the builder in one place means the driver memory, shuffle partition
count, and log level are configured identically for cleaning, amplification,
partitioning, and evaluation. Divergent settings between stages would make the
benchmark numbers meaningless.

Usage
-----
>>> from src.data.s00_spark_session import get_spark
>>> spark = get_spark("clean")
>>> spark.version
'4.3.0'
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from src.config import REPO_ROOT, get_config

logger = logging.getLogger(__name__)

# Spark 4 ships with Java 17+; the host here runs Java 21. Nothing to pin.
_DEFAULT_LOG_LEVEL = "WARN"


def get_spark(
    app_suffix: str = "app",
    *,
    master: str | None = None,
    driver_memory: str | None = None,
    shuffle_partitions: int | None = None,
    extra_conf: dict[str, Any] | None = None,
    s3: bool = False,
) -> "SparkSession":  # noqa: F821 - imported lazily to keep import cost low
    """Build (or reuse) a SparkSession configured from ``configs/base.yaml``.

    Parameters
    ----------
    app_suffix:
        Appended to the configured app name so the Spark UI can distinguish
        concurrent stages.
    master, driver_memory, shuffle_partitions:
        Optional overrides. Defaults come from ``cfg.spark``.
    extra_conf:
        Additional ``spark.*`` settings.
    s3:
        When ``True``, pull the ``hadoop-aws`` bundle so ``s3a://`` paths work.
        Spark does not ship the S3A filesystem, so writing to MinIO fails with
        ``ClassNotFoundException: S3AFileSystem`` without it.
    """
    from pyspark.sql import SparkSession

    cfg = get_config()
    spark_cfg = cfg.spark

    resolved_master = master or spark_cfg.master
    resolved_memory = driver_memory or spark_cfg.driver_memory
    resolved_shuffle = shuffle_partitions or spark_cfg.shuffle_partitions

    # Spark needs a writable temp dir; the default /tmp is fine but pinning it
    # avoids surprises in containers with a read-only root.
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    # Executors must use the same interpreter as the driver, otherwise they
    # cannot import the project's dependencies (pandas, scipy, ...).
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    builder = (
        SparkSession.builder.appName(f"{spark_cfg.app_name}-{app_suffix}")
        .master(resolved_master)
        .config("spark.driver.memory", resolved_memory)
        .config("spark.sql.shuffle.partitions", str(resolved_shuffle))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
    )

    if s3:
        # Spark does not ship the S3A filesystem. Prefer jars vendored under
        # ``jars/`` (downloaded once, no network at run time); fall back to
        # Maven coordinates when they are absent.
        local_jars = sorted((REPO_ROOT / "jars").glob("*.jar"))
        if local_jars:
            builder = builder.config(
                "spark.jars", ",".join(str(p) for p in local_jars)
            )
            logger.info("Using %s local Hadoop jars from jars/", len(local_jars))
        else:
            hadoop_version = _bundled_hadoop_version()
            builder = builder.config(
                "spark.jars.packages",
                f"org.apache.hadoop:hadoop-aws:{hadoop_version},"
                f"org.apache.hadoop:hadoop-client-api:{hadoop_version},"
                f"org.apache.hadoop:hadoop-client-runtime:{hadoop_version}",
            )

    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel(_DEFAULT_LOG_LEVEL)

    logger.info(
        "SparkSession ready: version=%s master=%s driver_memory=%s shuffle_partitions=%s s3=%s",
        spark.version,
        resolved_master,
        resolved_memory,
        resolved_shuffle,
        s3,
    )
    return spark


def _bundled_hadoop_version() -> str:
    """Return the Hadoop version PySpark was built against."""
    try:
        from pyspark import SparkContext

        version = SparkContext._jvm.org.apache.spark.SPARK_VERSION  # noqa: SLF001
        del version
    except Exception:  # noqa: BLE001
        pass
    # PySpark 4.x bundles Hadoop 3.4.x; 3.5.x bundles 3.3.x.
    import pyspark

    major = int(pyspark.__version__.split(".")[0])
    return "3.4.1" if major >= 4 else "3.3.4"


def stop_spark(spark: "SparkSession") -> None:  # noqa: F821
    """Stop a session, swallowing the shutdown noise Spark emits."""
    try:
        spark.stop()
    except Exception:  # noqa: BLE001 - shutdown must never mask a real error
        logger.debug("Spark session already stopped", exc_info=True)
