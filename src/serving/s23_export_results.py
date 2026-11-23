"""Push experiment results into postgres so Superset can read them.

Pipeline position: **EVALUATE → EXPORT (BI) → DASHBOARD**

Superset cannot read JSON files. It reads SQL tables. This module flattens the
experiment artifacts into three narrow tables, one per dashboard:

* ``e1_convergence`` — AUC per round, for the convergence dashboard
* ``e5_fairness``    — per-client and per-reason fairness, for the fairness one
* ``e4_privacy``     — AUC against epsilon, for the privacy curve

Why a separate table per dashboard
----------------------------------
Each dashboard asks a different question and needs a different grain. Forcing
them into one wide table would mean every chart filters out most of the rows,
and the SQL behind each chart would be harder to read than the chart itself.

Idempotency
-----------
Every table is dropped and recreated on each run. The source of truth is the
JSON in ``results/``, so a partial re-export can never leave stale rows behind.

Usage
-----
>>> from src.serving.s23_export_results import export_results
>>> export_results()          # writes to postgres
>>> export_results(dry_run=True)   # prints what would be written
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_config

logger = logging.getLogger(__name__)

# Connection string for the postgres service in docker-compose.yml. Superset
# points at the same database, so the tables appear without extra wiring.
DEFAULT_DSN: str = "postgresql+psycopg2://mlflow:mlflow@localhost:5432/mlflow"


# =============================================================================
# Builders — pure functions, testable without a database
# =============================================================================
def build_convergence_table() -> pd.DataFrame:
    """AUC per round for every run that recorded a curve.

    Only E2 stores a per-round curve. E1 writes final metrics per fold but not
    the round-by-round history, so it cannot contribute rows here — the
    dashboard shows E2's three partitions instead, which is the comparison the
    convergence view is actually for.

    Returns
    -------
    pandas.DataFrame
        Columns ``source, run, round, auc``.
    """
    cfg = get_config()
    rows: list[dict[str, Any]] = []

    e2 = cfg.paths.results_dir / "E2_partition_impact.json"
    if e2.exists():
        payload = json.loads(e2.read_text(encoding="utf-8"))
        for run in payload.get("runs", []):
            for rnd, auc in run.get("auc_curve", []):
                rows.append({
                    "source": "E2",
                    "run": run["partition"],
                    "round": int(rnd),
                    "auc": float(auc),
                })

    return pd.DataFrame(rows, columns=["source", "run", "round", "auc"])


def build_fairness_table() -> pd.DataFrame:
    """Per-client and per-reason fairness rows from E5.

    Returns
    -------
    pandas.DataFrame
        Columns ``view, strategy, label, value`` where ``view`` is
        ``client`` or ``reason``.
    """
    cfg = get_config()
    path = cfg.paths.results_dir / "E5_fairness.json"
    if not path.exists():
        return pd.DataFrame(columns=["view", "strategy", "label", "value"])

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    for run in payload.get("runs", []):
        strategy = run["strategy"]
        for client in run.get("per_client", []):
            rows.append({
                "view": "client",
                "strategy": strategy,
                "label": f"client_{client['client_id']}",
                "value": float(client["auc"]),
            })
        for reason in run.get("per_reason", []):
            rows.append({
                "view": "reason",
                "strategy": strategy,
                "label": reason["reason"],
                "value": float(reason["recall"]),
            })

    return pd.DataFrame(rows, columns=["view", "strategy", "label", "value"])


def build_privacy_table() -> pd.DataFrame:
    """AUC against epsilon from E4.

    Returns
    -------
    pandas.DataFrame
        Columns ``noise_multiplier, epsilon, auc, pr_auc``.
    """
    cfg = get_config()
    path = cfg.paths.results_dir / "E4_dp_sgd.json"
    if not path.exists():
        return pd.DataFrame(columns=["noise_multiplier", "epsilon", "auc", "pr_auc"])

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [
        {
            "noise_multiplier": float(r["noise_multiplier"]),
            "epsilon": float(r["epsilon"]),
            "auc": float(r["auc"]),
            "pr_auc": float(r["pr_auc"]),
        }
        for r in payload.get("runs", [])
    ]
    return pd.DataFrame(rows, columns=["noise_multiplier", "epsilon", "auc", "pr_auc"])


# =============================================================================
# Export
# =============================================================================
def export_results(
    *,
    dsn: str | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    """Write the three dashboard tables to postgres.

    Parameters
    ----------
    dsn:
        SQLAlchemy connection string. Defaults to the compose postgres.
    dry_run:
        Build the tables and report row counts without touching the database.

    Returns
    -------
    dict
        ``{table_name: row_count}``
    """
    tables = {
        "e1_convergence": build_convergence_table(),
        "e5_fairness": build_fairness_table(),
        "e4_privacy": build_privacy_table(),
    }
    counts = {name: len(df) for name, df in tables.items()}

    if dry_run:
        for name, df in tables.items():
            logger.info("[dry-run] %s: %s rows", name, len(df))
        return counts

    try:
        from sqlalchemy import create_engine
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sqlalchemy is required to export results; install with "
            "`pip install sqlalchemy psycopg2-binary`"
        ) from exc

    engine = create_engine(dsn or DEFAULT_DSN)
    for name, df in tables.items():
        if df.empty:
            logger.warning("Skipping %s: no rows (source JSON missing?)", name)
            continue
        df.to_sql(name, engine, if_exists="replace", index=False)
        logger.info("Wrote %s rows to %s", len(df), name)

    return counts


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.serving.s23_export_results``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Export results to postgres")
    parser.add_argument("--dsn", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    counts = export_results(dsn=args.dsn, dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("RESULTS EXPORTED" + (" (dry run)" if args.dry_run else ""))
    print("=" * 70)
    for name, n in counts.items():
        print(f"  {name:<20} {n:>6} rows")


if __name__ == "__main__":
    main()
