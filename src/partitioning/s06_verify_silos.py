"""Silo verification gate — the last check before Layer 2 may start.

Pipeline position: **CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION → VERIFY**

This module is a hard gate. If any assertion fails, FL training must not begin,
because a leaked column or an overlapping silo would invalidate every
downstream experiment.

Checks
------
1. Every expected client file exists and is readable.
2. No silo contains a leakage column (Loại 1-6).
3. Every silo has exactly ``Customer ID`` + 29 features + target.
4. Silo IDs are pairwise disjoint.
5. Silo row counts sum to the expected total.
6. Every silo contains both classes.
7. No NaN anywhere.

Artifacts produced
------------------
``results/silo_verification.json``  per-strategy pass/fail with details.

Usage
-----
>>> from src.partitioning.s06_verify_silos import verify_all
>>> report = verify_all(write=True)
>>> report["passed"]
True
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_config
from src.data.s01_clean import DROPPED_COLS, FEATURE_COLS, ID_COL, TARGET_COL

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: list[str] = [ID_COL, *FEATURE_COLS, TARGET_COL]
EXPECTED_FEATURE_COUNT: int = len(FEATURE_COLS)


# =============================================================================
# Single strategy
# =============================================================================
def verify_strategy(
    part_dir: Path,
    k: int,
    *,
    expected_total_rows: int | None = None,
) -> dict[str, Any]:
    """Verify one partition directory and return a structured report."""
    errors: list[str] = []
    per_client: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    total_rows = 0

    for c in range(k):
        path = part_dir / f"client_{c}.parquet"
        if not path.exists():
            errors.append(f"client_{c}: file missing")
            continue

        df = pd.read_parquet(path)
        total_rows += len(df)

        # 1. Schema
        if list(df.columns) != EXPECTED_COLUMNS:
            errors.append(
                f"client_{c}: unexpected columns "
                f"(got {len(df.columns)}, expected {len(EXPECTED_COLUMNS)})"
            )

        # 2. Leakage
        leaked = [col for col in DROPPED_COLS if col in df.columns]
        if leaked:
            errors.append(f"client_{c}: leakage columns present {leaked}")

        # 3. Feature count
        present_features = [col for col in FEATURE_COLS if col in df.columns]
        if len(present_features) != EXPECTED_FEATURE_COUNT:
            errors.append(
                f"client_{c}: {len(present_features)} features, "
                f"expected {EXPECTED_FEATURE_COUNT}"
            )

        # 4. Disjointness
        ids = set(df[ID_COL])
        overlap = ids & seen_ids
        if overlap:
            errors.append(f"client_{c}: {len(overlap)} IDs overlap earlier clients")
        seen_ids |= ids

        # 5. Both classes
        classes = set(df[TARGET_COL].unique())
        if classes != {0, 1}:
            errors.append(f"client_{c}: classes {sorted(classes)}, expected [0, 1]")

        # 6. NaN
        n_nan = int(df.isna().sum().sum())
        if n_nan:
            errors.append(f"client_{c}: {n_nan} NaN cells")

        per_client.append({
            "client_id": c,
            "n_rows": int(len(df)),
            "n_churned": int(df[TARGET_COL].sum()),
            "churn_rate": round(float(df[TARGET_COL].mean()), 6),
            "n_features": len(present_features),
            "n_nan": n_nan,
        })

    # 7. Total row count
    if expected_total_rows is not None and total_rows != expected_total_rows:
        errors.append(
            f"total rows {total_rows:,} != expected {expected_total_rows:,}"
        )

    churn_rates = [c["churn_rate"] for c in per_client]
    report = {
        "partition_dir": str(part_dir),
        "k": k,
        "total_rows": total_rows,
        "unique_ids": len(seen_ids),
        "churn_rate_min": min(churn_rates) if churn_rates else None,
        "churn_rate_max": max(churn_rates) if churn_rates else None,
        "churn_rate_std": (
            round(float(pd.Series(churn_rates).std()), 6) if churn_rates else None
        ),
        "per_client": per_client,
        "errors": errors,
        "passed": len(errors) == 0,
    }

    if report["passed"]:
        logger.info(
            "verify_strategy: PASS (%s, k=%s, %s rows, churn %.3f..%.3f)",
            part_dir.name,
            k,
            f"{total_rows:,}",
            report["churn_rate_min"],
            report["churn_rate_max"],
        )
    else:
        logger.error("verify_strategy: FAIL (%s) — %s", part_dir.name, errors)

    return report


# =============================================================================
# All strategies
# =============================================================================
def verify_all(*, write: bool = True) -> dict[str, Any]:
    """Verify every partition directory produced by the partitioner."""
    cfg = get_config()
    synth_dir = cfg.paths.synthetic_dir

    expected_total = None
    if cfg.paths.synth_train_parquet.exists():
        expected_total = len(pd.read_parquet(cfg.paths.synth_train_parquet))

    results: dict[str, Any] = {}
    for spec in cfg.partitioning.strategies:
        tag = spec.name if spec.alpha is None else f"{spec.name}_a{spec.alpha}"
        part_dir = synth_dir / f"partitions_{tag}"
        if not part_dir.exists():
            results[tag] = {
                "passed": False,
                "errors": [f"partition directory missing: {part_dir}"],
            }
            continue
        results[tag] = verify_strategy(
            part_dir,
            cfg.partitioning.n_clients,
            expected_total_rows=expected_total,
        )

    overall = all(r["passed"] for r in results.values())
    report = {
        "passed": overall,
        "expected_total_rows": expected_total,
        "expected_feature_count": EXPECTED_FEATURE_COUNT,
        "expected_columns": EXPECTED_COLUMNS,
        "strategies": results,
    }

    if write:
        out_path = cfg.paths.silo_verification
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out_path)

    return report


def main() -> None:
    """CLI entry point: ``python -m src.partitioning.s06_verify_silos``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    report = verify_all(write=True)

    print("\n" + "=" * 74)
    print("SILO VERIFICATION GATE")
    print("=" * 74)
    print(f"  expected features : {report['expected_feature_count']}")
    print(f"  expected rows     : {report['expected_total_rows']:,}"
          if report["expected_total_rows"] else "  expected rows     : n/a")
    print("-" * 74)

    for tag, res in report["strategies"].items():
        status = "PASS" if res["passed"] else "FAIL"
        print(f"\n  [{status}] {tag}")
        if res.get("per_client"):
            print(f"    total rows   : {res['total_rows']:,}")
            print(f"    unique IDs   : {res['unique_ids']:,}")
            print(f"    churn range  : {res['churn_rate_min']:.4f} .. "
                  f"{res['churn_rate_max']:.4f} (std {res['churn_rate_std']:.4f})")
            sizes = [c["n_rows"] for c in res["per_client"]]
            print(f"    client sizes : min {min(sizes):,} / max {max(sizes):,}")
        for err in res.get("errors", []):
            print(f"    ERROR: {err}")

    print("\n" + "=" * 74)
    print(f"  RESULT: {'PASS — Layer 2 may start' if report['passed'] else 'FAIL — do not start Layer 2'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
