"""E1 — FL gap: centralized vs federated vs local-only.

Pipeline position: **TRAIN → EVALUATE → REPORT**

Answers RQ1 / H1: what fraction of centralized performance does FL retain when
it never sees the data in one place?

Three models, one test set
--------------------------
* **centralized** — trains on all 900,000 synthetic rows at once (upper bound)
* **federated**   — trains on 10 disjoint silos, aggregates weights (the subject)
* **local-only**  — each client trains alone, no aggregation (lower bound)

All three are scored on ``test_real``, the 1,318 real rows held out before
amplification. Synthetic test data is never used here.

Fold protocol
-------------
Each fold re-splits ``clean_7043`` with a different seed, then re-amplifies and
re-partitions. Reusing synthetic data across folds would leak: the Copula of
fold 0 has already seen the rows that become fold 1's test set.

Fold 0 uses seed 42, matching the artifacts already on disk, so it can reuse
them. Folds 1-4 use seeds 43-46 and rebuild everything.

Usage
-----
>>> from src.experiments.s15_e1_fl_gap import run_fold, aggregate_folds
>>> result = run_fold(fold_id=0, n_rounds=50)
>>> table = aggregate_folds([0, 1, 2, 3, 4])
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.data.s01_clean import TARGET_COL
from src.evaluation.s12_eval import (
    build_comparison_table,
    evaluate_with_ci,
    predict_proba,
)
from src.features.s07_preprocess import (
    encode,
    fit_local_scaler,
    fit_vocabulary,
    load_client,
)
from src.models.s08_mlp import set_weights

logger = logging.getLogger(__name__)

# Fold 0 reuses the artifacts already on disk (seed 42)
FOLD_SEEDS: dict[int, int] = {0: 42, 1: 43, 2: 44, 3: 45, 4: 46}


# =============================================================================
# Fold artifacts
# =============================================================================
def fold_paths(fold_id: int) -> dict[str, Path]:
    """Return the artifact paths for one fold.

    Fold 0 points at the existing top-level artifacts; folds 1-4 get their own
    sub-directories so nothing is overwritten.
    """
    cfg = get_config()
    if fold_id == 0:
        return {
            "train_real": cfg.paths.train_real_parquet,
            "test_real": cfg.paths.test_real_parquet,
            "synthetic": cfg.paths.synthetic_parquet,
            "synth_train": cfg.paths.synth_train_parquet,
            "partition_dir": cfg.paths.synthetic_dir / "partitions_iid",
        }

    base_proc = cfg.paths.processed_dir / f"fold_{fold_id}"
    base_syn = cfg.paths.synthetic_dir / f"fold_{fold_id}"
    return {
        "train_real": base_proc / "train_real.parquet",
        "test_real": base_proc / "test_real.parquet",
        "synthetic": base_syn / "synthetic_1m.parquet",
        "synth_train": base_syn / "synth_train.parquet",
        "partition_dir": base_syn / "partitions_iid",
    }


def build_fold_artifacts(fold_id: int) -> dict[str, Path]:
    """Create the split, synthetic data, and partitions for a non-zero fold.

    Fold 0 is a no-op because its artifacts already exist.
    """
    paths = fold_paths(fold_id)
    if fold_id == 0:
        logger.info("Fold 0 reuses existing artifacts")
        return paths

    if paths["partition_dir"].exists():
        logger.info("Fold %s artifacts already present — reusing", fold_id)
        return paths

    from src.data.s02_split import split_real, split_synthetic
    from src.data.s03_amplify import build_synthetic
    from src.partitioning.s05_partition import build_partitions

    cfg = get_config()
    seed = FOLD_SEEDS[fold_id]

    # 1. Re-split clean_7043 with this fold's seed
    clean = pd.read_parquet(cfg.paths.clean_parquet)
    train, test = split_real(clean, test_size=cfg.split.test_size, seed=seed)
    paths["train_real"].parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(paths["train_real"], index=False)
    test.to_parquet(paths["test_real"], index=False)
    logger.info("Fold %s split: train %s, test %s", fold_id, len(train), len(test))

    # 2. Amplify from this fold's train_real only
    synthetic = build_synthetic(
        train_path=paths["train_real"],
        n_rows=cfg.amplification.n_rows,
        write=False,
    )
    paths["synthetic"].parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_parquet(paths["synthetic"], index=False)

    # 3. Split 90/10
    synth_train, _ = split_synthetic(
        synthetic, test_size=cfg.split.synthetic_test_size, seed=seed
    )
    synth_train.to_parquet(paths["synth_train"], index=False)

    # 4. Partition IID
    build_partitions(
        "iid",
        k=cfg.partitioning.n_clients,
        seed=seed,
        synthetic_path=paths["synth_train"],
        write=True,
        out_dir=paths["partition_dir"].parent,
    )
    logger.info("Fold %s artifacts ready", fold_id)
    return paths


# =============================================================================
# One fold
# =============================================================================
def run_fold(
    fold_id: int,
    *,
    n_rounds: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Run E1 for one fold and return the three-model comparison.

    Returns
    -------
    dict
        ``{fold_id, seed, centralized, federated, local_only, table}``
    """
    from src.baselines.s13_centralized import train_centralized
    from src.baselines.s13_local_only import train_local_only
    from src.fl.s11_server import run_sequential

    cfg = get_config()
    rounds = n_rounds if n_rounds is not None else cfg.fl.n_rounds
    seed = FOLD_SEEDS[fold_id]
    paths = build_fold_artifacts(fold_id)

    t_start = time.perf_counter()
    logger.info("=" * 70)
    logger.info("E1 fold %s (seed %s, %s rounds)", fold_id, seed, rounds)
    logger.info("=" * 70)

    # --- Shared vocabulary and canonical test set -------------------------
    vocab = fit_vocabulary(paths["partition_dir"], cfg.partitioning.n_clients)

    test_df = pd.read_parquet(paths["test_real"])
    X_test_raw = encode(test_df, vocab)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int64)

    train_real_df = pd.read_parquet(paths["train_real"])
    canonical_scaler = fit_local_scaler(encode(train_real_df, vocab))
    X_test = canonical_scaler.transform(X_test_raw).astype(np.float32)

    # --- 1. Centralized ---------------------------------------------------
    logger.info("[1/3] Centralized baseline")
    central = train_centralized(
        n_epochs=rounds,
        seed=seed,
        vocab=vocab,
        write=False,
    )
    central_metrics = central["metrics"]

    # --- 2. Federated -----------------------------------------------------
    logger.info("[2/3] Federated (%s rounds)", rounds)
    fl_history = run_sequential(
        n_rounds=rounds,
        num_clients=cfg.partitioning.n_clients,
        strategy_name="fedavg",
        partition=f"fold_{fold_id}/partitions_iid" if fold_id else "iid",
        seed=seed,
    )

    from src.models.s08_mlp import build_model

    fl_model = build_model(X_test.shape[1], seed=seed)
    set_weights(fl_model, fl_history["final_weights"])
    fl_metrics = evaluate_with_ci(fl_model, X_test, y_test, seed=seed)

    # --- 3. Local-only ----------------------------------------------------
    logger.info("[3/3] Local-only baseline")
    local_df = train_local_only(
        partition=f"fold_{fold_id}/partitions_iid" if fold_id else "iid",
        n_epochs=rounds,
        seed=seed,
        vocab=vocab,
        write=False,
    )
    local_metrics = {
        "auc": float(local_df["auc"].mean()),
        "auc_std": float(local_df["auc"].std()),
        "auc_min": float(local_df["auc"].min()),
        "auc_max": float(local_df["auc"].max()),
        "f1": float(local_df["f1"].mean()),
        "recall_at_10pct": float(local_df["recall_at_10pct"].mean()),
    }

    # --- Comparison -------------------------------------------------------
    table = build_comparison_table({
        "centralized": central_metrics,
        "federated": fl_metrics,
        "local_only": local_metrics,
    })

    duration = time.perf_counter() - t_start
    result = {
        "fold_id": fold_id,
        "seed": seed,
        "n_rounds": rounds,
        "centralized": central_metrics,
        "federated": fl_metrics,
        "local_only": local_metrics,
        "table": table.to_dict(orient="records"),
        "duration_sec": round(duration, 1),
    }

    logger.info("=" * 70)
    logger.info("E1 fold %s complete in %.1fs", fold_id, duration)
    logger.info(
        "  centralized AUC %.4f | federated AUC %.4f | local-only AUC %.4f",
        central_metrics["auc"],
        fl_metrics["auc"],
        local_metrics["auc"],
    )
    logger.info("  FL gap: %.2f%%", table.loc[1, "gap_vs_centralized_pct"])
    logger.info("=" * 70)

    if write:
        out = cfg.paths.results_dir / f"E1_fold_{fold_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote %s", out)

    return result


# =============================================================================
# Aggregation across folds
# =============================================================================
def aggregate_folds(fold_ids: list[int], *, write: bool = True) -> pd.DataFrame:
    """Combine per-fold results into a mean ± std table.

    Returns
    -------
    pandas.DataFrame
        One row per model with ``auc_mean, auc_std, auc_min, auc_max,
        gap_mean_pct``.
    """
    cfg = get_config()
    rows: list[dict[str, Any]] = []

    for fold_id in fold_ids:
        path = cfg.paths.results_dir / f"E1_fold_{fold_id}.json"
        if not path.exists():
            logger.warning("Missing fold result: %s", path)
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload["table"]:
            rows.append({
                "fold": fold_id,
                "model": entry["model"],
                "auc": entry["auc"],
                "f1": entry["f1"],
                "recall_at_10pct": entry["recall_at_10pct"],
                "gap_vs_centralized_pct": entry["gap_vs_centralized_pct"],
            })

    if not rows:
        raise RuntimeError(f"No fold results found for {fold_ids}")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("model")
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_min=("auc", "min"),
            auc_max=("auc", "max"),
            f1_mean=("f1", "mean"),
            recall_mean=("recall_at_10pct", "mean"),
            gap_mean_pct=("gap_vs_centralized_pct", "mean"),
            n_folds=("fold", "nunique"),
        )
        .reset_index()
    )

    if write:
        out = cfg.paths.results_dir / "E1_fl_gap.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out, index=False)
        logger.info("Wrote %s", out)

        md = cfg.paths.results_dir / "E1_fl_gap.md"
        md.write_text(_render_markdown(summary, fold_ids), encoding="utf-8")
        logger.info("Wrote %s", md)

    return summary


def _render_markdown(summary: pd.DataFrame, fold_ids: list[int]) -> str:
    """Render the E1 summary as a markdown table for the report."""
    lines = [
        "# E1 — FL gap",
        "",
        f"Folds: {fold_ids} · Test set: `test_real` (1,318 real rows)",
        "",
        "| Model | AUC (mean ± std) | AUC min | AUC max | F1 | Recall@10% | Gap vs centralized |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['model']} | {r['auc_mean']:.4f} ± {r['auc_std']:.4f} | "
            f"{r['auc_min']:.4f} | {r['auc_max']:.4f} | {r['f1_mean']:.4f} | "
            f"{r['recall_mean']:.4f} | {r['gap_mean_pct']:.2f}% |"
        )

    fl_gap = summary.loc[summary["model"] == "federated", "gap_mean_pct"]
    if not fl_gap.empty:
        gap = float(fl_gap.iloc[0])
        verdict = "PASS" if gap < 5.0 else "FAIL"
        lines += [
            "",
            f"**Success criterion #2 (FL gap < 5%): {verdict}** — measured {gap:.2f}%",
        ]
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.experiments.s15_e1_fl_gap``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run E1 (FL gap)")
    parser.add_argument("--fold-ids", default="0", help="Comma-separated fold ids")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--aggregate", action="store_true", help="Only aggregate existing folds")
    args = parser.parse_args()

    fold_ids = [int(x) for x in args.fold_ids.split(",") if x.strip()]

    if args.aggregate:
        summary = aggregate_folds(fold_ids)
        print("\n" + "=" * 78)
        print("E1 — FL GAP (aggregated)")
        print("=" * 78)
        print(summary.to_string(index=False))
        print("=" * 78)
        return

    for fold_id in fold_ids:
        result = run_fold(fold_id, n_rounds=args.rounds)
        print("\n" + "=" * 78)
        print(f"E1 — FOLD {fold_id}")
        print("=" * 78)
        print(pd.DataFrame(result["table"]).to_string(index=False))
        print("=" * 78)

    if len(fold_ids) > 1:
        summary = aggregate_folds(fold_ids)
        print("\n" + "=" * 78)
        print("E1 — AGGREGATED")
        print("=" * 78)
        print(summary.to_string(index=False))
        print("=" * 78)


if __name__ == "__main__":
    main()
