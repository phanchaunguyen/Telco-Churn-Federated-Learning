"""Backfill MLflow with the experiment results already on disk.

Pipeline position: **EVALUATE → LOG (retrospective)**

Every experiment runner writes its results to ``results/E*.json`` but none of
them calls :func:`src.tracking.s14_mlflow.log_run`. The tracking module exists
and works, it was simply never wired in, so the MLflow UI stayed empty while
the JSON files accumulated.

Re-running E1-E7 just to populate MLflow would cost about two hours of compute
for numbers that are already computed. This module reads the JSON artifacts and
logs them instead, which is both faster and honest: the metrics in MLflow are
exactly the metrics in the report.

What gets logged
----------------
One MLflow run per experiment, tagged with the six required tags
(``experiment, partition, alpha, strategy, K, noise_multiplier``) so runs can
be filtered the way the spec asks for.

Usage
-----
>>> from src.tracking.s26_backfill_mlflow import backfill
>>> result = backfill()
>>> result["logged"]
7
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import get_config
from src.tracking.s14_mlflow import log_run, mlflow_reachable

logger = logging.getLogger(__name__)


# =============================================================================
# Readers — one per experiment shape
# =============================================================================
def _read(path: Path) -> dict[str, Any] | None:
    """Load a JSON artifact, returning ``None`` when it is absent."""
    if not path.exists():
        logger.warning("Skipping missing artifact: %s", path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _e0_runs(results_dir: Path) -> list[dict[str, Any]]:
    """E0 — IBM Churn Score baseline."""
    payload = _read(results_dir / "E0_ibm_baseline.json")
    if payload is None:
        return []
    return [{
        "experiment": "E0",
        "params": {
            "partition": "n/a",
            "alpha": "n/a",
            "strategy": "ibm_churn_score",
            "K": 0,
            "noise_multiplier": 0.0,
            "score_column": payload.get("score_column", "Churn Score"),
        },
        "metrics": {
            "auc": payload["auc"],
            "n_samples": payload["n_samples"],
            "n_churned": payload["n_churned"],
        },
    }]


def _e1_runs(results_dir: Path) -> list[dict[str, Any]]:
    """E1 — one run per fold, plus the aggregated summary."""
    runs: list[dict[str, Any]] = []

    for fold in range(5):
        payload = _read(results_dir / f"E1_fold_{fold}.json")
        if payload is None:
            continue
        fed = payload.get("federated", {})
        runs.append({
            "experiment": "E1",
            "params": {
                "partition": "iid",
                "alpha": "n/a",
                "strategy": "fedavg",
                "K": 10,
                "noise_multiplier": 0.0,
                "fold": fold,
                "seed": payload.get("seed", 42),
                "n_rounds": payload.get("n_rounds", 50),
            },
            "metrics": {
                "auc": fed.get("auc", float("nan")),
                "f1": fed.get("f1", float("nan")),
                "recall_at_10pct": fed.get("recall_at_10pct", float("nan")),
                "centralized_auc": payload.get("centralized", {}).get("auc", float("nan")),
                "local_only_auc": payload.get("local_only", {}).get("auc", float("nan")),
            },
        })

    return runs


def _e2_runs(results_dir: Path) -> list[dict[str, Any]]:
    """E2 — one run per partition."""
    payload = _read(results_dir / "E2_partition_impact.json")
    if payload is None:
        return []

    runs = []
    for run in payload.get("runs", []):
        partition = run["partition"]
        alpha = "n/a" if partition == "iid" else partition.rsplit("a", 1)[-1]
        runs.append({
            "experiment": "E2",
            "params": {
                "partition": partition,
                "alpha": alpha,
                "strategy": "fedavg",
                "K": 10,
                "noise_multiplier": 0.0,
                "n_rounds": run.get("n_rounds", 50),
            },
            "metrics": {
                "auc": run["auc"],
                "pr_auc": run.get("pr_auc", float("nan")),
                "recall_at_10pct": run.get("recall_at_10pct", float("nan")),
                "convergence_round": run.get("convergence_round", 0),
                "per_client_auc_std": run.get("per_client_auc_std", float("nan")),
            },
        })
    return runs


def _e3_runs(results_dir: Path) -> list[dict[str, Any]]:
    """E3 — one run per aggregation strategy."""
    payload = _read(results_dir / "E3_aggregation.json")
    if payload is None:
        return []

    runs = []
    for run in payload.get("runs", []):
        runs.append({
            "experiment": "E3",
            "params": {
                "partition": run.get("partition", "label_skew_a0.1"),
                "alpha": "0.1",
                "strategy": run["strategy"],
                "K": 10,
                "noise_multiplier": 0.0,
                "n_rounds": run.get("n_rounds", 50),
            },
            "metrics": {
                "auc": run["auc"],
                "pr_auc": run.get("pr_auc", float("nan")),
                "convergence_round": run.get("convergence_round", 0),
                "per_client_auc_std": run.get("per_client_auc_std", float("nan")),
            },
        })
    return runs


def _e4_runs(results_dir: Path) -> list[dict[str, Any]]:
    """E4 — one run per noise level."""
    payload = _read(results_dir / "E4_dp_sgd.json")
    if payload is None:
        return []

    runs = []
    for run in payload.get("runs", []):
        runs.append({
            "experiment": "E4",
            "params": {
                "partition": run.get("partition", "iid"),
                "alpha": "n/a",
                "strategy": "fedavg",
                "K": 10,
                "noise_multiplier": run["noise_multiplier"],
                "n_rounds": run.get("n_rounds", 50),
            },
            "metrics": {
                "auc": run["auc"],
                "pr_auc": run.get("pr_auc", float("nan")),
                "epsilon": run.get("epsilon", 0.0),
            },
        })
    return runs


def _e5_runs(results_dir: Path) -> list[dict[str, Any]]:
    """E5 — one run per strategy, covering both fairness views."""
    payload = _read(results_dir / "E5_fairness.json")
    if payload is None:
        return []

    runs = []
    for run in payload.get("runs", []):
        runs.append({
            "experiment": "E5",
            "params": {
                "partition": run.get("partition", "label_skew_a0.1"),
                "alpha": "0.1",
                "strategy": run["strategy"],
                "K": 10,
                "noise_multiplier": 0.0,
                "n_rounds": run.get("n_rounds", 50),
            },
            "metrics": {
                "auc": run["auc"],
                "client_auc_std": run.get("client_auc_std", float("nan")),
                "client_auc_min": run.get("client_auc_min", float("nan")),
                "reason_recall_std": run.get("reason_recall_std", float("nan")),
                "reason_recall_min": run.get("reason_recall_min", float("nan")),
            },
        })
    return runs


def _e6_runs(results_dir: Path) -> list[dict[str, Any]]:
    """E6 — one run per client count."""
    payload = _read(results_dir / "E6_scalability.json")
    if payload is None:
        return []

    runs = []
    for run in payload.get("runs", []):
        runs.append({
            "experiment": "E6",
            "params": {
                "partition": "iid",
                "alpha": "n/a",
                "strategy": "fedavg",
                "K": run["k"],
                "noise_multiplier": 0.0,
                "n_rounds": run.get("rounds", 50),
            },
            "metrics": {
                "total_seconds": run["total_seconds"],
                "sec_per_round": run["sec_per_round"],
                "sec_per_client_round": run["sec_per_client_round"],
                "sanity_auc_synth": run.get("sanity_auc_synth", float("nan")),
            },
        })
    return runs


def _e7_runs(results_dir: Path) -> list[dict[str, Any]]:
    """E7 — one run per sampling fraction."""
    payload = _read(results_dir / "E7_communication.json")
    if payload is None:
        return []

    runs = []
    for run in payload.get("runs", []):
        runs.append({
            "experiment": "E7",
            "params": {
                "partition": "iid",
                "alpha": "n/a",
                "strategy": "fedavg",
                "K": run.get("n_clients", 10),
                "noise_multiplier": 0.0,
                "fraction": run["fraction"],
                "n_rounds": run.get("rounds", 50),
            },
            "metrics": {
                "auc": run["auc_real"],
                "pr_auc": run.get("pr_auc_real", float("nan")),
                "mb_per_round": run["mb_per_round"],
                "cumulative_mb": run["cumulative_mb"],
            },
        })
    return runs


READERS = (
    _e0_runs,
    _e1_runs,
    _e2_runs,
    _e3_runs,
    _e4_runs,
    _e5_runs,
    _e6_runs,
    _e7_runs,
)


# =============================================================================
# Backfill
# =============================================================================
def collect_runs(results_dir: Path | None = None) -> list[dict[str, Any]]:
    """Gather every run from the JSON artifacts, without touching MLflow."""
    cfg = get_config()
    base = results_dir if results_dir is not None else cfg.paths.results_dir

    runs: list[dict[str, Any]] = []
    for reader in READERS:
        runs.extend(reader(base))
    return runs


def backfill(
    *,
    results_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Log every experiment artifact to MLflow.

    Parameters
    ----------
    results_dir:
        Override for the results directory. Defaults to the configured one.
    dry_run:
        Collect and report without writing to MLflow.

    Returns
    -------
    dict
        ``{collected, logged, by_experiment, mlflow_reachable}``
    """
    runs = collect_runs(results_dir)

    by_experiment: dict[str, int] = {}
    for run in runs:
        by_experiment[run["experiment"]] = by_experiment.get(run["experiment"], 0) + 1

    reachable = mlflow_reachable()
    logger.info(
        "Collected %s runs across %s experiments (MLflow %s)",
        len(runs),
        len(by_experiment),
        "reachable" if reachable else "unreachable — will use JSON fallback",
    )

    if dry_run:
        return {
            "collected": len(runs),
            "logged": 0,
            "by_experiment": by_experiment,
            "mlflow_reachable": reachable,
        }

    logged = 0
    for run in runs:
        log_run(
            experiment=run["experiment"],
            params=run["params"],
            metrics=run["metrics"],
        )
        logged += 1

    logger.info("Logged %s runs to MLflow", logged)
    return {
        "collected": len(runs),
        "logged": logged,
        "by_experiment": by_experiment,
        "mlflow_reachable": reachable,
    }


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.tracking.s26_backfill_mlflow``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Backfill MLflow from results JSON")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = backfill(dry_run=args.dry_run)

    print("\n" + "=" * 70)
    print("MLFLOW BACKFILL" + (" (dry run)" if args.dry_run else ""))
    print("=" * 70)
    print(f"  MLflow reachable : {result['mlflow_reachable']}")
    print(f"  collected        : {result['collected']}")
    print(f"  logged           : {result['logged']}")
    for name, count in sorted(result["by_experiment"].items()):
        print(f"    {name:<4} {count:>3} runs")


if __name__ == "__main__":
    main()
