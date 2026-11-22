"""E4 — differential privacy utility cost.

Pipeline position: **TRAIN (local, DP-SGD) → AGGREGATE → EVALUATE**

Sweeps the DP-SGD noise multiplier and reports the privacy/utility trade-off:
how much AUC is lost as epsilon shrinks. The non-private run (``sigma=0``) is
the reference point, so the curve always starts at the FedAvg baseline.

Why noise multiplier and not epsilon
------------------------------------
Epsilon is an *output* of the accountant, not an input. Fixing epsilon would
require solving for sigma per client per round, and because clients hold
different numbers of samples the resulting sigma would differ across clients —
which breaks the assumption that all clients add comparable noise. Sweeping
sigma keeps the mechanism identical everywhere and lets epsilon fall out of the
composition.

Usage
-----
>>> from src.experiments.s17_e4_dp import run_e4
>>> result = run_e4(n_rounds=20)
>>> result["table"]
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
from src.evaluation.s12_eval import evaluate_with_ci
from src.features.s07_preprocess import fit_vocabulary
from src.fl.s11_server import run_sequential
from src.models.s08_mlp import build_model, set_weights

logger = logging.getLogger(__name__)

# Target column name in the processed parquet artifacts
TARGET_COL = "churn"


# =============================================================================
# Helpers
# =============================================================================
def _partition_dir(partition: str) -> Path:
    """Resolve a partition name to its directory."""
    cfg = get_config()
    return cfg.paths.synthetic_dir / f"partitions_{partition}"


def _load_test_set(vocab: Any) -> tuple[np.ndarray, np.ndarray]:
    """Encode ``test_real`` and scale it with the canonical ``train_real`` scaler."""
    from src.features.s07_preprocess import encode, fit_local_scaler

    cfg = get_config()
    test_df = pd.read_parquet(cfg.paths.test_real_parquet)
    train_df = pd.read_parquet(cfg.paths.train_real_parquet)

    scaler = fit_local_scaler(encode(train_df, vocab))
    X_test = scaler.transform(encode(test_df, vocab)).astype(np.float32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int64)
    return X_test, y_test


def _run_one(
    *,
    partition: str,
    noise_multiplier: float,
    n_rounds: int,
    seed: int,
    vocab: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Run one DP-SGD configuration and return its metrics."""
    cfg = get_config()
    part_dir = _partition_dir(partition)

    t0 = time.perf_counter()
    history = run_sequential(
        n_rounds=n_rounds,
        num_clients=cfg.partitioning.n_clients,
        strategy_name="fedavg",
        partition=partition,
        partition_dir=part_dir,
        noise_multiplier=noise_multiplier,
        seed=seed,
    )
    duration = time.perf_counter() - t0

    model = build_model(X_test.shape[1], seed=seed)
    set_weights(model, history["final_weights"])
    metrics = evaluate_with_ci(model, X_test, y_test, seed=seed)

    epsilons = [e for e in history.get("epsilons", []) if np.isfinite(e)]
    result = {
        "partition": partition,
        "noise_multiplier": noise_multiplier,
        "n_rounds": n_rounds,
        "seed": seed,
        "duration_sec": round(duration, 1),
        "epsilon": float(max(epsilons)) if epsilons else 0.0,
        "final_round_auc": float(history["metrics_distributed"]["auc"][-1][1]),
        **metrics,
    }

    logger.info(
        "sigma=%.1f: AUC %.4f [%.4f, %.4f], eps=%.3f, %.0fs",
        noise_multiplier,
        metrics["auc"],
        metrics["auc_ci_lower"],
        metrics["auc_ci_upper"],
        result["epsilon"],
        duration,
    )
    return result


# =============================================================================
# E4
# =============================================================================
def run_e4(
    *,
    n_rounds: int | None = None,
    seed: int | None = None,
    partition: str = "iid",
    noise_multipliers: list[float] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Sweep the DP-SGD noise multiplier and report AUC vs epsilon.

    Parameters
    ----------
    n_rounds:
        Federated rounds per configuration. Defaults to ``cfg.fl.n_rounds``.
    partition:
        Which silo layout to train on. IID is the default because E4 isolates
        the privacy cost, not the partition effect (that is E2).
    noise_multipliers:
        Sigma values to sweep. Defaults to
        ``cfg.privacy.sweep_noise_multiplier`` = ``[0.0, 0.5, 1.0, 2.0]``.

    Returns
    -------
    dict
        ``{runs, table, baseline_auc, utility_loss_pct}``
    """
    cfg = get_config()
    rounds = n_rounds if n_rounds is not None else cfg.fl.n_rounds
    torch_seed = seed if seed is not None else cfg.project.seed
    sigmas = (
        noise_multipliers
        if noise_multipliers is not None
        else list(cfg.privacy.sweep_noise_multiplier)
    )

    vocab = fit_vocabulary(_partition_dir(partition), cfg.partitioning.n_clients)
    X_test, y_test = _load_test_set(vocab)

    logger.info("=" * 70)
    logger.info("E4 — DP-SGD utility cost (%s rounds, partition=%s)", rounds, partition)
    logger.info("=" * 70)

    runs = [
        _run_one(
            partition=partition,
            noise_multiplier=sigma,
            n_rounds=rounds,
            seed=torch_seed,
            vocab=vocab,
            X_test=X_test,
            y_test=y_test,
        )
        for sigma in sigmas
    ]

    table = pd.DataFrame([
        {
            "noise_multiplier": r["noise_multiplier"],
            "epsilon": r["epsilon"],
            "auc": r["auc"],
            "auc_ci_lower": r["auc_ci_lower"],
            "auc_ci_upper": r["auc_ci_upper"],
            "pr_auc": r["pr_auc"],
            "recall_at_10pct": r["recall_at_10pct"],
        }
        for r in runs
    ])

    baseline = float(table.loc[table["noise_multiplier"] == 0.0, "auc"].iloc[0])
    losses = {
        f"sigma_{row['noise_multiplier']}_loss_pct": (baseline - row["auc"])
        / baseline
        * 100
        for _, row in table.iterrows()
    }

    result = {
        "runs": runs,
        "table": table.to_dict(orient="records"),
        "baseline_auc": baseline,
        **losses,
    }

    if write:
        out = cfg.paths.results_dir / "E4_dp_sgd.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote %s", out)

        md = cfg.paths.results_dir / "E4_dp_sgd.md"
        md.write_text(_render_e4(table, losses, baseline, partition), encoding="utf-8")
        logger.info("Wrote %s", md)

    return result


def _render_e4(
    table: pd.DataFrame,
    losses: dict[str, float],
    baseline: float,
    partition: str,
) -> str:
    lines = [
        "# E4 — Differential privacy utility cost",
        "",
        f"Partition `{partition}`, FedAvg with DP-SGD, scored on `test_real` (1,318 rows).",
        "",
        "| Noise sigma | Epsilon (delta=1e-5) | AUC | 95% CI | PR-AUC | Recall@10% | AUC loss vs non-DP |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        loss = losses.get(f"sigma_{r['noise_multiplier']}_loss_pct", 0.0)
        eps = "inf" if r["noise_multiplier"] == 0.0 else f"{r['epsilon']:.2f}"
        lines.append(
            f"| {r['noise_multiplier']:.1f} | {eps} | {r['auc']:.4f} | "
            f"[{r['auc_ci_lower']:.4f}, {r['auc_ci_upper']:.4f}] | "
            f"{r['pr_auc']:.4f} | {r['recall_at_10pct']:.4f} | {loss:+.2f}% |"
        )

    # The success criterion is that the curve is monotone: more noise must not
    # improve utility. A non-monotone curve means the sweep is too short or the
    # noise is being cancelled by something else.
    dp_rows = table[table["noise_multiplier"] > 0.0].sort_values("noise_multiplier")
    monotone = bool(
        len(dp_rows) < 2
        or all(
            dp_rows["auc"].iloc[i] >= dp_rows["auc"].iloc[i + 1]
            for i in range(len(dp_rows) - 1)
        )
    )
    verdict = "PASS" if monotone else "FAIL"
    worst = max(
        (losses.get(f"sigma_{s}_loss_pct", 0.0) for s in table["noise_multiplier"]),
        default=0.0,
    )
    lines += [
        "",
        f"**Success criterion #5 (privacy costs utility, monotonically): {verdict}** — "
        f"worst-case AUC loss {worst:+.2f}% at the highest noise level",
        "",
        f"Non-DP reference AUC: {baseline:.4f}",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.experiments.s17_e4_dp``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run E4 (DP-SGD sweep)")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--partition", default="iid")
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=None,
        help="Noise multipliers to sweep (default: config)",
    )
    args = parser.parse_args()

    result = run_e4(
        n_rounds=args.rounds,
        partition=args.partition,
        noise_multipliers=args.sigmas,
    )

    print("\n" + "=" * 70)
    print("E4 — DP-SGD UTILITY COST")
    print("=" * 70)
    for row in result["table"]:
        eps = "inf" if row["noise_multiplier"] == 0.0 else f"{row['epsilon']:.2f}"
        print(
            f"  sigma={row['noise_multiplier']:.1f}  eps={eps:>6}  "
            f"AUC={row['auc']:.4f}  PR-AUC={row['pr_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
