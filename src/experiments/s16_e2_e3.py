"""E2 — partition impact, and E3 — aggregation robustness.

Pipeline position: **TRAIN → EVALUATE → REPORT**

E2 answers RQ1/H1: how much does non-IID-ness cost?
E3 answers RQ2/H2: which aggregation strategy survives severe label skew?

Both share the same machinery, so they live in one module. The only difference
is which variable is swept:

    E2: partition in {iid, label_skew_a0.5, label_skew_a0.1}, strategy = fedavg
    E3: strategy in {fedavg, fedprox, fedavgm}, partition = label_skew_a0.1

Protocol (docs/10_experiment_protocol.md section 1.2)
-----------------------------------------------------
Fold 0 only, 50 rounds, scored on ``test_real`` with bootstrap CIs. E1 already
established that the split barely matters (centralized std 0.0020 across five
folds), so spending five folds on every E2/E3 cell would cost hours for no
extra signal.

Convergence metric
------------------
E2 also reports the round at which each run first reaches 95% of its own final
AUC. A partition that needs 40 rounds to get where IID gets in 15 is worse for
practical deployment even if the final AUC matches.

Usage
-----
>>> from src.experiments.s16_e2_e3 import run_e2, run_e3
>>> e2 = run_e2()
>>> e3 = run_e3()
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
    evaluate_per_client,
    evaluate_with_ci,
)
from src.features.s07_preprocess import (
    encode,
    fit_local_scaler,
    fit_vocabulary,
)
from src.fl.s11_server import run_sequential
from src.models.s08_mlp import build_model, set_weights

logger = logging.getLogger(__name__)

# A run counts as converged once its smoothed AUC is within this absolute
# distance of the smoothed final AUC. An absolute tolerance is used instead of
# a fraction because the AUC curve rises steeply then flattens: 95% of the
# final value is reached in round 1, which would report every run as converged
# immediately and make the metric useless for comparing partitions.
CONVERGENCE_TOLERANCE: float = 0.005

# Moving-average window applied before measuring convergence. Per-round AUC is
# computed on 5 clients' local silos, so it bounces by ±0.005 from client
# sampling alone; without smoothing the "first round within tolerance" is
# decided by noise rather than by learning.
CONVERGENCE_SMOOTHING: int = 5


# =============================================================================
# Shared helpers
# =============================================================================
def _partition_dir(name: str) -> Path:
    """Resolve a top-level partition directory by short name."""
    cfg = get_config()
    path = cfg.paths.synthetic_dir / f"partitions_{name}"
    if not path.exists():
        raise FileNotFoundError(f"Partition directory not found: {path}")
    return path


def _load_test_set(vocab: Any) -> tuple[np.ndarray, np.ndarray]:
    """Encode ``test_real`` and scale it with the canonical ``train_real`` scaler."""
    cfg = get_config()
    test_df = pd.read_parquet(cfg.paths.test_real_parquet)
    train_df = pd.read_parquet(cfg.paths.train_real_parquet)

    scaler = fit_local_scaler(encode(train_df, vocab))
    X_test = scaler.transform(encode(test_df, vocab)).astype(np.float32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int64)
    return X_test, y_test


def _smooth(values: list[float], window: int = CONVERGENCE_SMOOTHING) -> list[float]:
    """Centred moving average, shrinking the window at the edges.

    A centred window is used so the smoothed value at round ``r`` reflects
    rounds on both sides of ``r``; a trailing window would systematically lag
    the curve and report convergence later than it actually happened.
    """
    n = len(values)
    if n == 0 or window <= 1:
        return list(values)

    half = window // 2
    out: list[float] = []
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = [v for v in values[lo:hi] if np.isfinite(v)]
        out.append(float(np.mean(chunk)) if chunk else float("nan"))
    return out


def convergence_round(
    aucs: list[tuple[int, float]],
    *,
    tolerance: float = CONVERGENCE_TOLERANCE,
    window: int = CONVERGENCE_SMOOTHING,
) -> int:
    """First round whose smoothed AUC is within ``tolerance`` of the best one.

    The reference is the *best* smoothed AUC, not the final one. A run that
    peaks and then diverges would otherwise report convergence at round 1,
    because its final value is low enough that the opening rounds already sit
    within tolerance of it. Measuring against the peak reports when the run
    first reached the quality it was capable of.

    Returns the total number of rounds when the run never gets there, which
    means the model was still improving at the end.
    """
    if not aucs:
        return 0

    rounds = [r for r, _ in aucs]
    smoothed = _smooth([a for _, a in aucs], window)

    finite = [v for v in smoothed if np.isfinite(v)]
    if not finite:
        return len(aucs)

    target = max(finite) - tolerance
    for rnd, value in zip(rounds, smoothed, strict=True):
        if np.isfinite(value) and value >= target:
            return rnd
    return len(aucs)


def _run_one(
    *,
    partition: str,
    strategy: str,
    n_rounds: int,
    seed: int,
    vocab: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Run one FL configuration and return its metrics."""
    cfg = get_config()
    part_dir = _partition_dir(partition)

    t0 = time.perf_counter()
    history = run_sequential(
        n_rounds=n_rounds,
        num_clients=cfg.partitioning.n_clients,
        strategy_name=strategy,
        partition=partition,
        partition_dir=part_dir,
        seed=seed,
    )
    duration = time.perf_counter() - t0

    model = build_model(X_test.shape[1], seed=seed)
    set_weights(model, history["final_weights"])
    metrics = evaluate_with_ci(model, X_test, y_test, seed=seed)

    per_client = evaluate_per_client(model, part_dir, vocab, cfg.partitioning.n_clients)

    aucs = history["metrics_distributed"]["auc"]
    conv = convergence_round(aucs)

    result = {
        "partition": partition,
        "strategy": strategy,
        "n_rounds": n_rounds,
        "seed": seed,
        "duration_sec": round(duration, 1),
        "convergence_round": conv,
        "final_round_auc": float(aucs[-1][1]) if aucs else float("nan"),
        "per_client_auc_mean": float(per_client["auc"].mean()),
        "per_client_auc_std": float(per_client["auc"].std()),
        "per_client_auc_min": float(per_client["auc"].min()),
        "per_client_auc_max": float(per_client["auc"].max()),
        "auc_curve": [[int(r), float(a)] for r, a in aucs],
        **metrics,
    }

    logger.info(
        "%s / %s: AUC %.4f [%.4f, %.4f], converged at round %s, per-client std %.4f, %.0fs",
        partition,
        strategy,
        metrics["auc"],
        metrics["auc_ci_lower"],
        metrics["auc_ci_upper"],
        conv,
        result["per_client_auc_std"],
        duration,
    )
    return result


# =============================================================================
# E2 — partition impact
# =============================================================================
def run_e2(
    *,
    n_rounds: int | None = None,
    seed: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Compare IID vs alpha=0.5 vs alpha=0.1 under FedAvg.

    Returns
    -------
    dict
        ``{runs, table, iid_gap_pct, alpha05_gap_pct, alpha01_gap_pct}``
    """
    cfg = get_config()
    rounds = n_rounds if n_rounds is not None else cfg.fl.n_rounds
    torch_seed = seed if seed is not None else cfg.project.seed

    partitions = ["iid", "label_skew_a0.5", "label_skew_a0.1"]
    vocab = fit_vocabulary(_partition_dir("iid"), cfg.partitioning.n_clients)
    X_test, y_test = _load_test_set(vocab)

    logger.info("=" * 70)
    logger.info("E2 — partition impact (%s rounds, FedAvg)", rounds)
    logger.info("=" * 70)

    runs = [
        _run_one(
            partition=p,
            strategy="fedavg",
            n_rounds=rounds,
            seed=torch_seed,
            vocab=vocab,
            X_test=X_test,
            y_test=y_test,
        )
        for p in partitions
    ]

    table = pd.DataFrame([
        {
            "partition": r["partition"],
            "auc": r["auc"],
            "auc_ci_lower": r["auc_ci_lower"],
            "auc_ci_upper": r["auc_ci_upper"],
            "pr_auc": r["pr_auc"],
            "recall_at_10pct": r["recall_at_10pct"],
            "convergence_round": r["convergence_round"],
            "per_client_auc_std": r["per_client_auc_std"],
        }
        for r in runs
    ])

    iid_auc = float(table.loc[table["partition"] == "iid", "auc"].iloc[0])
    gaps = {
        f"{row['partition']}_gap_pct": (iid_auc - row["auc"]) / iid_auc * 100
        for _, row in table.iterrows()
    }

    result = {"runs": runs, "table": table.to_dict(orient="records"), **gaps}

    if write:
        out = cfg.paths.results_dir / "E2_partition_impact.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote %s", out)

        md = cfg.paths.results_dir / "E2_partition_impact.md"
        md.write_text(_render_e2(table, gaps), encoding="utf-8")
        logger.info("Wrote %s", md)

    return result


def _render_e2(table: pd.DataFrame, gaps: dict[str, float]) -> str:
    lines = [
        "# E2 — Partition impact",
        "",
        "FedAvg, 50 rounds, scored on `test_real` (1,318 real rows).",
        "",
        "| Partition | AUC | 95% CI | PR-AUC | Recall@10% | Converged at round | Per-client AUC std | Gap vs IID |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        gap = gaps.get(f"{r['partition']}_gap_pct", 0.0)
        lines.append(
            f"| {r['partition']} | {r['auc']:.4f} | "
            f"[{r['auc_ci_lower']:.4f}, {r['auc_ci_upper']:.4f}] | "
            f"{r['pr_auc']:.4f} | {r['recall_at_10pct']:.4f} | "
            f"{r['convergence_round']} | {r['per_client_auc_std']:.4f} | {gap:+.2f}% |"
        )

    a01 = gaps.get("label_skew_a0.1_gap_pct")
    if a01 is not None:
        verdict = "PASS" if a01 > 0 else "FAIL"
        lines += [
            "",
            f"**Success criterion #3 (non-IID degrades AUC): {verdict}** — "
            f"alpha=0.1 costs {a01:.2f}% vs IID",
        ]
    return "\n".join(lines) + "\n"


# =============================================================================
# E3 — aggregation robustness
# =============================================================================
def run_e3(
    *,
    n_rounds: int | None = None,
    seed: int | None = None,
    partition: str = "label_skew_a0.1",
    write: bool = True,
) -> dict[str, Any]:
    """Compare FedAvg vs FedProx vs FedAvgM under severe label skew.

    Returns
    -------
    dict
        ``{runs, table, fedprox_gain_pct, fedavgm_gain_pct}``
    """
    cfg = get_config()
    rounds = n_rounds if n_rounds is not None else cfg.fl.n_rounds
    torch_seed = seed if seed is not None else cfg.project.seed

    strategies = ["fedavg", "fedprox", "fedavgm"]
    vocab = fit_vocabulary(_partition_dir("iid"), cfg.partitioning.n_clients)
    X_test, y_test = _load_test_set(vocab)

    logger.info("=" * 70)
    logger.info("E3 — aggregation robustness (%s rounds, partition=%s)", rounds, partition)
    logger.info("=" * 70)

    runs = [
        _run_one(
            partition=partition,
            strategy=s,
            n_rounds=rounds,
            seed=torch_seed,
            vocab=vocab,
            X_test=X_test,
            y_test=y_test,
        )
        for s in strategies
    ]

    table = pd.DataFrame([
        {
            "strategy": r["strategy"],
            "auc": r["auc"],
            "auc_ci_lower": r["auc_ci_lower"],
            "auc_ci_upper": r["auc_ci_upper"],
            "pr_auc": r["pr_auc"],
            "recall_at_10pct": r["recall_at_10pct"],
            "convergence_round": r["convergence_round"],
            "per_client_auc_std": r["per_client_auc_std"],
            "per_client_auc_min": r["per_client_auc_min"],
        }
        for r in runs
    ])

    fedavg_auc = float(table.loc[table["strategy"] == "fedavg", "auc"].iloc[0])
    gains = {
        f"{row['strategy']}_gain_pct": (row["auc"] - fedavg_auc) / fedavg_auc * 100
        for _, row in table.iterrows()
    }

    result = {"runs": runs, "table": table.to_dict(orient="records"), **gains}

    if write:
        out = cfg.paths.results_dir / "E3_aggregation.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote %s", out)

        md = cfg.paths.results_dir / "E3_aggregation.md"
        md.write_text(_render_e3(table, gains, partition), encoding="utf-8")
        logger.info("Wrote %s", md)

    return result


def _render_e3(table: pd.DataFrame, gains: dict[str, float], partition: str) -> str:
    lines = [
        "# E3 — Aggregation robustness",
        "",
        f"Partition `{partition}`, 50 rounds, scored on `test_real` (1,318 real rows).",
        "",
        "| Strategy | AUC | 95% CI | PR-AUC | Recall@10% | Converged at round | Per-client AUC std | Per-client AUC min | Gain vs FedAvg |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        gain = gains.get(f"{r['strategy']}_gain_pct", 0.0)
        lines.append(
            f"| {r['strategy']} | {r['auc']:.4f} | "
            f"[{r['auc_ci_lower']:.4f}, {r['auc_ci_upper']:.4f}] | "
            f"{r['pr_auc']:.4f} | {r['recall_at_10pct']:.4f} | "
            f"{r['convergence_round']} | {r['per_client_auc_std']:.4f} | "
            f"{r['per_client_auc_min']:.4f} | {gain:+.2f}% |"
        )

    best_gain = max(
        gains.get("fedprox_gain_pct", -999),
        gains.get("fedavgm_gain_pct", -999),
    )
    verdict = "PASS" if best_gain > 0 else "FAIL"
    lines += [
        "",
        f"**Success criterion #4 (FedProx/FedAvgM beat FedAvg): {verdict}** — "
        f"best alternative gains {best_gain:+.2f}%",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.experiments.s16_e2_e3``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run E2 and/or E3")
    parser.add_argument("--which", choices=["e2", "e3", "both"], default="both")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.which in ("e2", "both"):
        e2 = run_e2(n_rounds=args.rounds, seed=args.seed)
        print("\n" + "=" * 100)
        print("E2 — PARTITION IMPACT")
        print("=" * 100)
        print(pd.DataFrame(e2["table"]).to_string(index=False))
        print("=" * 100)

    if args.which in ("e3", "both"):
        e3 = run_e3(n_rounds=args.rounds, seed=args.seed)
        print("\n" + "=" * 100)
        print("E3 — AGGREGATION ROBUSTNESS")
        print("=" * 100)
        print(pd.DataFrame(e3["table"]).to_string(index=False))
        print("=" * 100)


if __name__ == "__main__":
    main()
