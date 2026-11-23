"""E7 — communication cost: what does one federated round cost on the wire?

Pipeline position: **TRAIN (local) → AGGREGATE → MEASURE (bytes)**

The federated loop only ever moves model weights. This module turns that into
a number: how many bytes cross the network per round, and how much accumulates
over a full run, as the client sampling fraction varies.

The formula
-----------
For one round with ``n`` selected clients::

    bytes_per_round = 2 * |theta| * 4 * n

* ``|theta|`` = 18,049 parameters (the MLP's weight count)
* ``4`` = bytes per float32
* ``2`` = one upload (client -> server) plus one download (server -> client)

The factor of 2 is the part people forget. Every round sends the global model
down to each selected client and sends the updated weights back up, so the
wire cost is double the model size per client.

What is NOT counted
-------------------
Only weights. No features, no labels, no gradients, no per-sample statistics.
``num_examples`` is sent so the server can weight the average, but it is a
single integer per client and is excluded from the byte count as negligible.

Usage
-----
>>> from src.experiments.s20_e7_communication import run_e7
>>> result = run_e7(rounds=50)
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
from src.evaluation.s12_eval import evaluate_model
from src.features.s07_preprocess import encode, fit_local_scaler, fit_vocabulary
from src.fl.s11_server import run_sequential
from src.models.s08_mlp import build_model, count_parameters, set_weights

logger = logging.getLogger(__name__)

TARGET_COL = "churn"

# Client sampling fractions from the protocol
DEFAULT_FRACTIONS: list[float] = [0.25, 0.5, 1.0]

# float32
BYTES_PER_PARAM: int = 4

# Upload + download
DIRECTIONS: int = 2


# =============================================================================
# Helpers
# =============================================================================
def _partition_dir() -> Path:
    """Return the K=10 IID partition directory."""
    cfg = get_config()
    path = cfg.paths.synthetic_dir / "partitions_iid"
    if not path.exists():
        raise FileNotFoundError(f"Partition directory not found: {path}")
    return path


def _load_test_sets(vocab: Any) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return ``((X_real, y_real), (X_synth, y_synth))``.

    ``test_real`` is the primary evaluation set because it is the only one that
    says anything about the real population. ``synth_test`` is a sanity check
    that the model did not collapse.
    """
    cfg = get_config()
    train_real = pd.read_parquet(cfg.paths.train_real_parquet)
    test_real = pd.read_parquet(cfg.paths.test_real_parquet)
    synth_train = pd.read_parquet(cfg.paths.synth_train_parquet)
    synth_test = pd.read_parquet(cfg.paths.synth_test_parquet)

    real_scaler = fit_local_scaler(encode(train_real, vocab))
    X_real = real_scaler.transform(encode(test_real, vocab)).astype(np.float32)
    y_real = test_real[TARGET_COL].to_numpy(dtype=np.int64)

    synth_scaler = fit_local_scaler(encode(synth_train, vocab))
    X_synth = synth_scaler.transform(encode(synth_test, vocab)).astype(np.float32)
    y_synth = synth_test[TARGET_COL].to_numpy(dtype=np.int64)

    return (X_real, y_real), (X_synth, y_synth)


def bytes_per_round(n_params: int, n_selected: int) -> int:
    """Bytes moved in one round: upload + download for every selected client."""
    return DIRECTIONS * n_params * BYTES_PER_PARAM * n_selected


def _run_one(
    *,
    fraction: float,
    rounds: int,
    seed: int,
    vocab: Any,
    real: tuple[np.ndarray, np.ndarray],
    synth: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    """Run one sampling fraction and return its cost and quality."""
    cfg = get_config()
    part_dir = _partition_dir()

    t0 = time.perf_counter()
    history = run_sequential(
        n_rounds=rounds,
        num_clients=cfg.partitioning.n_clients,
        strategy_name="fedavg",
        partition="iid",
        partition_dir=part_dir,
        fraction_fit=fraction,
        seed=seed,
    )
    duration = time.perf_counter() - t0

    n_selected = history["n_selected_per_round"]
    n_params = count_parameters(build_model(history["input_dim"], seed=seed))

    per_round = bytes_per_round(n_params, n_selected)
    cumulative = per_round * rounds

    model = build_model(history["input_dim"], seed=seed)
    set_weights(model, history["final_weights"])
    X_real, y_real = real
    X_synth, y_synth = synth
    metrics_real = evaluate_model(model, X_real, y_real)
    metrics_synth = evaluate_model(model, X_synth, y_synth)

    result = {
        "fraction": fraction,
        "rounds": rounds,
        "seed": seed,
        "n_clients": cfg.partitioning.n_clients,
        "n_selected_per_round": n_selected,
        "n_params": n_params,
        "bytes_per_round": per_round,
        "mb_per_round": round(per_round / 1e6, 4),
        "cumulative_bytes": cumulative,
        "cumulative_mb": round(cumulative / 1e6, 2),
        "duration_sec": round(duration, 1),
        "auc_real": metrics_real["auc"],
        "pr_auc_real": metrics_real["pr_auc"],
        "recall_at_10pct_real": metrics_real["recall_at_10pct"],
        "auc_synth": metrics_synth["auc"],
    }

    logger.info(
        "fraction=%.2f: %s clients/round, %.3f MB/round, %.1f MB total, "
        "AUC real %.4f, %.0fs",
        fraction,
        n_selected,
        result["mb_per_round"],
        result["cumulative_mb"],
        result["auc_real"],
        duration,
    )
    return result


# =============================================================================
# E7
# =============================================================================
def run_e7(
    *,
    rounds: int | None = None,
    seed: int | None = None,
    fractions: list[float] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Sweep the client sampling fraction and report communication cost.

    Parameters
    ----------
    rounds:
        Federated rounds per fraction. Defaults to ``cfg.fl.n_rounds`` (50).
    fractions:
        Sampling fractions to sweep. Defaults to ``[0.25, 0.5, 1.0]``.

    Returns
    -------
    dict
        ``{runs, table, bytes_per_param, directions}``
    """
    cfg = get_config()
    n_rounds = rounds if rounds is not None else cfg.fl.n_rounds
    torch_seed = seed if seed is not None else cfg.project.seed
    fracs = fractions if fractions is not None else list(DEFAULT_FRACTIONS)

    vocab = fit_vocabulary(_partition_dir(), cfg.partitioning.n_clients)
    real, synth = _load_test_sets(vocab)

    logger.info("=" * 70)
    logger.info("E7 — communication cost (%s rounds, K=%s)", n_rounds, cfg.partitioning.n_clients)
    logger.info("=" * 70)

    runs = [
        _run_one(
            fraction=f,
            rounds=n_rounds,
            seed=torch_seed,
            vocab=vocab,
            real=real,
            synth=synth,
        )
        for f in fracs
    ]

    table = pd.DataFrame([
        {
            "fraction": r["fraction"],
            "n_selected_per_round": r["n_selected_per_round"],
            "mb_per_round": r["mb_per_round"],
            "cumulative_mb": r["cumulative_mb"],
            "auc_real": r["auc_real"],
            "pr_auc_real": r["pr_auc_real"],
            "recall_at_10pct_real": r["recall_at_10pct_real"],
            "auc_synth": r["auc_synth"],
        }
        for r in runs
    ])

    result = {
        "runs": runs,
        "table": table.to_dict(orient="records"),
        "bytes_per_param": BYTES_PER_PARAM,
        "directions": DIRECTIONS,
    }

    if write:
        out = cfg.paths.results_dir / "E7_communication.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote %s", out)

        md = cfg.paths.results_dir / "E7_communication.md"
        md.write_text(_render_e7(table, runs, n_rounds), encoding="utf-8")
        logger.info("Wrote %s", md)

    return result


def _render_e7(
    table: pd.DataFrame,
    runs: list[dict[str, Any]],
    rounds: int,
) -> str:
    n_params = runs[0]["n_params"] if runs else 0
    lines = [
        "# E7 — Communication cost",
        "",
        f"FedAvg, IID, K=10, {rounds} rounds. Scored on `test_real` (1,318 rows).",
        "",
        f"`bytes_per_round = 2 x {n_params:,} params x 4 bytes x n_selected`",
        "",
        "| Fraction | Clients/round | MB/round | Cumulative MB | AUC (real) | PR-AUC (real) | Recall@10% | AUC (synth) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"| {r['fraction']:.2f} | {int(r['n_selected_per_round'])} | "
            f"{r['mb_per_round']:.3f} | {r['cumulative_mb']:.1f} | "
            f"{r['auc_real']:.4f} | {r['pr_auc_real']:.4f} | "
            f"{r['recall_at_10pct_real']:.4f} | {r['auc_synth']:.4f} |"
        )

    # The interesting question is whether the cheapest setting is also the
    # worst. If AUC is flat across fractions, sampling less is free.
    if len(table) >= 2:
        best_auc = float(table["auc_real"].max())
        cheapest = table.loc[table["mb_per_round"].idxmin()]
        gap = (best_auc - float(cheapest["auc_real"])) / best_auc * 100
        verdict = "PASS" if gap <= 2.0 else "FAIL"
        lines += [
            "",
            f"**Success criterion (cheapest setting stays within 2% AUC): {verdict}** — "
            f"fraction={cheapest['fraction']:.2f} costs {cheapest['mb_per_round']:.3f} MB/round "
            f"and loses {gap:.2f}% AUC vs the best setting",
        ]

    lines += [
        "",
        "> **What is counted.** Model weights only: no features, no labels, no",
        "> gradients. `num_examples` is sent for the weighted average but is a",
        "> single integer per client, so it is excluded as negligible.",
        "",
        "> **Why the factor of 2.** Every round downloads the global model to each",
        "> selected client and uploads the updated weights back, so the wire cost",
        "> is double the model size per client.",
        "",
        "> **Note on fraction=0.25.** With K=10 the selection is",
        "> `max(2, round(10 * 0.25)) = 2` clients, not the 2-3 the protocol",
        "> sketched. The floor of 2 exists because FedAvg needs at least two",
        "> clients to average.",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.experiments.s20_e7_communication``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run E7 (communication cost)")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--fractions", type=float, nargs="+", default=None)
    args = parser.parse_args()

    result = run_e7(rounds=args.rounds, fractions=args.fractions)

    print("\n" + "=" * 70)
    print("E7 — COMMUNICATION COST")
    print("=" * 70)
    for row in result["table"]:
        print(
            f"  fraction={row['fraction']:.2f}  "
            f"{int(row['n_selected_per_round'])} clients  "
            f"{row['mb_per_round']:.3f} MB/round  "
            f"{row['cumulative_mb']:.1f} MB total  "
            f"AUC={row['auc_real']:.4f}"
        )


if __name__ == "__main__":
    main()
