"""E5 — fairness of the federated model.

Pipeline position: **AGGREGATE → EVALUATE (fairness)**

Two complementary views of fairness:

* **E5.1 — per-client.** Score the global model on each silo separately. A
  model that is accurate on average but useless on the smallest silo is not
  deployable, so the *worst* client matters more than the mean.
* **E5.2 — per-stratum.** Score the global model within each ``Churn Category``
  (Price / Competitor / Dissatisfaction / Attitude / Other). This asks whether
  the model is equally good at catching customers who leave for different
  reasons.

The two views answer different questions and can disagree: a model can be fair
across silos while being systematically worse at detecting price-driven churn.

Caveat that must travel with E5.2
---------------------------------
``test_real`` has 374 churned rows spread over 5 categories, so each stratum
holds roughly 75 positives. AUC on 75 positives has a 95% CI of about ±0.10.
Differences smaller than that between strategies are noise, not signal.

Usage
-----
>>> from src.experiments.s18_e5_fairness import run_e5
>>> result = run_e5(n_rounds=50)
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
from src.evaluation.s12_eval import (
    evaluate_per_client,
    evaluate_per_stratum,
    evaluate_with_ci,
)
from src.features.s07_preprocess import encode, fit_local_scaler, fit_vocabulary
from src.fl.s11_server import run_sequential
from src.models.s08_mlp import build_model, set_weights

logger = logging.getLogger(__name__)

TARGET_COL = "churn"
ID_COL = "Customer ID"
STRATUM_COL = "Churn Category"

# Strata smaller than this get AUC = nan; AUC on a handful of rows is noise
MIN_STRATUM_SAMPLES: int = 30


# =============================================================================
# Helpers
# =============================================================================
def _partition_dir(partition: str) -> Path:
    """Resolve a partition name to its directory."""
    cfg = get_config()
    path = cfg.paths.synthetic_dir / f"partitions_{partition}"
    if not path.exists():
        raise FileNotFoundError(f"Partition directory not found: {path}")
    return path


def _load_test_set(vocab: Any) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Encode ``test_real`` and return the raw frame alongside for joining."""
    cfg = get_config()
    test_df = pd.read_parquet(cfg.paths.test_real_parquet)
    train_df = pd.read_parquet(cfg.paths.train_real_parquet)

    scaler = fit_local_scaler(encode(train_df, vocab))
    X_test = scaler.transform(encode(test_df, vocab)).astype(np.float32)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int64)
    return X_test, y_test, test_df


def load_strata(test_df: pd.DataFrame) -> np.ndarray:
    """Join ``Churn Category`` back from the EDA frame, aligned to ``test_df``.

    ``Churn Category`` is only populated for churned customers, so non-churners
    get the literal string ``"None"``. That is intentional: the stratum
    ``"None"`` is the non-churner group, and its AUC measures how well the
    model ranks the negatives against everything else.

    Raises
    ------
    KeyError
        If a test row has no match in the EDA frame, which would silently
        misalign the strata array.
    """
    cfg = get_config()
    eda = pd.read_parquet(cfg.paths.eda_parquet, columns=[ID_COL, STRATUM_COL])

    merged = test_df[[ID_COL]].merge(eda, on=ID_COL, how="left")
    if merged[STRATUM_COL].isna().any():
        missing = int(merged[STRATUM_COL].isna().sum())
        raise KeyError(
            f"{missing} test rows have no {STRATUM_COL} in the EDA frame; "
            "the join would misalign strata"
        )
    return merged[STRATUM_COL].astype(str).to_numpy()


def _run_one(
    *,
    partition: str,
    strategy: str,
    n_rounds: int,
    seed: int,
    vocab: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    strata: np.ndarray,
) -> dict[str, Any]:
    """Run one configuration and collect both fairness views."""
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

    overall = evaluate_with_ci(model, X_test, y_test, seed=seed)
    per_client = evaluate_per_client(model, part_dir, vocab, cfg.partitioning.n_clients)
    per_stratum = evaluate_per_stratum(
        model, X_test, y_test, strata, min_samples=MIN_STRATUM_SAMPLES
    )

    result = {
        "partition": partition,
        "strategy": strategy,
        "n_rounds": n_rounds,
        "seed": seed,
        "duration_sec": round(duration, 1),
        **overall,
        "per_client": per_client.to_dict(orient="records"),
        "per_stratum": per_stratum.to_dict(orient="records"),
        "client_auc_mean": float(per_client["auc"].mean()),
        "client_auc_std": float(per_client["auc"].std()),
        "client_auc_min": float(per_client["auc"].min()),
        "client_auc_max": float(per_client["auc"].max()),
        "client_auc_gap": float(per_client["auc"].max() - per_client["auc"].min()),
        "stratum_auc_std": float(per_stratum["auc"].std()),
        "stratum_auc_min": float(per_stratum["auc"].min()),
        "stratum_auc_gap": float(per_stratum["auc"].max() - per_stratum["auc"].min()),
    }

    logger.info(
        "%s / %s: AUC %.4f | client AUC %.4f±%.4f (min %.4f) | "
        "stratum AUC std %.4f (min %.4f), %.0fs",
        partition,
        strategy,
        overall["auc"],
        result["client_auc_mean"],
        result["client_auc_std"],
        result["client_auc_min"],
        result["stratum_auc_std"],
        result["stratum_auc_min"],
        duration,
    )
    return result


# =============================================================================
# E5
# =============================================================================
def run_e5(
    *,
    n_rounds: int | None = None,
    seed: int | None = None,
    partition: str = "label_skew_a0.1",
    strategies: list[str] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Compare FedAvg vs FedProx on both fairness views.

    Parameters
    ----------
    partition:
        Defaults to ``label_skew_a0.1`` because fairness only becomes
        interesting when the silos are genuinely unequal. On IID partitions
        every client sees the same distribution, so per-client AUC barely
        varies and the experiment has nothing to measure.
    strategies:
        Defaults to ``["fedavg", "fedprox"]``.

    Returns
    -------
    dict
        ``{runs, client_table, stratum_table, fedprox_client_std_gain_pct,
        fedprox_stratum_std_gain_pct}``
    """
    cfg = get_config()
    rounds = n_rounds if n_rounds is not None else cfg.fl.n_rounds
    torch_seed = seed if seed is not None else cfg.project.seed
    names = strategies if strategies is not None else ["fedavg", "fedprox"]

    vocab = fit_vocabulary(_partition_dir(partition), cfg.partitioning.n_clients)
    X_test, y_test, test_df = _load_test_set(vocab)
    strata = load_strata(test_df)

    logger.info("=" * 70)
    logger.info("E5 — fairness (%s rounds, partition=%s)", rounds, partition)
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
            strata=strata,
        )
        for s in names
    ]

    client_table = pd.DataFrame([
        {
            "strategy": r["strategy"],
            "auc": r["auc"],
            "client_auc_mean": r["client_auc_mean"],
            "client_auc_std": r["client_auc_std"],
            "client_auc_min": r["client_auc_min"],
            "client_auc_max": r["client_auc_max"],
            "client_auc_gap": r["client_auc_gap"],
        }
        for r in runs
    ])

    stratum_table = pd.DataFrame([
        {
            "strategy": r["strategy"],
            "stratum_auc_std": r["stratum_auc_std"],
            "stratum_auc_min": r["stratum_auc_min"],
            "stratum_auc_gap": r["stratum_auc_gap"],
        }
        for r in runs
    ])

    # Lower dispersion is fairer, so a positive gain means FedProx reduced the
    # spread. Sign convention is stated in the markdown so it cannot be misread.
    base = client_table.iloc[0]
    gains: dict[str, float] = {}
    for _, row in client_table.iterrows():
        if row["strategy"] == base["strategy"]:
            gains[f"{row['strategy']}_client_std_gain_pct"] = 0.0
            continue
        gains[f"{row['strategy']}_client_std_gain_pct"] = (
            (base["client_auc_std"] - row["client_auc_std"]) / base["client_auc_std"] * 100
            if base["client_auc_std"] > 0
            else 0.0
        )

    base_s = stratum_table.iloc[0]
    for _, row in stratum_table.iterrows():
        if row["strategy"] == base_s["strategy"]:
            gains[f"{row['strategy']}_stratum_std_gain_pct"] = 0.0
            continue
        gains[f"{row['strategy']}_stratum_std_gain_pct"] = (
            (base_s["stratum_auc_std"] - row["stratum_auc_std"])
            / base_s["stratum_auc_std"]
            * 100
            if base_s["stratum_auc_std"] > 0
            else 0.0
        )

    result = {
        "runs": runs,
        "client_table": client_table.to_dict(orient="records"),
        "stratum_table": stratum_table.to_dict(orient="records"),
        **gains,
    }

    if write:
        out = cfg.paths.results_dir / "E5_fairness.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logger.info("Wrote %s", out)

        md = cfg.paths.results_dir / "E5_fairness.md"
        md.write_text(
            _render_e5(client_table, stratum_table, gains, runs, partition),
            encoding="utf-8",
        )
        logger.info("Wrote %s", md)

    return result


def _render_e5(
    client_table: pd.DataFrame,
    stratum_table: pd.DataFrame,
    gains: dict[str, float],
    runs: list[dict[str, Any]],
    partition: str,
) -> str:
    lines = [
        "# E5 — Fairness",
        "",
        f"Partition `{partition}`, scored on `test_real` (1,318 rows).",
        "",
        "## E5.1 — Per-client fairness",
        "",
        "| Strategy | Overall AUC | Client AUC mean | Client AUC std | Worst client | Best client | Spread |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in client_table.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['auc']:.4f} | {r['client_auc_mean']:.4f} | "
            f"{r['client_auc_std']:.4f} | {r['client_auc_min']:.4f} | "
            f"{r['client_auc_max']:.4f} | {r['client_auc_gap']:.4f} |"
        )

    lines += [
        "",
        "## E5.2 — Per-stratum fairness (Churn Category)",
        "",
        "| Strategy | Stratum AUC std | Worst stratum | Spread |",
        "|---|---|---|---|",
    ]
    for _, r in stratum_table.iterrows():
        lines.append(
            f"| {r['strategy']} | {r['stratum_auc_std']:.4f} | "
            f"{r['stratum_auc_min']:.4f} | {r['stratum_auc_gap']:.4f} |"
        )

    # Per-stratum detail for the first run, so the report shows which category
    # is hardest rather than only the aggregate spread.
    if runs:
        lines += [
            "",
            f"### Per-stratum detail — {runs[0]['strategy']}",
            "",
            "| Stratum | Rows | Churned | AUC |",
            "|---|---|---|---|",
        ]
        for row in runs[0]["per_stratum"]:
            auc = "n/a" if not np.isfinite(row["auc"]) else f"{row['auc']:.4f}"
            lines.append(
                f"| {row['stratum']} | {row['n_rows']:,} | {row['n_churned']:,} | {auc} |"
            )

    alt = next((k for k in gains if k.endswith("_client_std_gain_pct") and not k.startswith("fedavg")), None)
    if alt:
        name = alt.replace("_client_std_gain_pct", "")
        c_gain = gains[alt]
        s_gain = gains.get(f"{name}_stratum_std_gain_pct", 0.0)
        verdict = "PASS" if c_gain > 0 or s_gain > 0 else "FAIL"
        lines += [
            "",
            f"**Success criterion #6 ({name} reduces unfairness): {verdict}** — "
            f"client AUC std {c_gain:+.2f}%, stratum AUC std {s_gain:+.2f}% "
            f"(positive = less dispersion = fairer)",
        ]

    lines += [
        "",
        "> **Caveat (E5.2).** With 374 churned customers spread across 5 categories, "
        "each stratum contains roughly 75 churned rows. AUC on 75 positives has a "
        "95% CI of roughly ±0.10, so differences smaller than that between "
        "strategies should not be interpreted as meaningful.",
    ]
    return "\n".join(lines) + "\n"


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.experiments.s18_e5_fairness``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run E5 (fairness)")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--partition", default="label_skew_a0.1")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        help="Strategies to compare (default: fedavg fedprox)",
    )
    args = parser.parse_args()

    result = run_e5(
        n_rounds=args.rounds,
        partition=args.partition,
        strategies=args.strategies,
    )

    print("\n" + "=" * 70)
    print("E5 — FAIRNESS")
    print("=" * 70)
    for row in result["client_table"]:
        print(
            f"  {row['strategy']:<8} AUC={row['auc']:.4f}  "
            f"client std={row['client_auc_std']:.4f}  "
            f"worst={row['client_auc_min']:.4f}"
        )


if __name__ == "__main__":
    main()
