"""Local-only baseline: each client trains alone, no aggregation.

Pipeline position: **TRAIN → EVALUATE → REPORT**

This is the lower bound for E1. Each client trains on its own silo and is
scored on ``test_real`` using **its own scaler** — the model never saw any other
scale, so evaluating it with a different one would be meaningless.

The spread across clients is the fairness signal that E5.1 quantifies: if some
clients do much worse than others, the federated setup is unfair to them.

Usage
-----
>>> from src.baselines.s13_local_only import train_local_only
>>> df = train_local_only()
>>> df["auc"].mean()
0.7...
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import get_config
from src.data.s01_clean import TARGET_COL
from src.evaluation.s12_eval import evaluate_model
from src.features.s07_preprocess import (
    Vocabulary,
    encode,
    fit_local_scaler,
    fit_vocabulary,
    load_client,
)
from src.models.s08_mlp import build_model, count_parameters

logger = logging.getLogger(__name__)


# =============================================================================
# Training
# =============================================================================
def train_one_client(
    client_id: int,
    partition_dir: Path,
    vocab: Vocabulary,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    *,
    n_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Train a model on one silo and score it on ``test_real``.

    The test set is scaled with **this client's** scaler, because that is the
    only scale the model has ever seen.
    """
    X, y, scaler = load_client(partition_dir, client_id, vocab)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    torch.manual_seed(seed + client_id)
    model = build_model(X.shape[1], seed=seed + client_id)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float().unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    t0 = time.perf_counter()
    model.train()
    for _ in range(n_epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    duration = time.perf_counter() - t0

    metrics = evaluate_model(model, X_test, y_test)

    logger.info(
        "Local-only client %s: %s rows, AUC %.4f, %.1fs",
        client_id,
        f"{len(X):,}",
        metrics["auc"],
        duration,
    )

    return {
        "client_id": client_id,
        "n_train_rows": int(len(X)),
        "n_churned": int(y.sum()),
        "churn_rate": float(y.mean()),
        "auc": metrics["auc"],
        "f1": metrics["f1"],
        "recall_at_10pct": metrics["recall_at_10pct"],
        "n_parameters": count_parameters(model),
        "duration_sec": round(duration, 2),
    }


def train_local_only(
    *,
    partition: str = "iid",
    n_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    vocab: Vocabulary | None = None,
    write: bool = True,
) -> pd.DataFrame:
    """Train every client in isolation and return per-client metrics.

    Parameters
    ----------
    partition:
        Sub-directory under ``data/synthetic`` holding the silos.
    n_epochs:
        Defaults to 50, matching the FL total.
    write:
        When ``True``, save the table to ``results/E1_local_only.csv``.

    Returns
    -------
    pandas.DataFrame
        One row per client plus a summary row is *not* included — callers
        aggregate with ``.mean()`` / ``.std()`` / ``.min()``.
    """
    cfg = get_config()
    epochs = n_epochs if n_epochs is not None else cfg.fl.n_rounds
    bs = batch_size if batch_size is not None else cfg.fl.batch_size
    lr = learning_rate if learning_rate is not None else cfg.fl.learning_rate
    torch_seed = seed if seed is not None else cfg.project.seed
    k = cfg.partitioning.n_clients

    part_dir = cfg.paths.synthetic_dir / f"partitions_{partition}"
    if not part_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {part_dir}")

    if vocab is None:
        vocab = fit_vocabulary(part_dir, k)

    # test_real encoded once, then scaled per client
    test_df = pd.read_parquet(cfg.paths.test_real_parquet)
    X_test_raw = encode(test_df, vocab)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int64)

    logger.info(
        "Local-only training: %s clients, %s epochs each, partition=%s",
        k,
        epochs,
        partition,
    )

    rows = [
        train_one_client(
            cid,
            part_dir,
            vocab,
            X_test_raw,
            y_test,
            n_epochs=epochs,
            batch_size=bs,
            learning_rate=lr,
            seed=torch_seed,
        )
        for cid in range(k)
    ]

    df = pd.DataFrame(rows)

    logger.info(
        "Local-only AUC: mean %.4f, std %.4f, min %.4f, max %.4f",
        df["auc"].mean(),
        df["auc"].std(),
        df["auc"].min(),
        df["auc"].max(),
    )

    if write:
        out = cfg.paths.results_dir / f"E1_local_only_{partition}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Wrote %s", out)

        summary = {
            "partition": partition,
            "n_clients": k,
            "n_epochs": epochs,
            "auc_mean": float(df["auc"].mean()),
            "auc_std": float(df["auc"].std()),
            "auc_min": float(df["auc"].min()),
            "auc_max": float(df["auc"].max()),
            "f1_mean": float(df["f1"].mean()),
            "recall_at_10pct_mean": float(df["recall_at_10pct"].mean()),
        }
        summary_path = cfg.paths.results_dir / f"E1_local_only_{partition}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Wrote %s", summary_path)

    return df


def main() -> None:
    """CLI entry point: ``python -m src.baselines.s13_local_only``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train the local-only baseline")
    parser.add_argument("--partition", default="iid")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    df = train_local_only(partition=args.partition, n_epochs=args.epochs)

    print("\n" + "=" * 70)
    print(f"LOCAL-ONLY BASELINE ({args.partition})")
    print("=" * 70)
    print(df[["client_id", "n_train_rows", "churn_rate", "auc", "f1"]].to_string(index=False))
    print("-" * 70)
    print(f"  AUC mean : {df['auc'].mean():.4f}")
    print(f"  AUC std  : {df['auc'].std():.4f}")
    print(f"  AUC min  : {df['auc'].min():.4f}")
    print(f"  AUC max  : {df['auc'].max():.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
