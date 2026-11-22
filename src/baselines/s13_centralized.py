"""Centralized baseline: train on the full synthetic training set.

Pipeline position: **TRAIN → EVALUATE → REPORT**

This is the upper bound for E1. It sees exactly the same 900,000 rows the
federated clients see in aggregate, but all in one place — no isolation, no
aggregation, no non-IID split. The gap between this and FL is the price of
privacy.

Fairness of the comparison
--------------------------
* Same total data: ``synth_train`` (900,000 rows) equals the sum of the 10 silos.
* Same architecture: identical MLP, GroupNorm, dropout.
* Same optimiser settings: Adam, batch 256, learning rate 1e-3.
* Same total epochs: 50, matching FL's 50 rounds x 1 local epoch.

The only difference is that the centralized model sees all rows at once instead
of 10 disjoint shards.

Usage
-----
>>> from src.baselines.s13_centralized import train_centralized
>>> result = train_centralized(n_epochs=50)
>>> result["metrics"]["auc"]
0.89...
"""

from __future__ import annotations

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
from src.evaluation.s12_eval import evaluate_with_ci
from src.features.s07_preprocess import (
    Vocabulary,
    encode,
    fit_local_scaler,
    fit_vocabulary,
)
from src.models.s08_mlp import build_model, count_parameters

logger = logging.getLogger(__name__)


# =============================================================================
# Data loading
# =============================================================================
def load_centralized_data(
    vocab: Vocabulary,
    *,
    train_path: str | Path | None = None,
    test_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ``synth_train`` and ``test_real``, scaled with the canonical scaler.

    The scaler is fitted on ``train_real`` — the same scaler used to evaluate
    the FL global model. Using one canonical scale for both makes the E1
    comparison like-for-like.

    Returns
    -------
    (X_train, y_train, X_test, y_test)
    """
    cfg = get_config()
    train_p = Path(train_path) if train_path else cfg.paths.synth_train_parquet
    test_p = Path(test_path) if test_path else cfg.paths.test_real_parquet

    for p in (train_p, test_p):
        if not p.exists():
            raise FileNotFoundError(f"Required artifact missing: {p}")

    train_df = pd.read_parquet(train_p)
    test_df = pd.read_parquet(test_p)

    X_train_raw = encode(train_df, vocab)
    X_test_raw = encode(test_df, vocab)

    # Canonical scaler: fitted on train_real, applied to both sides
    train_real = pd.read_parquet(cfg.paths.train_real_parquet)
    scaler = fit_local_scaler(encode(train_real, vocab))

    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    y_train = train_df[TARGET_COL].to_numpy(dtype=np.int64)
    y_test = test_df[TARGET_COL].to_numpy(dtype=np.int64)

    logger.info(
        "Centralized data: train %s, test %s, input_dim %s",
        X_train.shape,
        X_test.shape,
        X_train.shape[1],
    )
    return X_train, y_train, X_test, y_test


# =============================================================================
# Training
# =============================================================================
def train_centralized(
    *,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    vocab: Vocabulary | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Train one model on all 900,000 synthetic training rows.

    Parameters
    ----------
    n_epochs:
        Defaults to 50, matching FL's 50 rounds x 1 local epoch.
    batch_size, learning_rate:
        Default to the FL settings so the comparison is fair.
    seed:
        Torch seed.
    vocab:
        Shared one-hot layout. Fitted from the IID partition if omitted.
    write:
        When ``True``, save the metrics to ``results/E1_centralized.json``.

    Returns
    -------
    dict
        ``{model, metrics, n_epochs, duration_sec, n_parameters}``
    """
    cfg = get_config()
    epochs = n_epochs if n_epochs is not None else cfg.fl.n_rounds
    bs = batch_size if batch_size is not None else cfg.fl.batch_size
    lr = learning_rate if learning_rate is not None else cfg.fl.learning_rate
    torch_seed = seed if seed is not None else cfg.project.seed

    if vocab is None:
        part_dir = cfg.paths.synthetic_dir / "partitions_iid"
        vocab = fit_vocabulary(part_dir, cfg.partitioning.n_clients)

    X_train, y_train, X_test, y_test = load_centralized_data(vocab)

    torch.manual_seed(torch_seed)
    model = build_model(X_train.shape[1], seed=torch_seed)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float().unsqueeze(1),
    )
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, drop_last=False)

    logger.info(
        "Centralized training: %s rows, %s epochs, batch %s, lr %s",
        f"{len(X_train):,}",
        epochs,
        bs,
        lr,
    )

    t0 = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            logger.info(
                "  epoch %s/%s: train loss %.4f", epoch, epochs, running / max(n_batches, 1)
            )

    duration = time.perf_counter() - t0
    metrics = evaluate_with_ci(model, X_test, y_test, seed=torch_seed)

    logger.info(
        "Centralized done in %.1fs: AUC %.4f [%.4f, %.4f]",
        duration,
        metrics["auc"],
        metrics["auc_ci_lower"],
        metrics["auc_ci_upper"],
    )

    result = {
        "model": "centralized",
        "metrics": metrics,
        "n_epochs": epochs,
        "batch_size": bs,
        "learning_rate": lr,
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "n_parameters": count_parameters(model),
        "duration_sec": round(duration, 2),
        "seed": torch_seed,
    }

    if write:
        import json

        out = cfg.paths.results_dir / "E1_centralized.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote %s", out)

    return {"model_obj": model, **result}


def main() -> None:
    """CLI entry point: ``python -m src.baselines.s13_centralized``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train the centralized baseline")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    result = train_centralized(n_epochs=args.epochs, seed=args.seed)
    m = result["metrics"]

    print("\n" + "=" * 70)
    print("CENTRALIZED BASELINE")
    print("=" * 70)
    print(f"  train rows   : {result['n_train_rows']:,}")
    print(f"  test rows    : {result['n_test_rows']:,}")
    print(f"  epochs       : {result['n_epochs']}")
    print(f"  parameters   : {result['n_parameters']:,}")
    print(f"  duration     : {result['duration_sec']:.1f}s")
    print(f"  AUC          : {m['auc']:.4f} [{m['auc_ci_lower']:.4f}, {m['auc_ci_upper']:.4f}]")
    print(f"  F1           : {m['f1']:.4f}")
    print(f"  Recall@10%   : {m['recall_at_10pct']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
