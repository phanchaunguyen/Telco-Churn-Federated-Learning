"""Export a trained federated model for serving.

Pipeline position: **TRAIN (FL) → EXPORT → SERVE**

Every experiment so far kept the trained weights in memory and wrote only
metrics to JSON, so there was nothing on disk for an API to load. This module
closes that gap: it trains the exact E1 fold-0 configuration, then writes

* ``results/models/fl_fedavg.pt`` — weights plus the architecture metadata
* ``data/processed/preprocessing_iid.json`` — vocabulary and per-client scalers

Why the E1 fold-0 configuration
-------------------------------
E1 fold 0 is the run the report quotes (AUC 0.8949 on ``test_real``), so the
served model is the one whose numbers appear in the write-up. Training a fresh
model with different settings would make the API and the report disagree.

Why the canonical scaler and not a client scaler
------------------------------------------------
Each silo has its own ``LocalScaler`` fitted on that silo's statistics. A new
customer arriving at the API belongs to no silo, so there is no local scaler to
apply. The canonical scaler fitted on ``train_real`` is used instead, which is
exactly how E1-E7 score the global model. Serving and evaluation therefore
preprocess identically.

Usage
-----
>>> from src.serving.s21_export_model import export_model
>>> paths = export_model()
>>> paths["model"]
PosixPath('.../results/models/fl_fedavg.pt')
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.evaluation.s12_eval import evaluate_model
from src.features.s07_preprocess import (
    encode,
    fit_local_scaler,
    fit_vocabulary,
    load_all_clients,
    save_preprocessing,
)
from src.fl.s11_server import run_sequential
from src.models.s08_mlp import build_model, save_model, set_weights

logger = logging.getLogger(__name__)

TARGET_COL = "churn"


# =============================================================================
# Paths
# =============================================================================
def model_path() -> Path:
    """Where the served model lives."""
    return get_config().paths.results_dir / "models" / "fl_fedavg.pt"


def preprocessing_path() -> Path:
    """Where the vocabulary and scalers live."""
    return get_config().paths.processed_dir / "preprocessing_iid.json"


# =============================================================================
# Export
# =============================================================================
def export_model(
    *,
    rounds: int | None = None,
    seed: int | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Train the E1 fold-0 configuration and persist it for serving.

    Parameters
    ----------
    rounds:
        Federated rounds. Defaults to ``cfg.fl.n_rounds`` (50), matching E1.
    seed:
        Torch seed. Defaults to ``cfg.project.seed`` (42), matching E1 fold 0.

    Returns
    -------
    dict
        ``{model, preprocessing, metrics, n_params}``
    """
    cfg = get_config()
    n_rounds = rounds if rounds is not None else cfg.fl.n_rounds
    torch_seed = seed if seed is not None else cfg.project.seed
    part_dir = cfg.paths.synthetic_dir / "partitions_iid"
    k = cfg.partitioning.n_clients

    if not part_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {part_dir}")

    logger.info("=" * 70)
    logger.info("EXPORT — training E1 fold-0 configuration (%s rounds)", n_rounds)
    logger.info("=" * 70)

    # --- Preprocessing artifacts -----------------------------------------
    vocab = fit_vocabulary(part_dir, k)
    clients = load_all_clients(part_dir, k, vocab)
    scalers = {cid: s for cid, (_, _, s) in enumerate(clients)}

    # --- Train ------------------------------------------------------------
    history = run_sequential(
        n_rounds=n_rounds,
        num_clients=k,
        strategy_name="fedavg",
        partition="iid",
        partition_dir=part_dir,
        seed=torch_seed,
    )

    model = build_model(history["input_dim"], seed=torch_seed)
    set_weights(model, history["final_weights"])

    # --- Score on the real hold-out, the same way E1 does -----------------
    train_real = pd.read_parquet(cfg.paths.train_real_parquet)
    test_real = pd.read_parquet(cfg.paths.test_real_parquet)
    canonical = fit_local_scaler(encode(train_real, vocab))
    X_test = canonical.transform(encode(test_real, vocab)).astype(np.float32)
    y_test = test_real[TARGET_COL].to_numpy(dtype=np.int64)
    metrics = evaluate_model(model, X_test, y_test)

    logger.info(
        "Exported model: AUC %.4f, PR-AUC %.4f, Recall@10%% %.4f",
        metrics["auc"],
        metrics["pr_auc"],
        metrics["recall_at_10pct"],
    )

    result: dict[str, Any] = {
        "metrics": metrics,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "rounds": n_rounds,
        "seed": torch_seed,
    }

    if write:
        mpath = save_model(model, model_path())
        ppath = preprocessing_path()
        save_preprocessing(vocab, scalers, ppath)

        # A small sidecar so the API can report what it is serving without
        # having to load the whole checkpoint.
        meta = {
            "model_file": mpath.name,
            "preprocessing_file": ppath.name,
            "input_dim": history["input_dim"],
            "n_params": result["n_params"],
            "rounds": n_rounds,
            "seed": torch_seed,
            "strategy": "fedavg",
            "partition": "iid",
            "metrics_test_real": metrics,
        }
        meta_path = mpath.parent / "model_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        logger.info("Wrote %s", meta_path)

        result["model"] = mpath
        result["preprocessing"] = ppath
        result["metadata"] = meta_path

    return result


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.serving.s21_export_model``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Export the FL model for serving")
    parser.add_argument("--rounds", type=int, default=None)
    args = parser.parse_args()

    result = export_model(rounds=args.rounds)

    print("\n" + "=" * 70)
    print("MODEL EXPORTED")
    print("=" * 70)
    print(f"  model         : {result['model']}")
    print(f"  preprocessing : {result['preprocessing']}")
    print(f"  metadata      : {result['metadata']}")
    print(f"  params        : {result['n_params']:,}")
    print(f"  AUC (real)    : {result['metrics']['auc']:.4f}")
    print(f"  PR-AUC (real) : {result['metrics']['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
