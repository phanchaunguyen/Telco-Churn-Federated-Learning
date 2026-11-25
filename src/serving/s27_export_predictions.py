"""Export model predictions so Spark can score them independently.

Pipeline position: **EXPORT (model) → PREDICT → EVALUATE (Spark)**

``spark_eval`` needs a table of ``(y_true, y_score)`` to aggregate. Nothing in
the repository produced one: the experiment runners compute metrics in memory
and write only the summary numbers, so there was no per-row artifact for Spark
to read.

This module closes that gap. It loads the exported model, scores ``test_real``
with the canonical scaler, and writes one parquet file. The AUC computed from
that file must match the AUC the pandas path reports — that equality is the
whole point of D14.

Why the canonical scaler
------------------------
Same reason as the API: a hold-out row belongs to no silo, so the scaler fitted
on ``train_real`` is the correct one. Using a client scaler here would produce
predictions that disagree with every other evaluation in the project.

Usage
-----
>>> from src.serving.s27_export_predictions import export_predictions
>>> result = export_predictions()
>>> result["n_rows"]
1318
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config import get_config
from src.evaluation.s12_eval import evaluate_model
from src.features.s07_preprocess import encode, fit_local_scaler, load_preprocessing
from src.models.s08_mlp import load_model

logger = logging.getLogger(__name__)

TARGET_COL = "churn"


# =============================================================================
# Paths
# =============================================================================
def predictions_path() -> Path:
    """Where the per-row predictions live."""
    return get_config().paths.results_dir / "predictions" / "fl_fedavg.parquet"


def _model_path() -> Path:
    return get_config().paths.results_dir / "models" / "fl_fedavg.pt"


def _preprocessing_path() -> Path:
    return get_config().paths.processed_dir / "preprocessing_iid.json"


# =============================================================================
# Export
# =============================================================================
def export_predictions(*, write: bool = True) -> dict[str, Any]:
    """Score ``test_real`` with the exported model and persist the rows.

    Returns
    -------
    dict
        ``{n_rows, auc, pr_auc, path}``
    """
    cfg = get_config()

    model_file = _model_path()
    prep_file = _preprocessing_path()
    missing = [p for p in (model_file, prep_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Serving artifacts missing: "
            + ", ".join(str(p) for p in missing)
            + ". Run `python -m src.serving.s21_export_model` first."
        )

    model = load_model(model_file)
    vocab, scalers = load_preprocessing(prep_file)
    if 0 not in scalers:
        raise KeyError(f"{prep_file} has no scaler for client 0")

    # The canonical scaler is stored under client 0 by the export step.
    train_real = pd.read_parquet(cfg.paths.train_real_parquet)
    test_real = pd.read_parquet(cfg.paths.test_real_parquet)
    canonical = fit_local_scaler(encode(train_real, vocab))

    X_test = canonical.transform(encode(test_real, vocab)).astype(np.float32)
    y_true = test_real[TARGET_COL].to_numpy(dtype=np.int64)

    with torch.no_grad():
        y_score = model(torch.from_numpy(X_test)).squeeze(1).numpy()

    metrics = evaluate_model(model, X_test, y_true)
    logger.info(
        "Predictions: %s rows, AUC %.4f, PR-AUC %.4f",
        f"{len(y_true):,}",
        metrics["auc"],
        metrics["pr_auc"],
    )

    result: dict[str, Any] = {
        "n_rows": int(len(y_true)),
        "auc": metrics["auc"],
        "pr_auc": metrics["pr_auc"],
        "recall_at_10pct": metrics["recall_at_10pct"],
    }

    if write:
        out = predictions_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame({
            "y_true": y_true.astype(np.int32),
            "y_score": y_score.astype(np.float64),
        })
        frame.to_parquet(out, index=False)
        logger.info("Wrote %s", out)

        # A small sidecar so spark_eval can compare against the pandas number
        # without re-running the model.
        meta = out.parent / "predictions_metadata.json"
        meta.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote %s", meta)

        result["path"] = out
        result["metadata"] = meta

    return result


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.serving.s27_export_predictions``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    result = export_predictions()

    print("\n" + "=" * 70)
    print("PREDICTIONS EXPORTED")
    print("=" * 70)
    print(f"  rows     : {result['n_rows']:,}")
    print(f"  AUC      : {result['auc']:.4f}")
    print(f"  PR-AUC   : {result['pr_auc']:.4f}")
    print(f"  path     : {result['path']}")


if __name__ == "__main__":
    main()
