"""Local training loop for one federated client.

Pipeline position: **PREPROCESS → MODEL → TRAIN (local) → AGGREGATE (server)**

This module owns everything that happens inside a single client between
receiving global weights and returning updated weights. It is deliberately
free of Flower imports so it can be unit-tested without a running server.

Two extension points are stubbed here and filled in later phases:

* ``proximal_mu`` — FedProx proximal term (Phase 5)
* ``privacy_engine`` — Opacus DP-SGD wrapper (Phase 6)

Usage
-----
>>> from src.fl.s09_trainer import LocalTrainer
>>> trainer = LocalTrainer(X, y, input_dim=X.shape[1])
>>> result = trainer.fit(global_weights, epochs=1)
>>> result["auc"]
0.87...
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.config import get_config
from src.models.s08_mlp import build_model, get_weights, set_weights

logger = logging.getLogger(__name__)


# =============================================================================
# Result container
# =============================================================================
@dataclass
class TrainResult:
    """Outcome of one local training round."""

    weights: list[np.ndarray]
    n_samples: int
    loss: float
    auc: float
    duration_sec: float
    bytes_sent: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_metrics(self) -> dict[str, float]:
        """Flower expects a flat dict of floats."""
        return {"loss": self.loss, "auc": self.auc}


# =============================================================================
# Trainer
# =============================================================================
class LocalTrainer:
    """Trains one client's copy of the model on that client's silo.

    Parameters
    ----------
    X, y:
        Preprocessed features and labels for this client only.
    input_dim:
        Feature width. Must equal ``X.shape[1]``; passed explicitly so a
        mismatch fails loudly instead of silently building the wrong model.
    batch_size, learning_rate, local_epochs:
        Override the config defaults.
    proximal_mu:
        FedProx coefficient. ``0.0`` (default) is plain FedAvg local training.
    seed:
        Torch seed for reproducible dropout and weight init.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        input_dim: int,
        *,
        batch_size: int | None = None,
        learning_rate: float | None = None,
        local_epochs: int | None = None,
        proximal_mu: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if X.shape[1] != input_dim:
            raise ValueError(
                f"input_dim {input_dim} does not match X.shape[1] {X.shape[1]}"
            )
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")

        cfg = get_config()
        self.X = X
        self.y = y
        self.input_dim = input_dim
        self.batch_size = batch_size or cfg.fl.batch_size
        self.learning_rate = learning_rate or cfg.fl.learning_rate
        self.local_epochs = local_epochs or cfg.fl.local_epochs
        self.proximal_mu = proximal_mu
        self.seed = seed if seed is not None else cfg.project.seed

        torch.manual_seed(self.seed)
        self.model = build_model(input_dim, seed=self.seed)
        self.criterion = nn.BCELoss()

        self._dataset = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float().unsqueeze(1),
        )

    # -------------------------------------------------------------------------
    def _loader(self) -> DataLoader:
        return DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    # -------------------------------------------------------------------------
    def evaluate(self) -> tuple[float, float]:
        """Return ``(loss, auc)`` on this client's full silo."""
        self.model.eval()
        with torch.no_grad():
            x_all = torch.from_numpy(self.X).float()
            y_all = torch.from_numpy(self.y).float().unsqueeze(1)
            proba = self.model(x_all)
            loss = float(self.criterion(proba, y_all).item())

        y_true = self.y
        y_score = proba.squeeze(1).numpy()
        # AUC is undefined when a silo has only one class
        if len(np.unique(y_true)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_true, y_score))
        return loss, auc

    # -------------------------------------------------------------------------
    def fit(
        self,
        global_weights: list[np.ndarray] | None = None,
        *,
        epochs: int | None = None,
    ) -> TrainResult:
        """Train locally starting from ``global_weights``.

        Parameters
        ----------
        global_weights:
            Weights received from the server. ``None`` means train from the
            current local initialisation (used for the centralized baseline).
        epochs:
            Override ``local_epochs``.
        """
        t0 = time.perf_counter()

        if global_weights is not None:
            set_weights(self.model, global_weights)

        # Snapshot the starting point for the FedProx proximal term
        anchor = (
            [torch.tensor(w, dtype=torch.float32) for w in global_weights]
            if global_weights is not None and self.proximal_mu > 0
            else None
        )

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        n_epochs = epochs if epochs is not None else self.local_epochs

        running_loss = 0.0
        n_batches = 0

        for _ in range(n_epochs):
            for xb, yb in self._loader():
                optimizer.zero_grad()
                proba = self.model(xb)
                loss = self.criterion(proba, yb)

                if anchor is not None:
                    loss = loss + self._proximal_term(anchor)

                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
                n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        _, auc = self.evaluate()
        weights = get_weights(self.model)
        duration = time.perf_counter() - t0

        result = TrainResult(
            weights=weights,
            n_samples=len(self.X),
            loss=train_loss,
            auc=auc,
            duration_sec=duration,
            bytes_sent=sum(w.nbytes for w in weights),
        )
        logger.info(
            "Local fit: n=%s loss=%.4f auc=%.4f in %.2fs",
            f"{result.n_samples:,}",
            result.loss,
            result.auc,
            result.duration_sec,
        )
        return result

    # -------------------------------------------------------------------------
    def _proximal_term(self, anchor: list[torch.Tensor]) -> torch.Tensor:
        """FedProx penalty ``(mu/2) * ||w - w_global||^2``.

        Keeps the local model from drifting too far from the global one, which
        is what makes FedProx robust to label skew.
        """
        penalty = torch.zeros((), dtype=torch.float32)
        for param, ref in zip(self.model.parameters(), anchor, strict=True):
            penalty = penalty + ((param - ref) ** 2).sum()
        return (self.proximal_mu / 2.0) * penalty


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.fl.s09_trainer``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    from src.features.s07_preprocess import fit_vocabulary, load_client

    cfg = get_config()
    part_dir = cfg.paths.synthetic_dir / "partitions_iid"
    k = cfg.partitioning.n_clients

    vocab = fit_vocabulary(part_dir, k)
    X, y, _ = load_client(part_dir, 0, vocab)

    trainer = LocalTrainer(X, y, input_dim=X.shape[1])
    before_loss, before_auc = trainer.evaluate()
    result = trainer.fit(epochs=1)

    print("\n" + "=" * 70)
    print("LOCAL TRAINER SMOKE TEST (client 0, IID)")
    print("=" * 70)
    print(f"  samples      : {result.n_samples:,}")
    print(f"  input dim    : {X.shape[1]}")
    print(f"  loss before  : {before_loss:.4f}")
    print(f"  loss after   : {result.loss:.4f}")
    print(f"  auc before   : {before_auc:.4f}")
    print(f"  auc after    : {result.auc:.4f}")
    print(f"  duration     : {result.duration_sec:.2f}s")
    print(f"  bytes sent   : {result.bytes_sent:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
