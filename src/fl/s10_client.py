"""Flower client wrapper for one telecom silo.

Pipeline position: **PREPROCESS → MODEL → TRAIN (local) → AGGREGATE (server)**

This module is a thin adapter: it translates Flower's numpy-array protocol into
calls on :class:`src.fl.s09_trainer.LocalTrainer`. All training logic lives in
the trainer so it can be tested without a running server.

What crosses the wire
---------------------
Only model weights. The client never sends features, labels, or any statistic
derived from its silo. ``num_examples`` is sent so the server can weight the
aggregation, which is standard FedAvg and reveals only the silo size.

Usage
-----
>>> from src.fl.s10_client import make_client
>>> client = make_client(cid=0, partition_dir=..., vocab=...)
>>> weights, n, metrics = client.fit(global_weights, config={})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_config
from src.features.s07_preprocess import Vocabulary, load_client
from src.fl.s09_trainer import LocalTrainer

logger = logging.getLogger(__name__)

try:  # Flower is optional at import time so unit tests can run without it
    import flwr as fl
    from flwr.client import NumPyClient

    _HAS_FLOWER = True
except ImportError:  # pragma: no cover
    _HAS_FLOWER = False
    NumPyClient = object  # type: ignore[misc,assignment]


# =============================================================================
# Client
# =============================================================================
class TelcoClient(NumPyClient):
    """One federated client backed by a single silo.

    Parameters
    ----------
    client_id:
        Index of the silo (0..K-1).
    partition_dir:
        Directory holding ``client_{i}.parquet``.
    vocab:
        Shared one-hot layout. Must be identical across clients, otherwise the
        weight tensors have different shapes and FedAvg cannot average them.
    proximal_mu:
        FedProx coefficient. ``0.0`` is plain FedAvg.
    """

    def __init__(
        self,
        client_id: int,
        partition_dir: str | Path,
        vocab: Vocabulary,
        *,
        proximal_mu: float = 0.0,
    ) -> None:
        self.client_id = client_id
        self.partition_dir = Path(partition_dir)
        self.vocab = vocab
        self.proximal_mu = proximal_mu

        X, y, scaler = load_client(self.partition_dir, client_id, vocab)
        self.X = X
        self.y = y
        self.scaler = scaler
        self.input_dim = X.shape[1]

        self.trainer = LocalTrainer(
            X,
            y,
            input_dim=self.input_dim,
            proximal_mu=proximal_mu,
        )
        logger.info(
            "TelcoClient %s ready: %s samples, %s features, churn %.4f",
            client_id,
            f"{len(y):,}",
            self.input_dim,
            float(y.mean()),
        )

    # -------------------------------------------------------------------------
    def get_parameters(self, config: dict[str, Any] | None = None) -> list[np.ndarray]:
        """Return the current local weights."""
        from src.models.s08_mlp import get_weights

        return get_weights(self.trainer.model)

    # -------------------------------------------------------------------------
    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[list[np.ndarray], int, dict[str, float]]:
        """Train locally and return updated weights.

        Returns
        -------
        (weights, num_examples, metrics)
            ``num_examples`` lets the server weight the aggregation by silo
            size, which is what makes FedAvg correct under unequal partitions.
        """
        epochs = config.get("local_epochs") if config else None
        result = self.trainer.fit(parameters, epochs=epochs)

        logger.info(
            "Client %s fit: round=%s loss=%.4f auc=%.4f",
            self.client_id,
            config.get("server_round", "?") if config else "?",
            result.loss,
            result.auc,
        )
        return result.weights, result.n_samples, result.to_metrics()

    # -------------------------------------------------------------------------
    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict[str, Any],
    ) -> tuple[float, int, dict[str, float]]:
        """Score the global weights on this client's silo."""
        from src.models.s08_mlp import set_weights

        set_weights(self.trainer.model, parameters)
        loss, auc = self.trainer.evaluate()
        return loss, len(self.X), {"loss": loss, "auc": auc}


# =============================================================================
# Factory
# =============================================================================
def make_client(
    client_id: int,
    partition_dir: str | Path | None = None,
    vocab: Vocabulary | None = None,
    *,
    proximal_mu: float = 0.0,
) -> TelcoClient:
    """Build a client, filling defaults from the config and IID partition."""
    cfg = get_config()

    if partition_dir is None:
        partition_dir = cfg.paths.synthetic_dir / "partitions_iid"

    if vocab is None:
        from src.features.s07_preprocess import fit_vocabulary

        vocab = fit_vocabulary(partition_dir, cfg.partitioning.n_clients)

    return TelcoClient(
        client_id,
        partition_dir,
        vocab,
        proximal_mu=proximal_mu,
    )


def client_fn_factory(
    partition_dir: str | Path,
    vocab: Vocabulary,
    *,
    proximal_mu: float = 0.0,
):
    """Return a ``client_fn(cid)`` suitable for ``fl.client.start_client``."""
    if not _HAS_FLOWER:
        raise ImportError("flwr is required to build a Flower client_fn")

    def client_fn(cid: str) -> Any:
        return make_client(
            int(cid),
            partition_dir,
            vocab,
            proximal_mu=proximal_mu,
        ).to_client()

    return client_fn


def main() -> None:
    """CLI entry point: ``python -m src.fl.s10_client``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    client = make_client(0)
    weights = client.get_parameters()
    loss, n, metrics = client.evaluate(weights, {})

    print("\n" + "=" * 70)
    print("FLOWER CLIENT SMOKE TEST (client 0)")
    print("=" * 70)
    print(f"  client_id    : {client.client_id}")
    print(f"  samples      : {n:,}")
    print(f"  input dim    : {client.input_dim}")
    print(f"  weight tensors: {len(weights)}")
    print(f"  loss         : {loss:.4f}")
    print(f"  auc          : {metrics['auc']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
