"""Flower server: FedAvg aggregation over the 10 telecom silos.

Pipeline position: **PREPROCESS → MODEL → TRAIN (local) → AGGREGATE (server)**

The server is a stateless aggregator. It never sees data — only the weight
tensors and sample counts that clients return. That is the entire point of the
project: ``W_global = sum_i (n_i / n) * W_i``.

Strategy selection
------------------
``fedavg``   baseline, weighted average of client weights
``fedprox``  same aggregation, but clients add a proximal term locally
``fedavgm``  server-side momentum on the aggregated update

FedProx is a *client-side* change, so the server strategy is identical to
FedAvg; the difference is passed to the client through ``proximal_mu``.

Usage
-----
>>> from src.fl.s11_server import run_simulation
>>> history = run_simulation(n_rounds=1, num_clients=10)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_config
from src.features.s07_preprocess import Vocabulary, fit_vocabulary

logger = logging.getLogger(__name__)

try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters
    from flwr.server.strategy import FedAvg, FedAvgM

    _HAS_FLOWER = True
except ImportError:  # pragma: no cover
    _HAS_FLOWER = False


# =============================================================================
# Strategy factory
# =============================================================================
def build_strategy(
    strategy_name: str,
    *,
    initial_parameters: Any = None,
    fraction_fit: float | None = None,
    fraction_evaluate: float | None = None,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
) -> Any:
    """Return a Flower strategy configured from ``configs/base.yaml``.

    Parameters
    ----------
    strategy_name:
        ``fedavg``, ``fedprox``, or ``fedavgm``.
    initial_parameters:
        Starting weights. ``None`` lets Flower initialise from the first client.
    """
    if not _HAS_FLOWER:
        raise ImportError("flwr is required to build a strategy")

    cfg = get_config()
    fl_cfg = cfg.fl

    common = {
        "fraction_fit": fraction_fit if fraction_fit is not None else fl_cfg.fraction_fit,
        "fraction_evaluate": (
            fraction_evaluate if fraction_evaluate is not None else fl_cfg.fraction_evaluate
        ),
        "min_fit_clients": min_fit_clients,
        "min_evaluate_clients": min_evaluate_clients,
        "min_available_clients": min_available_clients,
        "initial_parameters": initial_parameters,
    }

    if strategy_name == "fedavg":
        return FedAvg(**common)

    if strategy_name == "fedprox":
        # FedProx differs only in the client's local loss; aggregation is FedAvg.
        return FedAvg(**common)

    if strategy_name == "fedavgm":
        return FedAvgM(
            server_momentum=fl_cfg.strategies.fedavgm.server_momentum,
            **common,
        )

    raise ValueError(f"Unknown strategy: {strategy_name!r}")


def proximal_mu_for(strategy_name: str) -> float:
    """Return the proximal coefficient a client should use for this strategy."""
    cfg = get_config()
    if strategy_name == "fedprox":
        return cfg.fl.strategies.fedprox.proximal_mu
    return 0.0


# =============================================================================
# Simulation
# =============================================================================
def run_simulation(
    *,
    n_rounds: int | None = None,
    num_clients: int | None = None,
    strategy_name: str = "fedavg",
    partition: str = "iid",
    seed: int | None = None,
) -> dict[str, Any]:
    """Run a Flower simulation and return the per-round history.

    Parameters
    ----------
    n_rounds:
        Number of federated rounds. Defaults to ``cfg.fl.n_rounds`` (50).
    num_clients:
        Number of virtual clients. Defaults to ``cfg.partitioning.n_clients``.
    strategy_name:
        ``fedavg``, ``fedprox``, or ``fedavgm``.
    partition:
        Sub-directory under ``data/synthetic`` holding the silos.
    seed:
        Torch seed for reproducible local training.

    Returns
    -------
    dict
        ``{"strategy", "partition", "n_rounds", "num_clients",
           "losses_distributed", "metrics_distributed"}``
    """
    if not _HAS_FLOWER:
        raise ImportError("flwr is required to run a simulation")

    from src.fl.s10_client import client_fn_factory
    from src.models.s08_mlp import build_model, get_weights

    cfg = get_config()
    rounds = n_rounds if n_rounds is not None else cfg.fl.n_rounds
    k = num_clients if num_clients is not None else cfg.partitioning.n_clients
    part_dir = cfg.paths.synthetic_dir / f"partitions_{partition}"

    if not part_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {part_dir}")

    vocab = fit_vocabulary(part_dir, k)

    # Initialise the global model from the first client's feature width so the
    # server never has to guess the input dimension.
    from src.features.s07_preprocess import load_client

    X0, _, _ = load_client(part_dir, 0, vocab)
    model = build_model(X0.shape[1], seed=seed if seed is not None else cfg.project.seed)
    initial = ndarrays_to_parameters(get_weights(model))

    strategy = build_strategy(
        strategy_name,
        initial_parameters=initial,
        min_fit_clients=max(2, int(k * cfg.fl.fraction_fit)),
        min_evaluate_clients=max(2, int(k * cfg.fl.fraction_evaluate)),
        min_available_clients=k,
    )

    client_fn = client_fn_factory(
        part_dir,
        vocab,
        proximal_mu=proximal_mu_for(strategy_name),
    )

    logger.info(
        "Simulation: strategy=%s partition=%s rounds=%s clients=%s",
        strategy_name,
        partition,
        rounds,
        k,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=k,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    return {
        "strategy": strategy_name,
        "partition": partition,
        "n_rounds": rounds,
        "num_clients": k,
        "losses_distributed": list(history.losses_distributed),
        "metrics_distributed": {
            key: list(values) for key, values in history.metrics_distributed.items()
        },
    }


# =============================================================================
# Manual sequential simulation (no Ray)
# =============================================================================
def run_sequential(
    *,
    n_rounds: int | None = None,
    num_clients: int | None = None,
    strategy_name: str = "fedavg",
    partition: str = "iid",
    partition_dir: str | Path | None = None,
    fraction_fit: float | None = None,
    noise_multiplier: float = 0.0,
    telemetry: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run FedAvg rounds sequentially, without Ray.

    Flower 1.44's ``start_simulation`` requires Ray. Ray adds 2-4 GB of memory,
    10-20 s of start-up, and fragments stack traces across worker processes,
    which makes the first end-to-end round much harder to debug. This function
    implements the same FedAvg loop directly:

        sample clients -> broadcast W_global -> local fit -> weighted average

    The aggregation is identical to Flower's ``FedAvg`` strategy, so results
    are comparable. Switch to ``run_simulation`` (Ray) only for E6, where
    K=50 makes sequential execution too slow.

    Parameters
    ----------
    partition:
        Short name resolved as ``data/synthetic/partitions_{partition}``.
        Ignored when ``partition_dir`` is given.
    partition_dir:
        Explicit path to a directory holding ``client_{i}.parquet``. Use this
        for fold-specific partitions, which live under
        ``data/synthetic/fold_{i}/partitions_iid`` and do not follow the
        ``partitions_*`` naming convention.
    noise_multiplier:
        DP-SGD noise scale applied on every client. ``0.0`` (default) is
        non-private FedAvg.
    telemetry:
        Publish per-client and per-round metrics to Kafka. Off by default so
        experiments do not depend on a broker being up. Telemetry is
        fire-and-forget: a missing broker logs a warning and the run continues,
        because metrics are observability while weights are the training.

    Returns
    -------
    dict
        Per-round losses and metrics, plus the final global weights.
    """
    from src.features.s07_preprocess import load_client
    from src.fl.s10_client import TelcoClient
    from src.models.s08_mlp import build_model, get_weights

    cfg = get_config()
    rounds = n_rounds if n_rounds is not None else cfg.fl.n_rounds
    k = num_clients if num_clients is not None else cfg.partitioning.n_clients
    frac = fraction_fit if fraction_fit is not None else cfg.fl.fraction_fit

    if partition_dir is not None:
        part_dir = Path(partition_dir)
    else:
        part_dir = cfg.paths.synthetic_dir / f"partitions_{partition}"

    if not part_dir.exists():
        raise FileNotFoundError(f"Partition directory not found: {part_dir}")

    vocab = fit_vocabulary(part_dir, k)
    X0, _, _ = load_client(part_dir, 0, vocab)
    input_dim = X0.shape[1]

    model = build_model(input_dim, seed=seed if seed is not None else cfg.project.seed)
    global_weights = get_weights(model)

    # Build every client once; they hold their silo in memory for the whole run
    clients = [
        TelcoClient(
            cid,
            part_dir,
            vocab,
            proximal_mu=proximal_mu_for(strategy_name),
            noise_multiplier=noise_multiplier,
        )
        for cid in range(k)
    ]

    n_selected = max(2, int(round(k * frac)))
    rng = np.random.default_rng(seed if seed is not None else cfg.project.seed)

    losses: list[tuple[int, float]] = []
    aucs: list[tuple[int, float]] = []
    round_times: list[tuple[int, float]] = []
    momentum: list[np.ndarray] | None = None
    beta = cfg.fl.strategies.fedavgm.server_momentum
    lr = cfg.fl.strategies.fedavgm.server_learning_rate

    # Telemetry is optional and never fatal. The producer swallows broker
    # errors internally, so nothing below needs a try/except.
    producer = None
    if telemetry:
        from src.telemetry.s29_producer import TelemetryProducer

        producer = TelemetryProducer()
        if not producer.available:
            logger.warning("Telemetry requested but Kafka is unreachable")
            producer = None

    # Wall-clock for the whole run, excluding client construction above. E6
    # measures scaling, so the timer must cover only the federated loop.
    run_start = time.perf_counter()

    logger.info(
        "Sequential simulation: strategy=%s partition=%s rounds=%s clients=%s "
        "selected_per_round=%s",
        strategy_name,
        partition,
        rounds,
        k,
        n_selected,
    )

    for rnd in range(1, rounds + 1):
        round_start = time.perf_counter()
        selected = rng.choice(k, size=n_selected, replace=False)
        selected = sorted(int(c) for c in selected)

        results = []
        for cid in selected:
            weights, n_samples, metrics = clients[cid].fit(
                global_weights, {"server_round": rnd}
            )
            results.append((weights, n_samples, metrics))

            if producer is not None:
                producer.send_client_metrics(
                    round=rnd,
                    client_id=cid,
                    local_auc=float(metrics["auc"]),
                    local_loss=float(metrics["loss"]),
                    n_samples=int(n_samples),
                    bytes_sent=int(sum(w.nbytes for w in weights)),
                    duration_sec=float(clients[cid].trainer.last_duration_sec),
                )

        # FedAvg: W_global = sum_i (n_i / n) * W_i
        total = sum(n for _, n, _ in results)
        aggregated = [
            sum(w[i] * (n / total) for w, n, _ in results)
            for i in range(len(global_weights))
        ]

        if strategy_name == "fedavgm":
            # Server-side SGD with momentum, matching Flower's FedAvgM:
            #   pseudo_gradient = W_global - W_aggregated   (updates are the
            #                                                 opposite of grads)
            #   v = beta * v + pseudo_gradient
            #   W_global = W_global - lr * v
            #
            # The learning rate is not cosmetic. Momentum turns the update into
            # a geometric series with steady-state gain lr / (1 - beta), so
            # lr=1.0 with beta=0.9 amplifies every round 10x and the run
            # diverges after ~40 rounds. lr=0.1 keeps the gain at 1.0.
            pseudo_gradient = [
                g - a for g, a in zip(global_weights, aggregated, strict=True)
            ]
            if momentum is None:
                momentum = pseudo_gradient
            else:
                momentum = [
                    beta * m + p for m, p in zip(momentum, pseudo_gradient, strict=True)
                ]
            global_weights = [
                g - lr * m for g, m in zip(global_weights, momentum, strict=True)
            ]
        else:
            global_weights = aggregated

        round_loss = float(np.mean([m["loss"] for _, _, m in results]))
        round_auc = float(np.nanmean([m["auc"] for _, _, m in results]))
        losses.append((rnd, round_loss))
        aucs.append((rnd, round_auc))
        round_elapsed = time.perf_counter() - round_start
        round_times.append((rnd, round_elapsed))

        if producer is not None:
            producer.send_round_summary(
                round=rnd,
                global_auc=round_auc,
                global_loss=round_loss,
                n_clients_selected=len(selected),
                total_bytes=sum(
                    sum(w.nbytes for w in weights) for weights, _, _ in results
                ),
                round_duration_sec=round_elapsed,
            )

        logger.info(
            "Round %s/%s: clients=%s loss=%.4f auc=%.4f",
            rnd,
            rounds,
            selected,
            round_loss,
            round_auc,
        )

    total_seconds = time.perf_counter() - run_start

    telemetry_stats: dict[str, int] = {}
    if producer is not None:
        telemetry_stats = producer.stats()
        producer.close()

    return {
        "strategy": strategy_name,
        "partition": partition,
        "n_rounds": rounds,
        "num_clients": k,
        "n_selected_per_round": n_selected,
        "losses_distributed": losses,
        "metrics_distributed": {"auc": aucs},
        "round_times": round_times,
        "total_seconds": total_seconds,
        "telemetry": telemetry_stats,
        "epsilons": [c.epsilon for c in clients],
        "final_weights": global_weights,
        "input_dim": input_dim,
    }


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.fl.s11_server``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run a sequential FL simulation")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument(
        "--strategy", choices=["fedavg", "fedprox", "fedavgm"], default="fedavg"
    )
    parser.add_argument("--partition", default="iid")
    parser.add_argument(
        "--ray",
        action="store_true",
        help="Use Flower's Ray-backed simulation instead of the sequential loop",
    )
    args = parser.parse_args()

    runner = run_simulation if args.ray else run_sequential
    history = runner(
        n_rounds=args.rounds,
        num_clients=args.clients,
        strategy_name=args.strategy,
        partition=args.partition,
    )

    print("\n" + "=" * 70)
    print("FL SIMULATION COMPLETE")
    print("=" * 70)
    print(f"  strategy   : {history['strategy']}")
    print(f"  partition  : {history['partition']}")
    print(f"  rounds     : {history['n_rounds']}")
    print(f"  clients    : {history['num_clients']}")
    print(f"  losses     : {history['losses_distributed']}")
    for key, values in history["metrics_distributed"].items():
        print(f"  {key:11}: {values}")
    print("=" * 70)


if __name__ == "__main__":
    main()
