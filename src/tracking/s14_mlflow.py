"""MLflow experiment tracking with a JSON fallback.

Pipeline position: **TRAIN → EVALUATE → LOG**

The MLflow server runs in Docker and is not always up. A tracking failure must
never kill an experiment that took 20 minutes to run, so every call here falls
back to writing a JSON file under ``results/runs/``.

Required tags (from the project spec)
-------------------------------------
``experiment, partition, alpha, strategy, K, noise_multiplier``

Usage
-----
>>> from src.tracking.s14_mlflow import log_run
>>> run_id = log_run(
...     experiment="E1",
...     params={"partition": "iid", "strategy": "fedavg", "K": 10},
...     metrics={"auc": 0.89},
... )
"""

from __future__ import annotations

import json
import logging
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import get_config

logger = logging.getLogger(__name__)

REQUIRED_TAGS: tuple[str, ...] = (
    "experiment",
    "partition",
    "alpha",
    "strategy",
    "K",
    "noise_multiplier",
)


# =============================================================================
# Server availability
# =============================================================================
def mlflow_reachable(uri: str | None = None, timeout: float = 1.0) -> bool:
    """Return ``True`` when the MLflow tracking server accepts connections."""
    cfg = get_config()
    target = uri or cfg.mlflow.tracking_uri

    # Parse "http://host:port"
    stripped = target.replace("http://", "").replace("https://", "")
    host, _, port_str = stripped.partition(":")
    port = int(port_str) if port_str else 5000

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


# =============================================================================
# Fallback
# =============================================================================
def _write_json_fallback(
    experiment: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    tags: dict[str, Any],
    artifacts: dict[str, Any] | None,
) -> str:
    """Persist a run to ``results/runs/{run_id}.json``."""
    cfg = get_config()
    run_id = f"{experiment}-{uuid.uuid4().hex[:8]}"
    out_dir = cfg.paths.results_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "experiment": experiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "metrics": metrics,
        "tags": tags,
        "artifacts": artifacts or {},
        "backend": "json_fallback",
    }
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.warning("MLflow unavailable — wrote fallback %s", path)
    return run_id


# =============================================================================
# Logging
# =============================================================================
def log_run(
    experiment: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    *,
    tags: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    force_fallback: bool = False,
) -> str:
    """Log one run to MLflow, falling back to JSON when the server is down.

    Parameters
    ----------
    experiment:
        Experiment name, e.g. ``"E1"``. Also used as the MLflow experiment.
    params:
        Hyper-parameters and configuration. Values are stringified by MLflow.
    metrics:
        Numeric results. Non-finite values are dropped because MLflow rejects
        NaN and inf.
    tags:
        Extra tags. The six required tags are added automatically from
        ``params`` when present.
    artifacts:
        Small JSON-serialisable payloads stored alongside the run.
    force_fallback:
        Skip the MLflow attempt entirely. Useful in tests.

    Returns
    -------
    str
        The run id (MLflow's or the generated fallback id).
    """
    cfg = get_config()

    # Build the required tag set from params, defaulting to "n/a"
    resolved_tags: dict[str, Any] = dict(tags or {})
    for key in REQUIRED_TAGS:
        resolved_tags.setdefault(key, params.get(key, "n/a"))
    resolved_tags["experiment"] = experiment

    # MLflow rejects NaN / inf
    clean_metrics = {
        k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v
    }

    if force_fallback or not mlflow_reachable():
        return _write_json_fallback(
            experiment, params, clean_metrics, resolved_tags, artifacts
        )

    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(experiment)

        with mlflow.start_run() as run:
            mlflow.log_params({k: str(v) for k, v in params.items()})
            mlflow.set_tags({k: str(v) for k, v in resolved_tags.items()})
            if clean_metrics:
                mlflow.log_metrics(clean_metrics)
            if artifacts:
                mlflow.log_dict(artifacts, "artifacts.json")
            run_id = run.info.run_id

        logger.info("Logged run %s to MLflow experiment %r", run_id, experiment)
        return run_id

    except Exception as exc:  # noqa: BLE001 - tracking must never kill a run
        logger.warning("MLflow logging failed (%s) — using JSON fallback", exc)
        return _write_json_fallback(
            experiment, params, clean_metrics, resolved_tags, artifacts
        )


def log_metrics_step(
    metrics: dict[str, float],
    step: int,
    *,
    run_id: str | None = None,
) -> None:
    """Log per-round metrics against an existing run.

    Silently does nothing when MLflow is unavailable — per-round logging is
    best-effort, the final metrics are what matter.
    """
    if not mlflow_reachable():
        return
    try:
        import mlflow

        cfg = get_config()
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        clean = {k: float(v) for k, v in metrics.items() if v == v}
        if run_id:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metrics(clean, step=step)
        else:
            mlflow.log_metrics(clean, step=step)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Per-step metric logging skipped: %s", exc)


def main() -> None:
    """CLI entry point: ``python -m src.tracking.s14_mlflow``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    reachable = mlflow_reachable()
    run_id = log_run(
        experiment="smoke_test",
        params={
            "partition": "iid",
            "alpha": "n/a",
            "strategy": "fedavg",
            "K": 10,
            "noise_multiplier": 0.0,
        },
        metrics={"auc": 0.89, "f1": 0.6},
        tags={"note": "smoke test"},
    )

    print("\n" + "=" * 70)
    print("MLFLOW TRACKING SMOKE TEST")
    print("=" * 70)
    print(f"  server reachable : {reachable}")
    print(f"  backend          : {'mlflow' if reachable else 'json_fallback'}")
    print(f"  run_id           : {run_id}")
    print("=" * 70)


if __name__ == "__main__":
    main()
