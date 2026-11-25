"""Kafka producer for FL telemetry.

Pipeline position: **TRAIN (local) → AGGREGATE → PUBLISH (telemetry)**

Two topics, both metrics-only:

* ``fl_telemetry`` — one message per client per round
* ``fl_round_summary`` — one message per round, after aggregation

Fire-and-forget, deliberately
-----------------------------
A training run takes minutes to hours. If Kafka is down, or the broker is slow,
the run must still finish — telemetry is observability, not a dependency. Every
send is wrapped so a broker failure logs a warning and returns ``False`` rather
than raising. The caller can count failures but never has to handle them.

This is the opposite of the gRPC path, where a transport failure *must* abort
the round: weights are the training, metrics are not.

Usage
-----
>>> from src.telemetry.s29_producer import TelemetryProducer
>>> producer = TelemetryProducer()
>>> producer.send_round_summary(round=1, global_auc=0.78, global_loss=0.51,
...                             n_clients_selected=5, total_bytes=360980,
...                             round_duration_sec=7.1)
True
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config import get_config

logger = logging.getLogger(__name__)

# Required fields per topic, mirroring docs/08_kafka_telemetry.md. Validating
# here means a malformed message is caught at the producer rather than showing
# up as a null column in the dashboard three hours later.
CLIENT_FIELDS: tuple[str, ...] = (
    "round",
    "client_id",
    "local_auc",
    "local_loss",
    "n_samples",
    "bytes_sent",
    "duration_sec",
)

ROUND_FIELDS: tuple[str, ...] = (
    "round",
    "global_auc",
    "global_loss",
    "n_clients_selected",
    "total_bytes",
    "round_duration_sec",
)


# =============================================================================
# Validation
# =============================================================================
def validate_client_message(payload: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a client message is missing or malformed."""
    missing = [f for f in CLIENT_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"fl_telemetry missing fields: {missing}")

    if payload["round"] < 1:
        raise ValueError(f"round must be >= 1, got {payload['round']}")
    if payload["client_id"] < 0:
        raise ValueError(f"client_id must be >= 0, got {payload['client_id']}")
    if not 0.0 <= payload["local_auc"] <= 1.0:
        raise ValueError(f"local_auc must be in [0, 1], got {payload['local_auc']}")
    if payload["n_samples"] < 1:
        raise ValueError(f"n_samples must be >= 1, got {payload['n_samples']}")


def validate_round_message(payload: dict[str, Any]) -> None:
    """Raise ``ValueError`` when a round summary is missing or malformed."""
    missing = [f for f in ROUND_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"fl_round_summary missing fields: {missing}")

    if payload["round"] < 1:
        raise ValueError(f"round must be >= 1, got {payload['round']}")
    if not 0.0 <= payload["global_auc"] <= 1.0:
        raise ValueError(
            f"global_auc must be in [0, 1], got {payload['global_auc']}"
        )
    if payload["n_clients_selected"] < 1:
        raise ValueError(
            f"n_clients_selected must be >= 1, got {payload['n_clients_selected']}"
        )


# =============================================================================
# Producer
# =============================================================================
class TelemetryProducer:
    """Publishes FL metrics to Kafka, tolerating a missing broker.

    Parameters
    ----------
    bootstrap_servers:
        Override for the configured broker list.
    enabled:
        When ``False`` every send is a no-op returning ``False``. Useful for
        tests and for runs where telemetry is not wanted.
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        cfg = get_config()
        self.topic_client = cfg.kafka.topics.telemetry
        self.topic_round = cfg.kafka.topics.round_summary
        self.enabled = enabled
        self._producer: Any = None
        self.sent = 0
        self.failed = 0

        if not enabled:
            logger.info("Telemetry producer disabled")
            return

        servers = bootstrap_servers or cfg.kafka.bootstrap_servers
        try:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Short timeouts: a slow broker must not stall training.
                request_timeout_ms=5000,
                max_block_ms=5000,
                retries=1,
            )
            logger.info("Telemetry producer connected to %s", servers)
        except Exception as exc:  # noqa: BLE001 - telemetry must never abort a run
            logger.warning(
                "Kafka unavailable at %s (%s) — telemetry disabled for this run",
                servers,
                exc,
            )
            self._producer = None

    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        """Whether a broker connection was established."""
        return self._producer is not None

    # ------------------------------------------------------------------
    def _send(self, topic: str, payload: dict[str, Any]) -> bool:
        """Publish one message, returning ``False`` on any failure."""
        if self._producer is None:
            self.failed += 1
            return False
        try:
            self._producer.send(topic, payload)
            self.sent += 1
            return True
        except Exception as exc:  # noqa: BLE001
            self.failed += 1
            logger.warning("Kafka send to %s failed: %s", topic, exc)
            return False

    # ------------------------------------------------------------------
    def send_client_metrics(
        self,
        *,
        round: int,
        client_id: int,
        local_auc: float,
        local_loss: float,
        n_samples: int,
        bytes_sent: int,
        duration_sec: float,
    ) -> bool:
        """Publish one client's metrics for one round."""
        payload = {
            "round": int(round),
            "client_id": int(client_id),
            "local_auc": float(local_auc),
            "local_loss": float(local_loss),
            "n_samples": int(n_samples),
            "bytes_sent": int(bytes_sent),
            "duration_sec": float(duration_sec),
        }
        validate_client_message(payload)
        return self._send(self.topic_client, payload)

    # ------------------------------------------------------------------
    def send_round_summary(
        self,
        *,
        round: int,
        global_auc: float,
        global_loss: float,
        n_clients_selected: int,
        total_bytes: int,
        round_duration_sec: float,
    ) -> bool:
        """Publish the aggregated metrics for one round."""
        payload = {
            "round": int(round),
            "global_auc": float(global_auc),
            "global_loss": float(global_loss),
            "n_clients_selected": int(n_clients_selected),
            "total_bytes": int(total_bytes),
            "round_duration_sec": float(round_duration_sec),
        }
        validate_round_message(payload)
        return self._send(self.topic_round, payload)

    # ------------------------------------------------------------------
    def flush(self, timeout: float = 5.0) -> None:
        """Block until buffered messages are delivered."""
        if self._producer is not None:
            try:
                self._producer.flush(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Kafka flush failed: %s", exc)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Flush and close the connection."""
        if self._producer is not None:
            self.flush()
            try:
                self._producer.close(timeout=5)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Kafka close failed: %s", exc)
            self._producer = None

    # ------------------------------------------------------------------
    def __enter__(self) -> "TelemetryProducer":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, int]:
        """Counters for the run summary."""
        return {"sent": self.sent, "failed": self.failed}


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.telemetry.s29_producer``.

    Sends one synthetic round so the broker and consumer can be checked without
    running a full federated job.
    """
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Send one test round to Kafka")
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()

    with TelemetryProducer() as producer:
        if not producer.available:
            print("Kafka unavailable — nothing sent")
            return

        for cid in range(3):
            producer.send_client_metrics(
                round=args.round,
                client_id=cid,
                local_auc=0.80 + cid * 0.01,
                local_loss=0.48,
                n_samples=90_000,
                bytes_sent=72_196,
                duration_sec=1.5,
            )
        producer.send_round_summary(
            round=args.round,
            global_auc=0.81,
            global_loss=0.48,
            n_clients_selected=3,
            total_bytes=216_588,
            round_duration_sec=4.5,
        )

    print("\n" + "=" * 70)
    print("TELEMETRY SENT")
    print("=" * 70)
    print(f"  stats: {producer.stats()}")


if __name__ == "__main__":
    main()
