"""Kafka consumer that drains FL telemetry into postgres.

Pipeline position: **PUBLISH (telemetry) → STORE → DASHBOARD**

Grafana and Superset read SQL, not Kafka. This consumer is the bridge: it
subscribes to both topics and appends each message to a table, so the
dashboards can query the same postgres instance MLflow already uses.

Why postgres and not a file
---------------------------
A JSONL file would work for a one-off run, but the dashboards need to filter
and aggregate. Postgres gives that for free and the connection is already
provisioned, so there is no second store to keep alive.

Idempotency
-----------
Both tables have a natural key — ``(round, client_id)`` for client metrics and
``round`` for round summaries — declared as a primary key. Replaying a topic
therefore updates rather than duplicates, which matters because the consumer
commits offsets only after a successful write.

Usage
-----
>>> from src.telemetry.s30_consumer import TelemetryConsumer
>>> consumer = TelemetryConsumer()
>>> consumer.ensure_tables()
>>> consumer.run(max_messages=10)   # returns after 10 messages
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.config import get_config

logger = logging.getLogger(__name__)

# Same DSN the BI export uses; the dashboards read these tables from here.
DEFAULT_DSN: str = "postgresql+psycopg2://mlflow:mlflow@localhost:5432/mlflow"

CLIENT_TABLE: str = "fl_telemetry"
ROUND_TABLE: str = "fl_round_summary"

DDL = f"""
CREATE TABLE IF NOT EXISTS {CLIENT_TABLE} (
    round        INTEGER NOT NULL,
    client_id    INTEGER NOT NULL,
    local_auc    DOUBLE PRECISION NOT NULL,
    local_loss   DOUBLE PRECISION NOT NULL,
    n_samples    INTEGER NOT NULL,
    bytes_sent   BIGINT NOT NULL,
    duration_sec DOUBLE PRECISION NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (round, client_id)
);

CREATE TABLE IF NOT EXISTS {ROUND_TABLE} (
    round               INTEGER PRIMARY KEY,
    global_auc          DOUBLE PRECISION NOT NULL,
    global_loss         DOUBLE PRECISION NOT NULL,
    n_clients_selected  INTEGER NOT NULL,
    total_bytes         BIGINT NOT NULL,
    round_duration_sec  DOUBLE PRECISION NOT NULL,
    ingested_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


# =============================================================================
# Consumer
# =============================================================================
class TelemetryConsumer:
    """Reads both telemetry topics and writes them to postgres.

    Parameters
    ----------
    dsn:
        SQLAlchemy connection string. Defaults to the compose postgres.
    group_id:
        Consumer group. Two consumers in the same group split the partitions;
        different groups each see every message.
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        group_id: str = "fl-telemetry-writer",
    ) -> None:
        cfg = get_config()
        self.dsn = dsn or DEFAULT_DSN
        self.group_id = group_id
        self.topic_client = cfg.kafka.topics.telemetry
        self.topic_round = cfg.kafka.topics.round_summary
        self.bootstrap = cfg.kafka.bootstrap_servers
        self._engine: Any = None
        self.written = {"client": 0, "round": 0}

    # ------------------------------------------------------------------
    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sqlalchemy is required; install with "
                    "`pip install sqlalchemy psycopg2-binary`"
                ) from exc
            self._engine = create_engine(self.dsn)
        return self._engine

    # ------------------------------------------------------------------
    def ensure_tables(self) -> None:
        """Create both tables if they do not exist."""
        from sqlalchemy import text

        with self._get_engine().begin() as conn:
            for statement in DDL.strip().split(";"):
                if statement.strip():
                    conn.execute(text(statement))
        logger.info("Ensured tables %s and %s", CLIENT_TABLE, ROUND_TABLE)

    # ------------------------------------------------------------------
    def write_client(self, payload: dict[str, Any]) -> None:
        """Upsert one client metric row."""
        from sqlalchemy import text

        with self._get_engine().begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {CLIENT_TABLE} "
                    "(round, client_id, local_auc, local_loss, n_samples, "
                    " bytes_sent, duration_sec) "
                    "VALUES (:round, :client_id, :local_auc, :local_loss, "
                    " :n_samples, :bytes_sent, :duration_sec) "
                    "ON CONFLICT (round, client_id) DO UPDATE SET "
                    " local_auc = EXCLUDED.local_auc, "
                    " local_loss = EXCLUDED.local_loss, "
                    " n_samples = EXCLUDED.n_samples, "
                    " bytes_sent = EXCLUDED.bytes_sent, "
                    " duration_sec = EXCLUDED.duration_sec"
                ),
                payload,
            )
        self.written["client"] += 1

    # ------------------------------------------------------------------
    def write_round(self, payload: dict[str, Any]) -> None:
        """Upsert one round summary row."""
        from sqlalchemy import text

        with self._get_engine().begin() as conn:
            conn.execute(
                text(
                    f"INSERT INTO {ROUND_TABLE} "
                    "(round, global_auc, global_loss, n_clients_selected, "
                    " total_bytes, round_duration_sec) "
                    "VALUES (:round, :global_auc, :global_loss, "
                    " :n_clients_selected, :total_bytes, :round_duration_sec) "
                    "ON CONFLICT (round) DO UPDATE SET "
                    " global_auc = EXCLUDED.global_auc, "
                    " global_loss = EXCLUDED.global_loss, "
                    " n_clients_selected = EXCLUDED.n_clients_selected, "
                    " total_bytes = EXCLUDED.total_bytes, "
                    " round_duration_sec = EXCLUDED.round_duration_sec"
                ),
                payload,
            )
        self.written["round"] += 1

    # ------------------------------------------------------------------
    def run(
        self,
        *,
        max_messages: int | None = None,
        timeout_ms: int = 10_000,
    ) -> dict[str, int]:
        """Consume until ``max_messages`` is reached or the poll times out.

        Parameters
        ----------
        max_messages:
            Stop after this many messages. ``None`` runs until interrupted.
        timeout_ms:
            How long to wait for a message before giving up. With
            ``max_messages`` set, this bounds the whole call.

        Returns
        -------
        dict
            ``{client, round}`` counts written.
        """
        try:
            from kafka import KafkaConsumer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "kafka-python is required; install with `pip install kafka-python`"
            ) from exc

        self.ensure_tables()

        consumer = KafkaConsumer(
            self.topic_client,
            self.topic_round,
            bootstrap_servers=self.bootstrap,
            group_id=self.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            consumer_timeout_ms=timeout_ms,
        )

        seen = 0
        try:
            for message in consumer:
                payload = message.value
                if message.topic == self.topic_client:
                    self.write_client(payload)
                else:
                    self.write_round(payload)

                consumer.commit()
                seen += 1
                if max_messages is not None and seen >= max_messages:
                    break
        finally:
            consumer.close()

        logger.info(
            "Consumed %s messages (client=%s, round=%s)",
            seen,
            self.written["client"],
            self.written["round"],
        )
        return dict(self.written)


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.telemetry.s30_consumer``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Drain Kafka telemetry to postgres")
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--timeout-ms", type=int, default=10_000)
    parser.add_argument("--dsn", default=None)
    args = parser.parse_args()

    consumer = TelemetryConsumer(dsn=args.dsn)
    written = consumer.run(
        max_messages=args.max_messages, timeout_ms=args.timeout_ms
    )

    print("\n" + "=" * 70)
    print("TELEMETRY CONSUMED")
    print("=" * 70)
    print(f"  client rows : {written['client']}")
    print(f"  round rows  : {written['round']}")


if __name__ == "__main__":
    main()
