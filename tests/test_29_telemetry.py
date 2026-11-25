"""Tests for the Kafka telemetry producer and consumer.

The broker is not required: the producer is designed to degrade to a no-op when
Kafka is unreachable, and that behaviour is what most of these tests pin down.
The consumer's SQL builders are tested without a database.

Run with::

    pytest tests/test_29_telemetry.py -v
"""

from __future__ import annotations

import pytest

from src.telemetry.s29_producer import (
    CLIENT_FIELDS,
    ROUND_FIELDS,
    TelemetryProducer,
    validate_client_message,
    validate_round_message,
)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------
def _client_payload(**overrides):
    payload = {
        "round": 1,
        "client_id": 0,
        "local_auc": 0.80,
        "local_loss": 0.48,
        "n_samples": 90_000,
        "bytes_sent": 72_196,
        "duration_sec": 1.5,
    }
    payload.update(overrides)
    return payload


def _round_payload(**overrides):
    payload = {
        "round": 1,
        "global_auc": 0.81,
        "global_loss": 0.48,
        "n_clients_selected": 5,
        "total_bytes": 360_980,
        "round_duration_sec": 7.1,
    }
    payload.update(overrides)
    return payload


def test_client_schema_lists_every_required_field() -> None:
    assert set(CLIENT_FIELDS) == {
        "round", "client_id", "local_auc", "local_loss",
        "n_samples", "bytes_sent", "duration_sec",
    }


def test_round_schema_lists_every_required_field() -> None:
    assert set(ROUND_FIELDS) == {
        "round", "global_auc", "global_loss",
        "n_clients_selected", "total_bytes", "round_duration_sec",
    }


def test_valid_client_message_passes() -> None:
    validate_client_message(_client_payload())


def test_valid_round_message_passes() -> None:
    validate_round_message(_round_payload())


@pytest.mark.parametrize("field", CLIENT_FIELDS)
def test_client_message_rejects_missing_field(field: str) -> None:
    payload = _client_payload()
    payload.pop(field)
    with pytest.raises(ValueError, match="missing fields"):
        validate_client_message(payload)


@pytest.mark.parametrize("field", ROUND_FIELDS)
def test_round_message_rejects_missing_field(field: str) -> None:
    payload = _round_payload()
    payload.pop(field)
    with pytest.raises(ValueError, match="missing fields"):
        validate_round_message(payload)


def test_client_rejects_round_zero() -> None:
    with pytest.raises(ValueError, match="round must be"):
        validate_client_message(_client_payload(round=0))


def test_client_rejects_negative_client_id() -> None:
    with pytest.raises(ValueError, match="client_id must be"):
        validate_client_message(_client_payload(client_id=-1))


def test_client_rejects_auc_above_one() -> None:
    with pytest.raises(ValueError, match="local_auc must be"):
        validate_client_message(_client_payload(local_auc=1.5))


def test_client_rejects_zero_samples() -> None:
    with pytest.raises(ValueError, match="n_samples must be"):
        validate_client_message(_client_payload(n_samples=0))


def test_round_rejects_auc_above_one() -> None:
    with pytest.raises(ValueError, match="global_auc must be"):
        validate_round_message(_round_payload(global_auc=1.2))


def test_round_rejects_zero_clients() -> None:
    with pytest.raises(ValueError, match="n_clients_selected must be"):
        validate_round_message(_round_payload(n_clients_selected=0))


# ---------------------------------------------------------------------------
# Producer degradation
# ---------------------------------------------------------------------------
def test_disabled_producer_sends_nothing() -> None:
    producer = TelemetryProducer(enabled=False)

    assert producer.available is False
    assert producer.send_round_summary(**_round_payload()) is False
    assert producer.stats() == {"sent": 0, "failed": 1}


def test_producer_with_unreachable_broker_does_not_raise() -> None:
    """A dead broker must degrade to a no-op, never abort the training run."""
    producer = TelemetryProducer("localhost:1", enabled=True)

    assert producer.available is False
    assert producer.send_client_metrics(**_client_payload()) is False
    assert producer.stats()["failed"] == 1


def test_producer_validates_before_sending() -> None:
    """A malformed message raises even when the broker is down.

    Validation is a programming error, not a transport error, so it must not be
    swallowed the way a send failure is.
    """
    producer = TelemetryProducer(enabled=False)

    with pytest.raises(ValueError, match="local_auc must be"):
        producer.send_client_metrics(**_client_payload(local_auc=2.0))


def test_producer_context_manager_closes_cleanly() -> None:
    with TelemetryProducer(enabled=False) as producer:
        assert producer.available is False


# ---------------------------------------------------------------------------
# Consumer SQL
# ---------------------------------------------------------------------------
def test_consumer_ddl_declares_both_tables() -> None:
    from src.telemetry.s30_consumer import CLIENT_TABLE, DDL, ROUND_TABLE

    assert f"CREATE TABLE IF NOT EXISTS {CLIENT_TABLE}" in DDL
    assert f"CREATE TABLE IF NOT EXISTS {ROUND_TABLE}" in DDL


def test_consumer_ddl_has_natural_keys() -> None:
    """Replaying a topic must update, not duplicate, so both tables need keys."""
    from src.telemetry.s30_consumer import DDL

    assert "PRIMARY KEY (round, client_id)" in DDL
    assert "round               INTEGER PRIMARY KEY" in DDL


def test_consumer_defaults_to_the_bi_database() -> None:
    """The dashboards read these tables, so the DSN must match the BI export."""
    from src.serving.s23_export_results import DEFAULT_DSN as BI_DSN
    from src.telemetry.s30_consumer import DEFAULT_DSN as TELEMETRY_DSN

    assert TELEMETRY_DSN == BI_DSN
