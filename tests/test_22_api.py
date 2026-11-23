"""Tests for :mod:`src.serving.s22_api`.

These run against the exported artifacts, so they are skipped when the export
step has not been run. That keeps the suite green on a fresh clone without
hiding a real failure once the artifacts exist.

Run with::

    pytest tests/test_22_api.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.config import get_config  # noqa: E402
from src.serving.s22_api import app  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _artifacts_exist() -> bool:
    cfg = get_config()
    return (
        (cfg.paths.results_dir / "models" / "fl_fedavg.pt").exists()
        and (cfg.paths.processed_dir / "preprocessing_iid.json").exists()
    )


pytestmark = pytest.mark.skipif(
    not _artifacts_exist(),
    reason="serving artifacts missing; run `python -m src.serving.s21_export_model`",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_customer() -> dict:
    """A realistic customer row taken from test_real."""
    import pandas as pd

    from src.data.s01_clean import FEATURE_COLS

    cfg = get_config()
    df = pd.read_parquet(cfg.paths.test_real_parquet)
    row = df.iloc[0]
    return {col: row[col].item() if hasattr(row[col], "item") else row[col]
            for col in FEATURE_COLS}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
def test_health_reports_ok(client) -> None:
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ---------------------------------------------------------------------------
# /model-info
# ---------------------------------------------------------------------------
def test_model_info_reports_metrics(client) -> None:
    body = client.get("/model-info").json()

    assert body["n_params"] == 18_049
    assert body["strategy"] == "fedavg"
    assert body["partition"] == "iid"
    assert 0.0 < body["metrics_test_real"]["auc"] < 1.0


def test_model_info_auc_matches_e1(client) -> None:
    """The served model must be the one the report quotes (E1 fold 0)."""
    body = client.get("/model-info").json()

    assert body["metrics_test_real"]["auc"] == pytest.approx(0.8949, abs=0.01)


# ---------------------------------------------------------------------------
# /predict
# ---------------------------------------------------------------------------
def test_predict_returns_probability(client, sample_customer) -> None:
    body = client.post("/predict", json=sample_customer).json()

    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn_prediction"], bool)
    assert body["threshold"] == 0.5


def test_predict_label_matches_threshold(client, sample_customer) -> None:
    body = client.post("/predict", json=sample_customer).json()

    assert body["churn_prediction"] == (
        body["churn_probability"] >= body["threshold"]
    )


def test_predict_is_deterministic(client, sample_customer) -> None:
    """Two identical requests must give the same probability."""
    a = client.post("/predict", json=sample_customer).json()
    b = client.post("/predict", json=sample_customer).json()

    assert a["churn_probability"] == b["churn_probability"]


def test_predict_rejects_missing_feature(client, sample_customer) -> None:
    """A missing field must be a 422, not a silent default."""
    broken = dict(sample_customer)
    broken.pop("Age")

    response = client.post("/predict", json=broken)

    assert response.status_code == 422


def test_predict_rejects_unknown_feature(client, sample_customer) -> None:
    """Extra fields are rejected so a typo cannot be silently ignored."""
    broken = dict(sample_customer)
    broken["Not_A_Feature"] = 1.0

    response = client.post("/predict", json=broken)

    assert response.status_code == 422


def test_predict_accepts_categorical_strings(client, sample_customer) -> None:
    """Categoricals arrive as strings and must not be coerced to numbers."""
    body = client.post("/predict", json=sample_customer).json()

    assert "churn_probability" in body


# ---------------------------------------------------------------------------
# Artifact contract
# ---------------------------------------------------------------------------
def test_metadata_file_matches_served_model(client) -> None:
    cfg = get_config()
    meta_path = cfg.paths.results_dir / "models" / "model_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    body = client.get("/model-info").json()

    assert body["model_file"] == meta["model_file"]
    assert body["n_params"] == meta["n_params"]
