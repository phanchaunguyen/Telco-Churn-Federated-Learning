"""FastAPI prediction endpoint for the federated churn model.

Pipeline position: **EXPORT → SERVE**

The API loads the artifacts written by :mod:`src.serving.s21_export_model` at
start-up and exposes a single prediction route. It never trains and never
touches a client silo — it is a pure inference surface over the global model
that FedAvg produced.

Preprocessing must match training exactly
-----------------------------------------
The model was trained on features encoded with the shared vocabulary and scaled
with the canonical ``train_real`` scaler. Serving applies the same two steps in
the same order. Using a different scaler here would silently shift every input
and degrade predictions without any error being raised, which is why the scaler
is loaded from the exported artifact rather than refitted.

Usage
-----
>>> from fastapi.testclient import TestClient
>>> from src.serving.s22_api import app
>>> client = TestClient(app)
>>> client.get("/health").json()["status"]
'ok'
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_config
from src.features.s07_preprocess import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    LocalScaler,
    Vocabulary,
    encode,
    load_preprocessing,
)
from src.models.s08_mlp import load_model

logger = logging.getLogger(__name__)

TARGET_COL = "churn"

# Decision threshold for the binary label. 0.5 is the default the report uses
# for F1; the probability itself is always returned so a caller can pick their
# own cut-off for a retention campaign.
DEFAULT_THRESHOLD: float = 0.5


# =============================================================================
# Request / response schemas
# =============================================================================
class CustomerFeatures(BaseModel):
    """One customer's 29 raw features, before encoding.

    Field names match the source dataset so a caller can pass a row through
    unchanged. Categorical fields are strings; numeric fields are floats.
    """

    model_config = {"extra": "forbid"}

    # Demographics
    Gender: str
    Age: float
    Senior_Citizen: str = Field(alias="Senior Citizen")
    Married: str
    Number_of_Dependents: float = Field(alias="Number of Dependents")

    # Account
    Tenure_in_Months: float = Field(alias="Tenure in Months")
    Offer: str
    Contract: str
    Paperless_Billing: str = Field(alias="Paperless Billing")
    Payment_Method: str = Field(alias="Payment Method")
    Number_of_Referrals: float = Field(alias="Number of Referrals")

    # Services
    Phone_Service: str = Field(alias="Phone Service")
    Multiple_Lines: str = Field(alias="Multiple Lines")
    Internet_Service: str = Field(alias="Internet Service")
    Internet_Type: str = Field(alias="Internet Type")
    Online_Security: str = Field(alias="Online Security")
    Online_Backup: str = Field(alias="Online Backup")
    Device_Protection_Plan: str = Field(alias="Device Protection Plan")
    Premium_Tech_Support: str = Field(alias="Premium Tech Support")
    Streaming_TV: str = Field(alias="Streaming TV")
    Streaming_Movies: str = Field(alias="Streaming Movies")
    Streaming_Music: str = Field(alias="Streaming Music")
    Unlimited_Data: str = Field(alias="Unlimited Data")

    # Usage & billing
    Avg_Monthly_Long_Distance_Charges: float = Field(
        alias="Avg Monthly Long Distance Charges"
    )
    Avg_Monthly_GB_Download: float = Field(alias="Avg Monthly GB Download")
    Monthly_Charge: float = Field(alias="Monthly Charge")
    Total_Charges: float = Field(alias="Total Charges")
    Total_Extra_Data_Charges: float = Field(alias="Total Extra Data Charges")
    Total_Long_Distance_Charges: float = Field(alias="Total Long Distance Charges")


class PredictionResponse(BaseModel):
    """Prediction for one customer."""

    churn_probability: float
    churn_prediction: bool
    threshold: float
    model: str


class ModelInfoResponse(BaseModel):
    """What the API is currently serving."""

    model_file: str
    input_dim: int
    n_params: int
    rounds: int
    seed: int
    strategy: str
    partition: str
    metrics_test_real: dict[str, Any]


# =============================================================================
# Application state
# =============================================================================
class _State:
    """Artifacts loaded once at start-up."""

    model: Any = None
    vocab: Vocabulary | None = None
    scaler: LocalScaler | None = None
    metadata: dict[str, Any] = {}


state = _State()


def _artifact_paths() -> tuple[Path, Path, Path]:
    """Return ``(model, preprocessing, metadata)`` paths."""
    cfg = get_config()
    model_dir = cfg.paths.results_dir / "models"
    return (
        model_dir / "fl_fedavg.pt",
        cfg.paths.processed_dir / "preprocessing_iid.json",
        model_dir / "model_metadata.json",
    )


def load_artifacts() -> None:
    """Load model, vocabulary, scaler and metadata into module state.

    Raises
    ------
    FileNotFoundError
        If the export step has not been run. The message names the command to
        run rather than failing with a bare path error.
    """
    model_file, prep_file, meta_file = _artifact_paths()

    missing = [p for p in (model_file, prep_file) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Serving artifacts missing: "
            + ", ".join(str(p) for p in missing)
            + ". Run `python -m src.serving.s21_export_model` first."
        )

    state.model = load_model(model_file)
    vocab, scalers = load_preprocessing(prep_file)

    # The canonical scaler is the one fitted on train_real. It is stored under
    # client id 0 by the export step; any client scaler would be wrong here
    # because a new customer belongs to no silo.
    if 0 not in scalers:
        raise KeyError(
            f"{prep_file} has no scaler for client 0; cannot serve predictions"
        )
    state.vocab = vocab
    state.scaler = scalers[0]

    state.metadata = (
        json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    )
    logger.info(
        "Serving model loaded: %s params, AUC %.4f",
        state.metadata.get("n_params", "?"),
        state.metadata.get("metrics_test_real", {}).get("auc", float("nan")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts on start-up so the first request is not slow."""
    load_artifacts()
    yield


app = FastAPI(
    title="FL Telco Churn API",
    description=(
        "Prediction endpoint for the federated churn model. "
        "The model was trained with FedAvg over 10 silos; only weights ever "
        "left a silo, never data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# =============================================================================
# Routes
# =============================================================================
@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness plus whether the model is actually loaded."""
    return {
        "status": "ok" if state.model is not None else "model_not_loaded",
        "model_loaded": state.model is not None,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Report what is being served, including the hold-out metrics."""
    if not state.metadata:
        raise HTTPException(status_code=503, detail="Model metadata not loaded")
    return ModelInfoResponse(**state.metadata)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """Score one customer.

    The 29 raw features are encoded with the shared vocabulary, scaled with the
    canonical scaler, and passed through the global model. The probability is
    returned alongside the thresholded label so a caller can apply their own
    cut-off.
    """
    if state.model is None or state.vocab is None or state.scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a one-row frame with the exact column names the encoder expects
    row = customer.model_dump(by_alias=True)
    frame = pd.DataFrame([row])

    missing = [c for c in FEATURE_COLS if c not in frame.columns]
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Missing features: {missing}"
        )

    # Categoricals must be strings; the encoder compares them to vocabulary
    # entries and a float would never match.
    for col in CATEGORICAL_COLS:
        frame[col] = frame[col].astype(str)

    X = state.scaler.transform(encode(frame, state.vocab)).astype(np.float32)

    with torch.no_grad():
        proba = float(state.model(torch.from_numpy(X)).squeeze().item())

    return PredictionResponse(
        churn_probability=proba,
        churn_prediction=proba >= DEFAULT_THRESHOLD,
        threshold=DEFAULT_THRESHOLD,
        model=state.metadata.get("model_file", "fl_fedavg.pt"),
    )


# =============================================================================
# CLI
# =============================================================================
def main() -> None:
    """CLI entry point: ``python -m src.serving.s22_api``."""
    import argparse

    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the prediction API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
