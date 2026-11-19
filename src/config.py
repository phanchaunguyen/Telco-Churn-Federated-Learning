"""Central configuration loader.

Single source of truth for all paths and hyper-parameters. Values come from
``configs/base.yaml`` and can be overridden by environment variables prefixed
with ``FLTELCO_`` (nested keys use ``__`` as separator, e.g.
``FLTELCO_FL__N_ROUNDS=20``).

Usage
-----
>>> from src.config import get_config
>>> cfg = get_config()
>>> cfg.paths.raw_data
PosixPath('data/raw/telco.csv')
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# Repository root = parent of the ``src`` package
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH: Path = REPO_ROOT / "configs" / "base.yaml"


# ---------------------------------------------------------------------------
# Nested config models
# ---------------------------------------------------------------------------
class ProjectCfg(BaseModel):
    name: str = "fl-telco-churn"
    seed: int = 42


class PathsCfg(BaseModel):
    raw_data: Path
    processed_dir: Path
    synthetic_dir: Path
    results_dir: Path
    eda_parquet: Path
    clean_parquet: Path
    train_real_parquet: Path
    test_real_parquet: Path
    synthetic_parquet: Path
    synth_train_parquet: Path
    synth_test_parquet: Path
    ctgan_parquet: Path
    quality_report: Path
    partition_metadata: Path
    silo_verification: Path

    def resolve_all(self) -> "PathsCfg":
        """Return a copy with every path made absolute against ``REPO_ROOT``."""
        data = {
            name: (p if p.is_absolute() else REPO_ROOT / p)
            for name, p in self.model_dump().items()
        }
        return PathsCfg(**data)


class SplitCfg(BaseModel):
    """Stratified split configuration.

    ``test_real`` is the PRIMARY evaluation set and must never be amplified
    or partitioned.
    """

    test_size: float = 0.2
    seed: int = 42
    stratify_on: str = "churn"
    n_folds: int = 5
    synthetic_test_size: float = 0.1


class QualityGateCfg(BaseModel):
    ks_alpha: float = 0.05
    max_auc_gap: float = 0.05


class AmplificationCfg(BaseModel):
    method: Literal["copula", "ctgan"] = "copula"
    n_rows: int = 1_000_000
    quality_gate: QualityGateCfg = Field(default_factory=QualityGateCfg)


class DataCfg(BaseModel):
    target_col: str = "churn"
    id_col: str = "Customer ID"
    drop_status: str = "Joined"
    expected_clean_columns: int = 31
    expected_feature_count: int = 29


class PartitionStrategyCfg(BaseModel):
    name: Literal["iid", "label_skew"]
    alpha: float | None = None


class PartitioningCfg(BaseModel):
    n_clients: int = 10
    strategies: list[PartitionStrategyCfg] = Field(default_factory=list)
    scalability_k: list[int] = Field(default_factory=lambda: [5, 10, 20, 50])


class ModelCfg(BaseModel):
    name: str = "mlp"
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.3
    batch_norm: bool = True


class FedProxCfg(BaseModel):
    proximal_mu: float = 0.01


class FedAvgMCfg(BaseModel):
    server_momentum: float = 0.9


class StrategiesCfg(BaseModel):
    fedavg: dict[str, Any] = Field(default_factory=dict)
    fedprox: FedProxCfg = Field(default_factory=FedProxCfg)
    fedavgm: FedAvgMCfg = Field(default_factory=FedAvgMCfg)


class FLCfg(BaseModel):
    n_rounds: int = 50
    fraction_fit: float = 0.5
    fraction_evaluate: float = 0.5
    local_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-3
    strategies: StrategiesCfg = Field(default_factory=StrategiesCfg)


class PrivacyCfg(BaseModel):
    enabled: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    sweep_noise_multiplier: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])


class MLflowCfg(BaseModel):
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "fl-telco-churn"


class MinioCfg(BaseModel):
    endpoint: str = "localhost:9000"
    console: str = "localhost:9001"
    access_key: str = "minioadmin"
    secret_key: str = "minioadmin"
    bucket_prefix: str = "clients"


class Config(BaseModel):
    """Root configuration object."""

    project: ProjectCfg = Field(default_factory=ProjectCfg)
    paths: PathsCfg
    data: DataCfg = Field(default_factory=DataCfg)
    split: SplitCfg = Field(default_factory=SplitCfg)
    amplification: AmplificationCfg = Field(default_factory=AmplificationCfg)
    partitioning: PartitioningCfg = Field(default_factory=PartitioningCfg)
    model: ModelCfg = Field(default_factory=ModelCfg)
    fl: FLCfg = Field(default_factory=FLCfg)
    privacy: PrivacyCfg = Field(default_factory=PrivacyCfg)
    mlflow: MLflowCfg = Field(default_factory=MLflowCfg)
    minio: MinioCfg = Field(default_factory=MinioCfg)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_config(path: str | Path | None = None) -> Config:
    """Load and validate the YAML config, resolving all paths to absolute."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    cfg = Config(**raw)
    cfg.paths = cfg.paths.resolve_all()
    return cfg


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached accessor for the default config."""
    return load_config()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    c = get_config()
    print(f"repo root      : {REPO_ROOT}")
    print(f"raw data       : {c.paths.raw_data}")
    print(f"clean parquet  : {c.paths.clean_parquet}")
    print(f"features       : {c.data.expected_feature_count}")
    print(f"n_clients      : {c.partitioning.n_clients}")
    print(f"n_rounds       : {c.fl.n_rounds}")
