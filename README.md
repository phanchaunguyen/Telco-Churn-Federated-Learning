# Federated Learning for Telecom Customer Churn Prediction

A systematic study of **non-IID robustness**, **aggregation strategies**, and the
**privacy-utility tradeoff** for churn prediction across isolated telecom data silos.

> **Core premise:** customer data never leaves its silo. Only model weights are transmitted.

---

## 1. Problem

Telecom operators hold customer data fragmented across branches and cannot centralize it
due to privacy regulations (GDPR, PDPA) and competitive concerns. This project trains an
accurate churn model **without centralizing data**, under heterogeneous (non-IID) client
distributions.

## 2. Dataset

**IBM Telco Customer Churn — EXTENDED** (`data/raw/telco.csv`)

| Property | Value |
|---|---|
| Shape | **7,043 rows × 50 columns** |
| `Total Charges` dtype | **float64** (no casting needed) |
| Churn rate | ≈ 26.5% (after dropping `Joined` rows) |
| Context | US / California / Q3 — single-quarter snapshot |
| `Customer Status` | Churned / Stayed / **Joined** |

### Pipeline order — **CLEAN → AMPLIFY → PARTITION**

```
telco.csv (7043 × 50)
   │
   ├─ [Spark] CLEAN ──► eda_7043.parquet   (D0.5, 50 cols, EDA only)
   │                └─► clean_7043.parquet (D0.6, 31 cols)
   │
   ├─ [Spark] AMPLIFY ─► synthetic_1M.parquet (29 features + target)
   │
   ├─ [Spark] PARTITION ─► 10 MinIO client silos
   │
   ├─ [Flower/gRPC] FL TRAINING ─► model weights (PRIMARY channel)
   │        └─ [Kafka] telemetry ─► MLflow + live dashboard
   │
   └─ [Spark] EVAL AGGREGATION ─► results/E*.parquet
```

### Leakage taxonomy (dropped before amplification)

| Loại | Name | Columns |
|---|---|---|
| 1 | TARGET LEAKAGE | `Churn Label`, `Customer Status` |
| 2 | POST-HOC DERIVED | `Churn Score`, `CLTV`, `Satisfaction Score` |
| 3 | INDIRECT POST-HOC | `Total Revenue`, `Total Refunds`, `Churn Category`, `Churn Reason` |
| 4 | ZERO-VARIANCE | `Country`, `State`, `Quarter` |
| 5 | GEO HIGH-CARDINALITY | `City`, `Zip Code`, `Latitude`, `Longitude`, `Population` |
| 6 | DERIVED DUPLICATES | `Under 30`, `Referred a Friend`, `Dependents` (binary) |

`Churn Score` is **not** a feature — it is the external baseline for **E0**.

## 3. Tech stack

| Layer | Component | Technology |
|---|---|---|
| **Data** | Data processing | **Apache Spark 3.5+** (`local[*]`) |
| **Data** | Storage | MinIO (S3-compatible) + Parquet |
| **Data** | Synthetic data | `copulas` (Gaussian Copula), `sdv` (CTGAN validation) |
| **Data** | Local preprocessing | Pandas, scikit-learn (per client) |
| **FL** | Communication | **gRPC via Flower** — PRIMARY, weights only |
| **FL** | Framework | **Flower (`flwr`)** |
| **FL** | DL framework | **PyTorch** (not TensorFlow) |
| **FL** | Distributed runtime | **Ray** |
| **FL** | Privacy | **Opacus** (DP-SGD) |
| **FL Telemetry** | Message bus | **Apache Kafka 3.x** (KRaft) — AUXILIARY, metrics only |
| **Tracking** | Experiment tracker | **MLflow** |
| **Serving** | API | **FastAPI** |
| **BI** | Dashboards | **Superset** (batch) + **Grafana** (live, optional) |
| **Orchestration** | Scheduler | **Apache Airflow** |
| **Containerization** | Runtime | **Docker Compose** |
| **Version control** | VCS | Git + GitHub |

> **Non-overlap rule:** gRPC = weights transport · Kafka = observability · Spark = batch data.
> These three never overlap. Kafka never carries model weights; Spark never trains models.

## 4. Repository layout

```
src/
  data/          # clean + amplification
  partitioning/  # IID + label skew
  features/      # preprocessing
  models/        # MLP
  fl/            # Flower server/client/strategies
  baselines/     # centralized + local-only
  tracking/      # MLflow
  serving/       # FastAPI
notebooks/       # EDA + analyses
results/         # experiment outputs
dags/            # Airflow
dashboards/      # Superset exports
docs/            # numbered docs (00_index, 01_kickoff, ...)
configs/         # YAML configs
scripts/         # entrypoint shell scripts
tests/           # pytest
```

## 5. Quick start

```bash
# 1. Environment
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Services (Spark + Kafka + MinIO + MLflow + Postgres)
docker compose up -d

# 2b. Full stack incl. Grafana (optional, needs ~12 GB RAM)
docker compose --profile full up -d

# 3. Verify
python scripts/verify_env.py

# 4. Clean the dataset (produces D0.5 + D0.6)
bash scripts/run_clean.sh

# 5. Tests
pytest -v
```

## 6. Documentation

All docs live in `docs/` and are **numbered by reading order**. Start at
[`docs/00_index.md`](docs/00_index.md).

## 7. Experiments

| # | Name | Independent variable |
|---|---|---|
| E0 | External baseline | IBM `Churn Score` vs actual label |
| E1 | FL gap | Centralized vs FL vs Local (IID) |
| E2 | Partition impact | IID vs α=0.5 vs α=0.1 |
| E3 | Aggregation robustness | FedAvg vs FedProx vs FedAvgM |
| E4 | Privacy-utility | noise_multiplier ∈ {0, 0.5, 1, 2} |
| E5.1 | Fairness — per client | FedAvg vs FedProx |
| E5.2 | Fairness — stratified | by `Churn Category` |
| E6 | Scalability | K ∈ {5, 10, 20, 50} |
| E7 | Communication | fraction_sampled ∈ {0.25, 0.5, 1.0} |

## 8. Team

| Person | Role |
|---|---|
| A | Data Lead |
| B | FL Infra Lead |
| C | ML Lead |
| D | MLOps & BI Lead |
