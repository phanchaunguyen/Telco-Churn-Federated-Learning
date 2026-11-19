# 12 — Architecture & Data Flow

**Owner:** All · **Tuần:** W1 (cập nhật liên tục) · **Trạng thái:** ✅
**Nguồn quyết định:** Project Decision Record (2026-11-18) + Patch v1 + Patch v2

> Tài liệu này là **bản đồ trực quan** của toàn bộ hệ thống. Mọi con số ở đây
> đã được đo trên dữ liệu thật (`notebooks/01_eda.ipynb`), không phải giả định.

---

## 0. Số liệu nền (authoritative)

| Metric | Value | Nguồn |
|---|---|---|
| Raw shape | 7,043 × 50 | `telco.csv` |
| `Joined` rows dropped | **454 (6.45%)** | EDA §3.1 |
| Cohort sau CLEAN | **6,589** | EDA §3.2 |
| Churn rate | **28.37%** | EDA §3.2 |
| Churned / Stayed | **1,869 / 4,720** | EDA §3.2 |
| Features | **29** | `src/data/clean.py` |
| `clean_7043.parquet` | **6,589 × 31** | D0.6 |
| `eda_7043.parquet` | **6,589 × 50** | D0.5 |
| E0 — IBM Churn Score AUC | **0.9421** | EDA §3.5 |
| `test_real` | **1,318 rows / 374 churned** | 20% stratified split |
| `train_real` | **5,271 rows** | 80% stratified split |

---

## 1. Kiến trúc tổng thể (4 layers)

```mermaid
flowchart TB
    subgraph L1["LAYER 1 · DATA PIPELINE — Apache Spark (batch)"]
        direction TB
        RAW["telco.csv<br/>7,043 × 50"]
        CLEAN["1a. CLEAN<br/>drop Joined + leakage Loại 1-6"]
        SPLIT1["1b. SPLIT<br/>stratified 80/20"]
        AMP["1c. AMPLIFY<br/>Gaussian Copula → 1M"]
        SPLIT2["1d. SPLIT<br/>90/10"]
        PART["1e. PARTITION<br/>IID / α=0.5 / α=0.1"]
        RAW --> CLEAN
        CLEAN --> SPLIT1
        SPLIT1 --> AMP
        AMP --> SPLIT2
        SPLIT2 --> PART
    end

    subgraph ART["ARTIFACTS"]
        direction LR
        A1["eda_7043.parquet<br/>6,589 × 50<br/>D0.5 · EDA only"]
        A2["clean_7043.parquet<br/>6,589 × 31<br/>D0.6"]
        A3["train_real<br/>5,271 × 31"]
        A4["test_real<br/>1,318 × 31<br/>PRIMARY EVAL"]
        A5["synth_train<br/>900K"]
        A6["synth_test<br/>100K<br/>SCALE ONLY"]
        A7["10 client silos<br/>MinIO buckets"]
    end

    CLEAN --> A1
    CLEAN --> A2
    SPLIT1 --> A3
    SPLIT1 --> A4
    SPLIT2 --> A5
    SPLIT2 --> A6
    PART --> A7

    subgraph L2["LAYER 2 · FL TRAINING — Flower + gRPC (PRIMARY)"]
        direction TB
        SRV["Flower Server :8080<br/>FedAvg / FedProx / FedAvgM<br/>50 rounds · fraction_fit=0.5"]
        C1["Client 0<br/>MLP + local stats"]
        C2["Client 1"]
        C3["Client ..."]
        C4["Client 9"]
        SRV <-->|"gRPC · model weights ONLY"| C1
        SRV <-->|"gRPC"| C2
        SRV <-->|"gRPC"| C3
        SRV <-->|"gRPC"| C4
    end

    A7 --> C1
    A7 --> C2
    A7 --> C3
    A7 --> C4

    subgraph L2B["LAYER 2b · TELEMETRY BUS — Apache Kafka (AUXILIARY)"]
        direction LR
        T1["topic: fl_telemetry<br/>per-client metrics"]
        T2["topic: fl_round_summary<br/>per-round aggregates"]
        CONS["Consumer<br/>flush every 10s"]
        T1 --> CONS
        T2 --> CONS
    end

    C1 -.->|"metrics only"| T1
    C2 -.->|"metrics only"| T1
    SRV -.->|"metrics only"| T2

    subgraph L3["LAYER 3 · EVALUATION — Apache Spark (batch)"]
        direction TB
        PRED["per-client predictions<br/>on test_real"]
        AGG["Spark SQL aggregation<br/>global AUC · per-client AUC<br/>per-Churn-Category AUC"]
        PRED --> AGG
    end

    SRV -->|"global model"| PRED
    A4 --> PRED
    A1 -.->|"join Churn Category<br/>for E5.2"| AGG

    subgraph L4["LAYER 4 · SERVING & BI"]
        direction LR
        MLF["MLflow<br/>experiment tracking"]
        SUP["Superset<br/>3 static dashboards"]
        GRA["Grafana<br/>1 live dashboard"]
        API["FastAPI<br/>POST /predict"]
    end

    CONS --> MLF
    CONS --> GRA
    AGG --> SUP
    AGG --> MLF
    MLF --> API

    classDef spark fill:#e8f4fd,stroke:#1f77b4,stroke-width:2px
    classDef fl fill:#fff4e6,stroke:#ff7f0e,stroke-width:2px
    classDef kafka fill:#f0e8fd,stroke:#9467bd,stroke-width:2px
    classDef art fill:#f5f5f5,stroke:#7f7f7f,stroke-width:1px
    classDef serve fill:#e8f8e8,stroke:#2ca02c,stroke-width:2px

    class RAW,CLEAN,SPLIT1,AMP,SPLIT2,PART,PRED,AGG spark
    class SRV,C1,C2,C3,C4 fl
    class T1,T2,CONS kafka
    class A1,A2,A3,A4,A5,A6,A7 art
    class MLF,SUP,GRA,API serve
```

---

## 2. Data pipeline — CLEAN → SPLIT → AMPLIFY → SPLIT → PARTITION

Thứ tự này là **bắt buộc**. Amplify trước khi split sẽ khiến Copula học cả phân phối
của test set → đánh giá trở thành circular, không đo được generalization.

```mermaid
flowchart LR
    RAW["telco.csv<br/>7,043 × 50"]

    subgraph S1["STEP 1 · CLEAN"]
        C1["drop Joined rows<br/>454 rows"]
        C2["drop leakage Loại 1-3<br/>9 cols"]
        C3["drop constants Loại 4<br/>3 cols"]
        C4["drop derived dup Loại 6<br/>3 cols"]
        C5["drop geo Loại 5<br/>5 cols"]
        C1 --> C2 --> C3 --> C4 --> C5
    end

    RAW --> S1

    S1 --> EDA["eda_7043.parquet<br/>6,589 × 50<br/>D0.5 · EDA only"]
    S1 --> CLN["clean_7043.parquet<br/>6,589 × 31<br/>ID + 29 features + churn"]

    subgraph S2["STEP 2 · SPLIT (stratified 80/20)"]
        TR["train_real<br/>5,271 rows"]
        TE["test_real<br/>1,318 rows<br/>374 churned"]
    end

    CLN --> S2

    subgraph S3["STEP 3 · AMPLIFY (train_real ONLY)"]
        COP["Gaussian Copula<br/>primary"]
        CTG["CTGAN<br/>validation"]
        GATE["Quality gate<br/>KS · corr · TSTR gap < 5%"]
        COP --> GATE
        CTG --> GATE
    end

    TR --> S3
    TE -.->|"NEVER amplified"| TE

    S3 --> SYN["synthetic_1M.parquet<br/>1,000,000 × 30"]

    subgraph S4["STEP 4 · SPLIT (90/10)"]
        ST["synth_train<br/>900,000"]
        SV["synth_test<br/>100,000<br/>scale only"]
    end

    SYN --> S4

    subgraph S5["STEP 5 · PARTITION"]
        IID["IID<br/>equal split"]
        A05["Label skew α=0.5"]
        A01["Label skew α=0.1"]
    end

    ST --> S5

    S5 --> SILO["10 client silos<br/>s3://clients/client_i/"]

    subgraph S6["STEP 6 · VERIFY"]
        V1["assert: no leakage cols"]
        V2["assert: disjoint by Customer ID"]
        V3["assert: 29 features per silo"]
        V1 --> V2 --> V3
    end

    SILO --> S6

    classDef clean fill:#e8f4fd,stroke:#1f77b4
    classDef split fill:#fff4e6,stroke:#ff7f0e
    classDef amp fill:#f0e8fd,stroke:#9467bd
    classDef part fill:#e8f8e8,stroke:#2ca02c
    classDef verify fill:#ffe8e8,stroke:#d62728

    class C1,C2,C3,C4,C5 clean
    class TR,TE,ST,SV split
    class COP,CTG,GATE amp
    class IID,A05,A01,SILO part
    class V1,V2,V3 verify
```

### 2.1 Bảng artifact

| Artifact | Shape | Vai trò | Được amplify? | Được partition? |
|---|---|---|---|---|
| `eda_7043.parquet` (D0.5) | 6,589 × 50 | EDA, E0, E5.2 strata | ❌ | ❌ |
| `clean_7043.parquet` (D0.6) | 6,589 × 31 | Nguồn cho split | ❌ | ❌ |
| `train_real` | 5,271 × 31 | Nguồn amplify | ✅ | ❌ |
| `test_real` | 1,318 × 31 | **Primary eval** | ❌ | ❌ |
| `synthetic_1M` (D1) | 1,000,000 × 30 | Nguồn partition | — | ❌ |
| `synth_train` | 900,000 × 30 | Nguồn partition | — | ✅ |
| `synth_test` | 100,000 × 30 | Scale eval only | — | ❌ |
| 10 client silos | ~90,000 × 30 mỗi silo | FL training | — | — |

---

## 3. FL round — sequence diagram

```mermaid
sequenceDiagram
    autonumber
    participant S as Flower Server<br/>:8080
    participant K as Kafka<br/>fl_round_summary
    participant C0 as Client 0
    participant C1 as Client 1
    participant C9 as Client 9
    participant KT as Kafka<br/>fl_telemetry
    participant M as MLflow

    Note over S: round t = 1..50
    S->>S: sample 5/10 clients<br/>fraction_fit = 0.5

    S->>C0: gRPC · broadcast W_global
    S->>C1: gRPC · broadcast W_global
    S->>C9: gRPC · broadcast W_global

    par Local training (parallel via Ray)
        C0->>C0: train MLP on local silo<br/>local stats only
        C0-->>KT: metrics {round, client_id,<br/>local_auc, local_loss,<br/>n_samples, bytes_sent, duration}
    and
        C1->>C1: train MLP on local silo
        C1-->>KT: metrics
    and
        C9->>C9: train MLP on local silo
        C9-->>KT: metrics
    end

    C0->>S: gRPC · return ΔW_0 + n_0
    C1->>S: gRPC · return ΔW_1 + n_1
    C9->>S: gRPC · return ΔW_9 + n_9

    S->>S: aggregate<br/>W_global = Σ (n_i/n) · W_i
    S-->>K: {round, global_auc, global_loss,<br/>n_clients_selected, total_bytes,<br/>round_duration_sec}

    KT-->>M: consumer flush every 10s
    K-->>M: consumer flush every 10s

    Note over S: next round
```

---

## 4. Experiment protocol — tiered evaluation

Quyết định: **E1 dùng 5-fold CV**, **E2–E7 dùng fold 0 + bootstrap CI**.
Lý do: 5-fold cho toàn bộ sẽ cần ~45 FL runs (2–4 giờ) và làm strata của E5.2
co lại còn ~75 churned/stratum.

```mermaid
flowchart TB
    START["clean_7043.parquet<br/>6,589 × 31"]

    START --> F0["Fold 0<br/>PRIMARY"]
    START --> F14["Folds 1–4<br/>E1 robustness only"]

    subgraph P0["Fold 0 pipeline"]
        direction TB
        T0["train_real 5,271"]
        E0["test_real 1,318"]
        AMP0["amplify → 1M"]
        P0A["partition → 10 silos"]
        T0 --> AMP0 --> P0A
    end

    F0 --> P0

    subgraph EXP["Experiments"]
        direction TB
        X0["E0 · IBM Churn Score<br/>external ceiling"]
        X1["E1 · FL gap<br/>centralized vs FL vs local"]
        X2["E2 · partition impact<br/>IID vs α=0.5 vs α=0.1"]
        X3["E3 · aggregation<br/>FedAvg vs FedProx vs FedAvgM"]
        X4["E4 · privacy-utility<br/>noise ∈ {0, 0.5, 1, 2}"]
        X5["E5.1 · per-client AUC"]
        X6["E5.2 · stratified by Churn Category"]
        X7["E6 · scalability K ∈ {5,10,20,50}"]
        X8["E7 · communication<br/>fraction ∈ {0.25, 0.5, 1.0}"]
    end

    P0 --> EXP

    E0 --> R0["test_real<br/>bootstrap CI"]
    X0 --> R0
    X1 --> R1["5-fold CV<br/>mean ± std + CI"]
    X2 --> R2["fold 0<br/>bootstrap CI"]
    X3 --> R2
    X4 --> R2
    X5 --> R2
    X6 --> R3["fold 0 + CI<br/>caveat: n≈75/stratum"]
    X7 --> R4["synth_test<br/>latency only"]
    X8 --> R5["test_real primary<br/>synth_test sanity"]

    F14 --> R1

    classDef primary fill:#e8f8e8,stroke:#2ca02c,stroke-width:2px
    classDef secondary fill:#fff4e6,stroke:#ff7f0e
    classDef scale fill:#f0e8fd,stroke:#9467bd
    classDef caveat fill:#ffe8e8,stroke:#d62728
    
    class R0,R1 primary
    class R2 secondary
    class R3 caveat
    class R4 scale
    class R5 secondary
```

### 4.1 Bảng protocol

| Experiment | Test set | Protocol | Reporting |
|---|---|---|---|
| E0 | `test_real` | 1 lần | AUC + bootstrap 95% CI |
| **E1** | `test_real` | **5-fold CV** | **mean ± std + bootstrap CI** |
| E2 | `test_real` | fold 0 | AUC + bootstrap CI |
| E3 | `test_real` | fold 0 | AUC + per-client std + CI |
| E4 | `test_real` | fold 0 | AUC vs ε curve + CI |
| E5.1 | `test_real` | fold 0 | per-client AUC distribution + CI |
| E5.2 | `test_real` + `eda_7043` | fold 0 | per-stratum AUC + CI + caveat |
| E6 | `synth_test` | fold 0 | latency, AUC consistency |
| E7 | `test_real` (primary) + `synth_test` (sanity) | fold 0 | bytes, AUC |

### 4.2 Diagnostic bắt buộc

Nếu `|AUC_synth_test − AUC_test_real| > 10%` → **synthetic data quality problem**.
Báo cáo như một finding, không được ẩn.

---

## 5. Non-overlap principle

Ba công cụ **không bao giờ** chồng chéo trách nhiệm.

```mermaid
flowchart LR
    subgraph G["gRPC via Flower"]
        G1["PRIMARY"]
        G2["model weights transport"]
        G3["NEVER: metrics, data processing"]
        G1 --- G2 --- G3
    end

    subgraph K["Apache Kafka"]
        K1["AUXILIARY"]
        K2["telemetry / observability"]
        K3["NEVER: model weights, training data"]
        K1 --- K2 --- K3
    end

    subgraph S["Apache Spark"]
        S1["BATCH"]
        S2["data pipeline + eval aggregation"]
        S3["NEVER: model training, FL loop"]
        S1 --- S2 --- S3
    end

    classDef grpc fill:#fff4e6,stroke:#ff7f0e,stroke-width:2px
    classDef kafka fill:#f0e8fd,stroke:#9467bd,stroke-width:2px
    classDef spark fill:#e8f4fd,stroke:#1f77b4,stroke-width:2px

    class G1,G2,G3 grpc
    class K1,K2,K3 kafka
    class S1,S2,S3 spark
```

| Tool | Role | Never does |
|---|---|---|
| **gRPC (Flower)** | Model weights transport | Metrics, data processing |
| **Kafka** | Observability / telemetry bus | Model weights, training data |
| **Spark** | Batch data processing + eval aggregation | Model training, FL loop |

---

## 6. Docker Compose services

```mermaid
flowchart TB
    subgraph CORE["Core (default profile)"]
        PG["postgres :5432<br/>MLflow backend"]
        MN["minio :9000 / :9001<br/>S3 storage"]
        MI["minio-init<br/>tạo 10 buckets"]
        ML["mlflow :5000"]
        KF["kafka :9092<br/>KRaft mode"]
        SM["spark-master :8080 / :7077"]
        SW["spark-worker"]
    end

    subgraph FULL["Optional (--profile full)"]
        GF["grafana :3000"]
    end

    subgraph W9["W9 — chưa thêm"]
        SS["superset :8088"]
        FS["flower-server :8080"]
        AF["airflow :8081"]
    end

    MI --> MN
    ML --> PG
    ML --> MN
    SW --> SM

    classDef have fill:#e8f8e8,stroke:#2ca02c,stroke-width:2px
    classDef opt fill:#fff4e6,stroke:#ff7f0e
    classDef todo fill:#f5f5f5,stroke:#7f7f7f,stroke-dasharray: 5 5

    class PG,MN,MI,ML,KF,SM,SW have
    class GF opt
    class SS,FS,AF todo
```

**Resource:** default ~8 GB RAM · `--profile full` ~12 GB RAM.

---

## 7. Đọc tiếp

| Cần hiểu | Đọc |
|---|---|
| Quyết định dữ liệu (leakage, 29 features) | `04_data_decisions.md` |
| Chi tiết Spark pipeline | `05_spark_pipeline.md` |
| Chi tiết Kafka telemetry | `08_kafka_telemetry.md` |
| Scope, hypothesis, RACI | `01_kickoff_notes.md` |
