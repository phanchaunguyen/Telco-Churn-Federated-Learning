# 📋 MASTER PROMPT — Copy & paste vào AI agent của bạn

Dưới đây là prompt duy nhất, self-contained, chứa **toàn bộ context của dự án**. Mỗi thành viên copy nguyên khối bên dưới, paste vào AI (Claude/ChatGPT/Gemini), và AI sẽ ngay lập tức hiểu dự án để hỗ trợ. Cuối prompt có slot `[MY_ROLE]` để điền vai trò của bạn.

---

````text
# SYSTEM CONTEXT: Federated Learning for Telco Customer Churn

You are assisting a 4-person university capstone team on a Big Data / Machine Learning project. You have full context below. From now on, whenever I ask anything, assume this context is active. Do NOT suggest approaches that contradict the constraints. If my question is ambiguous, ask for clarification before generating code or analysis.

## 1. PROJECT IDENTITY

**Title:** Federated Learning for Telecom Customer Churn Prediction: A Systematic Study of Non-IID Robustness, Aggregation Strategies, and Privacy-Utility Tradeoff

**Problem:** Telecom operators have customer data fragmented across branches, and cannot centralize it due to privacy regulations (GDPR, PDPA) and competition. How to train an accurate churn prediction model WITHOUT centralizing data, given that data across branches is heterogeneous (non-IID)?

**Hypotheses:**
- H1 (Main): FL achieves churn prediction performance close to centralized (< 5% AUC gap under IID; < 10% under non-IID) without sharing any customer data.
- H2: Under severe label skew (Dirichlet α ≤ 0.1), FedProx and FedAvgM outperform FedAvg on global AUC and per-client fairness.
- H3: Applying DP-SGD with ε ∈ [1, 10] reduces AUC by ≤ 5%; there exists a "sweet spot" ε where privacy is protected and performance is acceptable.
- H4: Clients with less data or extreme label distributions suffer performance penalties; FedProx reduces this disparity vs FedAvg.

**Research Questions:**
- RQ1: What % of centralized performance does FL achieve under IID vs non-IID? (H1)
- RQ2: Which aggregation strategy is most robust to label skew? (H2)
- RQ3: What is the privacy-utility tradeoff (ε vs AUC) for churn? (H3)
- RQ4: Does FL cause unfairness across clients? How to measure and mitigate? (H4)
- RQ5: Does FL scale when K (number of clients) grows from 5 to 50?
- RQ6: Is FL's communication cost feasible in practice?

## 2. HARD CONSTRAINTS — DO NOT VIOLATE

These are non-negotiable. If you suggest something that violates them, you are wrong.

### 2.1 Dataset & pipeline constraints

- ❌ Do NOT suggest using the "Temporal Churn & LTV" dataset (billing/usage/support panel). It was evaluated and rejected: it is a closed cohort (no new customers), right-censored, panel length 1–11 cycles, and non-joinable with any other dataset. Not used.
- ❌ Do NOT suggest IBM Telco original 7K rows as the sole training dataset. It is amplified to 1M+ synthetic rows first.
- ❌ Do NOT suggest a "single customer dataset joined across flows". Data isolation across client silos is a HARD requirement of FL — this is the whole point of the project.
- ❌ Do NOT use K-means, PCA, or any unsupervised step to "improve" the pipeline unless explicitly asked. It's tabular supervised classification.
- ❌ Do NOT use TensorFlow. Stack is PyTorch + Flower + Ray.
- ❌ Do NOT propose "just centralize the data" as an alternative. The entire premise is privacy-preserving FL.

### 2.1b Spark / Kafka role constraints (ADDED — Patch v2)

The project is a **Big Data** capstone. Spark and Kafka are **auxiliary layers** that support the FL core. They do NOT replace gRPC.

- ✅ **gRPC via Flower is the PRIMARY and ONLY channel for model weights.** This never changes.
- ✅ **Kafka is AUXILIARY — telemetry/metrics only.** Kafka carries `{round, client_id, auc, loss, bytes_sent, duration}` messages. Kafka NEVER carries model weights.
- ✅ **Spark is for DATA PIPELINE and EVALUATION only.** Spark cleans, amplifies, partitions, and aggregates evaluation results. Spark NEVER trains models and is NEVER inside the FL loop.
- ❌ Do NOT propose Kafka as the FL communication channel (replacing gRPC). This violates the core principle.
- ❌ Do NOT propose Spark for FL training itself.
- ❌ Do NOT add Spark/Kafka responsibilities beyond: data pipeline + evaluation aggregation + telemetry.
- ❌ Do NOT use Spark Structured Streaming or Kafka Streams for the data pipeline — the pipeline is **batch**. Kafka is a metrics bus, not a training data source.

**Non-overlap rule (memorize):**

| Tool | Role | Never does |
|---|---|---|
| **gRPC (Flower)** | Model weights transport | Metrics, data processing |
| **Kafka** | Observability / telemetry bus | Model weights, training data |
| **Spark** | Batch data processing + eval aggregation | Model training, FL loop |

### 2.2 Leakage & feature constraints (REVISED — see §3.2)

- ❌ Do NOT use `Churn Score`, `CLTV`, `Satisfaction Score`, `Total Revenue`, `Total Refunds` as features
- ❌ Do NOT use `Churn Category`, `Churn Reason` as features (EDA only)
- ❌ Do NOT use `Churn Label`, `Customer Status` as features
- ❌ Do NOT include `Joined` rows in the training cohort (drop BEFORE amplification)
- ❌ Do NOT include `Country`, `State`, `Quarter` (constants) in features
- ❌ Do NOT include `City`, `Zip Code`, `Latitude`, `Longitude`, `Population` as features (EDA only)
- ❌ Do NOT include `Under 30`, `Referred a Friend`, `Dependents` (binary) — keep original variants
- ❌ Do NOT amplify BEFORE cleaning — order is **CLEAN → AMPLIFY → PARTITION**
- ❌ Do NOT cast `Total Charges` (already float64)
- ❌ Do NOT hardcode MLP input dimension (measure at runtime)
- ❌ Do NOT claim results without measuring (say "must be measured" instead)

## 3. DATA

### 3.1 Source — IBM Telco Customer Churn **EXTENDED**

- **7,043 rows × 50 columns** (NOT 21 columns)
- `Total Charges` is **float64** (NOT VARCHAR with blanks) → no casting needed
- Churn rate ≈ 26.5% (after dropping `Joined` rows)
- Context: entire dataset is US / California / Q3 — a **single-quarter snapshot**

**Exact column list (50 columns, verbatim):**

```
Customer ID, Gender, Age, Under 30, Senior Citizen, Married, Dependents,
Number of Dependents, Country, State, City, Zip Code, Latitude, Longitude,
Population, Quarter, Referred a Friend, Number of Referrals, Tenure in Months,
Offer, Phone Service, Avg Monthly Long Distance Charges, Multiple Lines,
Internet Service, Internet Type, Avg Monthly GB Download, Online Security,
Online Backup, Device Protection Plan, Premium Tech Support, Streaming TV,
Streaming Movies, Streaming Music, Unlimited Data, Contract, Paperless Billing,
Payment Method, Monthly Charge, Total Charges, Total Refunds,
Total Extra Data Charges, Total Long Distance Charges, Total Revenue,
Satisfaction Score, Customer Status, Churn Label, Churn Score, CLTV,
Churn Category, Churn Reason
```

**Schema corrections (do not assume the classic 21-column IBM layout):**

- ❌ There is NO `Churn Value` column (1/0) — only `Churn Label` (Yes/No)
- ❌ There is NO `Churn Score Category` column
- ❌ There is NO `Satisfaction Score Label` column
- ❌ There is NO `Count` column
- ✅ `Customer Status` has 3 values: **Churned / Stayed / Joined**
- ✅ `Total Charges` is already float — no casting needed
- ✅ `Offer` and `Internet Type` use the literal string `"None"` — these are NOT NaN
- ✅ There is no `"No phone service"` / `"No internet service"` sentinel (unlike the classic 21-col version)

### 3.2 Leakage taxonomy (must drop before amplification)

| Loại | Name | Columns |
|---|---|---|
| **1** | TARGET LEAKAGE (direct) | `Churn Label`, `Customer Status` |
| **2** | POST-HOC DERIVED | `Churn Score`, `CLTV`, `Satisfaction Score` |
| **3** | INDIRECT POST-HOC | `Total Revenue`, `Total Refunds`, `Churn Category`, `Churn Reason` |
| **4** | ZERO-VARIANCE / CONSTANTS | `Country`, `State`, `Quarter` |
| **5** | HIGH-CARDINALITY GEOGRAPHIC | `City`, `Zip Code`, `Latitude`, `Longitude`, `Population` |
| **6** | DERIVED DUPLICATES | `Under 30`, `Referred a Friend`, `Dependents` (binary) |

**Special handling:**
- `Churn Category` + `Churn Reason`: drop from features, **preserve in `eda_7043.parquet`** for Chapter 1 (motivation) and Chapter 6 (E5.2 stratification)
- `CLTV` + `Satisfaction Score`: drop from features, **preserve for EDA**
- `Churn Score`: drop from features, **use as EXTERNAL BASELINE for E0** (see §5)

### 3.3 Final feature list — **29 raw features**

| Group | N | Columns |
|---|---|---|
| Demographics | 5 | Gender, Age, Senior Citizen, Married, Number of Dependents |
| Account | 6 | Tenure in Months, Offer, Contract, Paperless Billing, Payment Method, **Number of Referrals** |
| Services | 12 | Phone Service, Multiple Lines, Internet Service, Internet Type, Online Security, Online Backup, Device Protection Plan, Premium Tech Support, Streaming TV, Streaming Movies, Streaming Music, Unlimited Data |
| Usage & Billing | 6 | Avg Monthly Long Distance Charges, Avg Monthly GB Download, Monthly Charge, Total Charges, Total Extra Data Charges, Total Long Distance Charges |

> **Note:** `Number of Referrals` is kept as feature #29 (Loại 6 says "keep the count, drop the binary"). After one-hot encoding of categoricals, expect ~42–48 numeric features for the MLP input.

**Column accounting (must sum to 50):**
`29 features + Customer ID(1) + Loại6(3) + Loại4(3) + Loại5(5) + Loại1(2) + Loại2(3) + Loại3(4) = 50`

### 3.4 Pipeline order — **CLEAN → AMPLIFY → PARTITION**

Amplifying before cleaning pollutes CTGAN training, conditions synthetic data on leakage, and is unstable on zero-variance columns. Client silos must NEVER contain leakage columns.

```
Step 1 — CLEAN (on original telco.csv, 7043 × 50)
  a. Drop Customer Status = "Joined" rows
  b. Drop all leakage columns (Loại 1, 2, 3)
  c. Drop constants (Loại 4)
  d. Drop derived duplicates (Loại 6)
  e. Drop high-cardinality geographic from FEATURES only (Loại 5)
  f. Output A: clean_7043.parquet (Customer ID + 29 features + 1 binary target = 31 cols)
  g. Output B: eda_7043.parquet (original 50 cols minus Joined rows — for EDA only)

Step 2 — AMPLIFY (on clean_7043.parquet)
  a. Gaussian Copula → synthetic_1M.parquet (29 features + target only)
  b. CTGAN → validation comparison
  c. Quality gate: KS test per column, correlation matrix diff,
     train-on-synth-test-on-real AUC gap < 5%

Step 3 — PARTITION (on synthetic_1M.parquet)
  a. IID: shuffle + equal split K=10
  b. Label Skew α=0.5
  c. Label Skew α=0.1
  d. Write each partition to its own MinIO bucket (client silo)

Step 4 — VERIFY (before any FL training)
  a. Assertion: no silo contains {Churn Score, CLTV, Satisfaction Score, Total Revenue,
     Total Refunds, Churn Category, Churn Reason, Churn Label, Customer Status,
     Country, State, Quarter}
  b. Assertion: all silos are disjoint by Customer ID
  c. Assertion: feature count == 29 in every silo
```

### 3.5 Amplification (produces 1M+ synthetic rows)

- Primary: Gaussian Copula (fast, controls marginals + correlation)
- Validation: CTGAN (compare quality via KS test, correlation matrix, train-on-synthetic-test-on-real AUC gap)
- Choose one as primary based on quality metrics; document the decision

### 3.6 Partitioning into K = 10 clients (two strategies only)

1. **IID baseline**: random shuffle, equal split. Churn ratio ≈ 26.5% per client.
2. **Label Skew**: Dirichlet distribution over labels. Two α values:
   - α = 0.5 (moderate non-IID)
   - α = 0.1 (severe non-IID)

**K benchmarked at {5, 10, 20, 50} for scalability experiment (E6).**

## 4. ARCHITECTURE (5 layers)

```
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — DATA PIPELINE (Apache Spark)                              │
│  1a. Clean: telco.csv (7K×50) → Spark SQL drop leakage/constants    │
│              → clean_7043.parquet (6,589 × 31)                      │
│  1b. Amplify: Spark parallel Gaussian Copula → synthetic_1M.parquet │
│  1c. Partition: Spark repartition → 10 client silos on MinIO        │
│              (IID, α=0.5, α=0.1)                                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 2 — FL TRAINING (Flower + gRPC, UNCHANGED)                    │
│  Client ↔ Server: model weights via gRPC (PRIMARY, standard FL)     │
│  Rounds: 50, fraction_fit=0.5, strategies: FedAvg/FedProx/FedAvgM   │
│                                                                     │
│  PARALLEL — TELEMETRY BUS (Apache Kafka, NEW):                      │
│  Client → topic "fl_telemetry"        (per-client metrics)          │
│  Server → topic "fl_round_summary"    (per-round aggregates)        │
│  Consumer → MLflow (batch) + Superset/Grafana (real-time)           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 3 — EVALUATION (Apache Spark)                                 │
│  Load 1M test set → Spark SQL aggregate per-client predictions      │
│  → global AUC, per-client AUC, fairness strata, churn-category AUC  │
│  → results/E*.parquet + results/E*.csv                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 4 — SERVING & BI (Superset + FastAPI + MLflow, UNCHANGED)     │
└─────────────────────────────────────────────────────────────────────┘
```

**Key clarifications:**
- Layer 2 is **unchanged** — Flower/gRPC is still the FL engine.
- Kafka is **inside** Layer 2 as a parallel bus, not on the critical path.
- Spark appears at Layer 1 (prep) and Layer 3 (post-eval), never inside FL.

**Critical architectural principles:**
- Data never leaves client silos — only model weights/gradients are transmitted.
- Local preprocessing uses ONLY local statistics.
- Evaluation compares against centralized and local-only baselines.
- All experiments logged to MLflow.
- **gRPC = weights transport · Kafka = observability · Spark = batch data. These three never overlap.**

## 5. 8 EXPERIMENTS

| # | Name | Independent Variable | Metrics | Answers |
|---|---|---|---|---|
| **E0** | **External baseline** | IBM `Churn Score` vs actual `Churn Label` | AUC | Reference ceiling |
| E1 | FL gap | Centralized vs FL vs Local (IID) | AUC, F1, Recall@10% | RQ1, H1 |
| E2 | Partition impact | IID vs α=0.5 vs α=0.1 | AUC, convergence rounds | RQ1, H1 |
| E3 | Aggregation robustness | FedAvg vs FedProx vs FedAvgM (α=0.1) | AUC, per-client std | RQ2, H2 |
| E4 | Privacy-utility | noise_multiplier ∈ {0, 0.5, 1, 2} | AUC vs ε curve | RQ3, H3 |
| E5 | Fairness | FedAvg vs FedProx | min AUC, std AUC | RQ4, H4 |
| E5.1 | Fairness — per client | FedAvg vs FedProx | per-client AUC distribution | RQ4, H4 |
| E5.2 | Fairness — stratified | FedAvg vs FedProx, stratified by `Churn Category` | per-stratum AUC | RQ4, H4 |
| E6 | Scalability | K ∈ {5, 10, 20, 50} | Time, AUC | RQ5 |
| E7 | Communication | fraction_sampled ∈ {0.25, 0.5, 1.0} | Bytes, AUC | RQ6 |

### E0 — External baseline (no FL required)

The IBM `Churn Score` column is a precomputed prediction using all post-hoc data (Satisfaction, CLTV, Churn Category, etc.). Evaluate its AUC against the actual `Churn Label`.

Purpose: establish an "upper reference" that uses information our models legally cannot access. Report:

> "IBM's precomputed Churn Score (which leverages post-hoc features) achieves AUC = X. Our centralized model without post-hoc features achieves AUC = Y. Our FL model achieves AUC = Z. Despite lacking post-hoc information, FL retains (Z/Y)×100% of centralized performance and (Z/X)×100% of IBM's score."

E0 is a simple comparison table + one paragraph in the report.

### E5.2 — Fairness stratified by churn reason

Use `Churn Category` from `eda_7043.parquet` to label test-set customers. For each client, compute AUC separately per churn reason (Price / Competitor / Dissatisfaction / Attitude / Other). Plot as grouped boxplot.

Question: Does FedProx mitigate unfairness across churn reasons better than FedAvg? This is a research contribution — analyzing whether FL aggregation strategy affects fairness **across reason strata**, not just across clients.

## 6. TECH STACK

| Layer | Component | Technology | Role |
|---|---|---|---|
| **Data** | Data processing | **Apache Spark 3.5+** (`local[*]`) | Clean, amplify, partition 1M rows |
| **Data** | Storage | MinIO + Parquet + Delta Lake | Client silos |
| **Data** | Synthetic data | Python, NumPy, `sdv` (CTGAN), `copulas` | Amplification (fit on pandas sample) |
| **Data** | Local preprocessing | Pandas, scikit-learn (per client) | Per-client feature pipeline |
| **FL** | Communication | **gRPC via Flower** | **PRIMARY — model weights only** |
| **FL** | Framework | Flower (`flwr`) | Server + client orchestration |
| **FL** | DL framework | **PyTorch** (NOT TensorFlow) | Local MLP |
| **FL** | Distributed runtime | **Ray** | Parallel client training (simulation) |
| **FL** | Privacy | **Opacus** (DP-SGD) | Differential privacy |
| **FL Telemetry** | Message bus | **Apache Kafka 3.x** (KRaft) | **AUXILIARY — metrics only** |
| **Tracking** | Experiment tracker | **MLflow** | From both gRPC returns and Kafka |
| **Serving** | API | **FastAPI** | Prediction endpoint |
| **BI** | Dashboards | **Superset** (batch) + **Grafana** (live) | 3 static + 1 live dashboard |
| **Orchestration** | Scheduler | **Apache Airflow** | DAGs: data prep → FL → eval → dashboard refresh |
| **Containerization** | Runtime | **Docker Compose** | Spark + Kafka + MinIO + MLflow + Superset + Flower |
| **Version control** | VCS | Git + GitHub | — |

**Local MLP architecture (input dimension is RUNTIME-DETERMINED — do NOT hardcode):**

```
Input (D_features, read from X_train.shape[1], likely 42–48 after one-hot)
  → Linear(D, 128) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(128, 64) → BatchNorm → ReLU → Dropout(0.3)
  → Linear(64, 32) → ReLU
  → Linear(32, 1) → Sigmoid
```

## 7. TEAM STRUCTURE (4 people)

| Person | Role | Owns |
|---|---|---|
| A | Data Lead | Amplification, partitioning, client silos, E1/E2 baselines |
| B | FL Infra Lead | Flower, aggregation strategies, Ray, Airflow, E3/E6/E7 |
| C | ML Lead | Local models, feature engineering, DP, E4/E5 |
| D | MLOps & BI Lead | MLflow, Superset, FastAPI, Docker, documentation, integration |

## 8. TIMELINE (11 weeks)

- W1: Setup + EDA + **CLEAN (D0.5, D0.6)** + reading
- W2: **Spark** clean + **Spark** amplification (1M rows) + quality gate
- W3: **Spark** partitioning → 10 MinIO silos + **Kafka** topic setup
- W4: FL core (gRPC) + **Kafka telemetry hookup** (producer/consumer)
- W5: Full 50 rounds IID + E1 (FL gap) + **Spark eval aggregation**
- W6: Non-IID runs + FedProx + FedAvgM + E2 + E3
- W7: DP-SGD + E4 + E5.1 + E5.2 (fairness)
- W8: Scalability + communication + E6 + E7
- W9: Serving (FastAPI, Superset 3 static + **1 live**, Airflow + `flush_kafka_to_mlflow`)
- W10: Integration + report draft
- W11: Slides + demo video + final report

**Critical path:** A (data) → B (FL infra) → experiments → D (serving) → report.
**Parallel track:** C (ML) runs alongside B after W4.

**Writing policy:** Report sections are written IN PARALLEL from W6 onwards (each experiment done → corresponding section drafted immediately). Do NOT defer writing to W10.

## 9. SUCCESS CRITERIA

Project passes if **≥ 4 mandatory + ≥ 3 "should"** criteria met (11 items total).

### MANDATORY (4)

- ✅ FL runs end-to-end 50 rounds IID without crash
- ✅ E1: FL gap < 5% AUC vs centralized (IID)
- ✅ E2: non-IID degrades AUC (measured)
- ✅ E3: FedProx/FedAvgM outperform FedAvg (measured)

### SHOULD (5)

- ⚪ E4: full privacy-utility curve
- ⚪ E5: fairness analysis (E5.1 per-client AUC + E5.2 stratified by churn reason)
- ⚪ Superset dashboards functional
- ⚪ **Spark pipeline processes 1M rows end-to-end** (clean → amplify → partition),
  total time **< 15 minutes** on `local[*]` mode — benchmark table required
- ⚪ **Kafka telemetry shows real-time** per-round metrics on Superset/Grafana during FL training

### NICE (2)

- ⚪ FastAPI prediction demo working
- ⚪ **Spark-based evaluation produces the same AUC (within 0.001)** as pandas-based
  evaluation on a 10K sample (sanity check)

## 10. KNOWN RISKS & FALLBACKS

| Risk | Xác suất | Impact | Fallback / Mitigation |
|---|---|---|---|
| Synthetic quality poor (KS test fails many cols) | Trung bình | Cao | Use original 7K rows; reduce K to 5 |
| Flower integration too complex | Trung bình | Cao | Fallback to FedML or custom gRPC |
| DP-SGD drops AUC too much | Thấp | Trung bình | Report as finding (H3 may be rejected — still valid research) |
| Client dropout in Ray | Thấp | Thấp | Implement retry + log (real FL behavior) |
| Report deadline slip | Trung bình | Cao | Parallel writing from W6 |
| **Spark local mode slow / memory-hungry on laptop** | Trung bình | Trung bình | `spark.driver.memory=6g`, `spark.sql.shuffle.partitions=8`; benchmark before full 1M run |
| **Kafka + Spark + FL + Superset all in Docker Compose → resource exhaustion** | Trung bình | Cao | Skip Grafana in dev (`--profile full` only), use Superset polling; cap container memory |

**Spark fallback:** use pandas for clean/amplify (7K is small enough); use Spark **ONLY** for evaluation aggregation of the 1M test set — that is where it is truly needed.

**Kafka fallback:** replace the Kafka topic with a local file (`results/telemetry.jsonl`) that Airflow reads — loses "real-time" but keeps the metrics pipeline. Report as limitation §7.4.

## 11. DELIVERABLES

D0.5. `eda_7043.parquet` — original 50-col dataset minus `Joined` rows, kept untouched for EDA
D0.6. `clean_7043.parquet` — Customer ID + 29 features + binary target, ready for amplification
D1. Synthetic 1M rows + quality report (29 features + target, NOT 50 columns)
D2. Partitioning module (IID + label skew)
D3. FL pipeline (Flower + PyTorch)
D4. **8** experiment results (tables + plots) — E0, E1, E2, E3, E4, E5.1, E5.2, E6, E7
D5. Centralized vs FL vs Local comparison **+ IBM Churn Score reference (E0)**
D6. Privacy-utility curve
D7. MLflow experiment dashboard
D8. Superset 3 dashboards (convergence, fairness, privacy)
D9. FastAPI endpoint
D10. Final report (30–50 pages) — see §13 for outline
D11. Slides (15–20 min)
D12. Demo video (5 min)
D13. `src/data/spark_clean.py` + `src/data/spark_amplify.py` (Spark pipeline)
D14. `src/evaluation/spark_eval.py` (Spark aggregation)
D15. Kafka producer/consumer code + `docker-compose.yml` full-stack

## 12. CODING CONVENTIONS

- Python 3.10+
- Type hints mandatory in public functions
- Config via YAML or `.env` (no hardcoded paths)
- Every experiment run logs to MLflow with tags: `{experiment, partition, alpha, strategy, K, noise_multiplier}`
- Repo structure:
  ```
  src/
    data/          # clean + amplification (clean.py, spark_clean.py, spark_amplify.py)
    partitioning/  # IID + label skew (spark_partition.py)
    features/      # preprocessing
    models/        # MLP
    fl/            # Flower server/client/strategies
    baselines/     # centralized + local-only
    evaluation/    # Spark aggregation (spark_eval.py)
    telemetry/     # Kafka producer/consumer
    tracking/      # MLflow
    serving/       # FastAPI
  notebooks/       # EDA + analyses
  results/         # experiment outputs
  dags/            # Airflow
  dashboards/      # Superset exports
  docs/            # numbered docs: 00_index, 01_kickoff, 02_reading_fl, ...
  configs/         # YAML configs
  scripts/         # entrypoint shell scripts
  tests/           # pytest
  ```
- **Docs numbering convention:** every file in `docs/` is prefixed with a 2-digit number that defines reading/execution order (`00_index.md`, `01_kickoff_notes.md`, `02_reading_notes_fl.md`, `03_reading_notes_dp.md`, `04_data_decisions.md`, `05_spark_pipeline.md`, `06_synthetic_decision.md`, `07_partition_design.md`, `08_kafka_telemetry.md`, `09_fl_architecture.md`, `10_experiment_protocol.md`, `11_integration_log.md`). Never renumber existing docs — only append the next number. `00_index.md` must be updated whenever a doc is added.
- Testing: at least smoke tests for data pipeline + FL round
- Code review required (1 approval before merge)

## 13. REPORT OUTLINE (D10)

**Chapter 1 (Introduction)** — add subsection:
"1.4 Data leakage in telco churn literature — and how we avoid it"
(Discuss why most published churn papers that use the IBM Extended dataset accidentally include post-hoc features → inflated results → our contribution to methodological rigor.)

**Chapter 4 (Methodology)** — add subsections:
"4.7 Data pipeline with Spark — 1M rows in X minutes (benchmark table)"
"4.8 Kafka-based telemetry for real-time FL observability"

**Chapter 6 (Results)** — add subsections:
"6.1 E0: External baseline (IBM Churn Score)"
"6.7 E5.2: Fairness stratified by churn reason"

**Chapter 7 (Limitations)** — add:
"7.3 Single-quarter snapshot: dataset is Q3/California only — findings may not generalize temporally or geographically. Future work: multi-quarter cohort with survival analysis."
"7.4 Single-machine deployment: Spark runs in `local[*]` mode and Flower clients are simulated as processes on one host. Future work: multi-node Spark cluster and distributed Flower deployment."

## 14. COMMUNICATION NORMS

- All experiments reproducible (fixed seeds)
- Each result table must state: `(partition, alpha, strategy, K, noise_multiplier)` explicitly
- Every hypothesis must map to at least one experiment with measured numbers
- Limitations must be stated honestly in the report (e.g., right-censoring from synthetic data, cohort vs open population)
- **Every Spark benchmark must state:** dataset size (10K/100K/1M), mode (`local[*]`), driver memory, shuffle partitions, wall-clock time.
