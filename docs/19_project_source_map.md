# 19 — Bản đồ source code

> **Mục đích:** mô tả cấu trúc repo, ý nghĩa từng module, và vị trí của nó trong
> kiến trúc 4 layer. Dùng khi cần tìm "code này nằm đâu" hoặc "module này làm gì".
>
> **Quy ước đánh số:** mọi file có thứ tự thực thi đều có tiền tố số. Python
> không cho module bắt đầu bằng chữ số nên dùng `sNN_`; mọi thứ khác dùng `NN_`.
> Số = vị trí trong pipeline, không phải thứ tự viết.

---

## 1. Cây thư mục

```
source/
├── configs/
│   └── base.yaml                    # Toàn bộ tham số, không hardcode ở code
├── data/
│   ├── raw/telco.csv                # 7,043 × 50 (gitignored)
│   ├── processed/                   # ✅ tracked (~2.6 MB)
│   │   ├── eda_7043.parquet         # 6,589 × 50 — giữ cột gốc cho EDA/E0
│   │   ├── clean_7043.parquet       # 6,589 × 31 — ID + 29 features + target
│   │   ├── train_real.parquet       # 5,271 × 31
│   │   ├── test_real.parquet        # 1,318 × 31
│   │   ├── preprocessing_iid.json   # vocabulary + scaler từng client
│   │   └── fold_1..4/               # artifact riêng cho E1 fold 1–4
│   └── synthetic/                   # ❌ gitignored (~184 MB)
│       ├── synthetic_1m.parquet     # 1,000,000 × 30
│       ├── synth_train.parquet      # 900,000 × 30
│       ├── synth_test.parquet       # 100,000 × 30
│       ├── quality_report.json      # ✅ tracked — 6 chiều đánh giá
│       ├── amplify_metadata.json    # ✅ tracked
│       └── partitions_*/            # 10 silo × 3 strategy + K sweep
├── src/
│   ├── config.py                    # Pydantic loader, resolve path tuyệt đối
│   ├── data/                        # LAYER 1 — chuẩn bị dữ liệu
│   ├── partitioning/                # LAYER 1 — chia silo
│   ├── features/                    # LAYER 2 — encode + scale
│   ├── models/                      # LAYER 2 — kiến trúc MLP
│   ├── fl/                          # LAYER 2 — vòng lặp federated
│   ├── telemetry/                   # LAYER 2 — Kafka bus
│   ├── evaluation/                  # LAYER 3 — metrics + Spark eval
│   ├── baselines/                   # LAYER 3 — centralized / local-only
│   ├── experiments/                 # LAYER 3 — runner E0–E7
│   ├── tracking/                    # LAYER 3 — MLflow
│   └── serving/                     # LAYER 4 — API + BI export
├── tests/                           # 272 test
├── docs/                            # 00–21
├── dags/                            # Airflow DAG (viết, không chạy)
├── dashboards/                      # Superset + Grafana provisioning
├── scripts/                         # shell/python tiện ích
├── results/                         # ✅ E*.json/md tracked
│   ├── models/fl_fedavg.pt          # ✅ 76 KB — model đã export
│   └── predictions/                 # ✅ 12 KB — per-row y_true/y_score
└── docker-compose.yml               # 7 service
```

---

## 2. Layer 1 — Chuẩn bị dữ liệu

### 2.1 pandas path (reference implementation)

| Module | Trách nhiệm | Input → Output |
|---|---|---|
| `s01_clean.py` | Single source of truth cho column taxonomy. Sửa header lỗi `"Total Refu    nds"`, drop 20 cột leakage/constant/duplicate, derive target | `telco.csv` → `eda_7043.parquet` + `clean_7043.parquet` |
| `s02_split.py` | Stratified 80/20 **trước** khi amplify, để test set là dữ liệu thật | `clean_7043` → `train_real` + `test_real` |
| `s03_amplify.py` | Rank-based Gaussian Copula sinh 1M rows | `train_real` → `synthetic_1m` |
| `s04_quality_gate.py` | 6 chiều: marginal (KS/TVD), correlation, TSTR, diversity, memorization, validity | `synthetic_1m` → `quality_report.json` |
| `s05_partition.py` | IID + Dirichlet (α=0.5, 0.1) với floor 0.02 + `MIN_CLASS_COUNT=20` | `synth_train` → `partitions_*/` |
| `s06_verify_silos.py` | 7 chiều: completeness, disjointness, schema, class presence, size floor, skew fidelity, reproducibility | `partitions_*/` → `silo_verification.json` |

### 2.2 Spark path (parity với pandas)

| Module | Trách nhiệm |
|---|---|
| `s00_spark_session.py` | Factory. `PYSPARK_PYTHON=sys.executable` để executor dùng đúng venv |
| `s01_clean_spark.py` | Bản Spark của `s01_clean`, parity Δ<1e-9 |
| `s03_amplify_spark.py` | Bản Spark của `s03_amplify`, 1M rows trong 36.9s |
| `s05_partition_spark.py` | Bản Spark của `s05_partition`, khớp pandas chính xác |

**Vì sao có 2 path:** pandas là reference dễ đọc và test; Spark chứng minh pipeline
scale được. Benchmark 1M rows: pandas ~15 phút, Spark **0.39 phút**.

---

## 3. Layer 2 — Federated training

| Module | Trách nhiệm | Ghi chú thiết kế |
|---|---|---|
| `s07_preprocess.py` | One-hot theo vocabulary chung + `LocalScaler` fit trên **từng silo** | Vocabulary là union category name — metadata, không phải dữ liệu |
| `s08_mlp.py` | MLP 56→128→64→32→1, **GroupNorm** không phải BatchNorm | GroupNorm vì Opacus không wrap được BatchNorm, và BatchNorm buffer hỏng dưới non-IID FedAvg |
| `s09_trainer.py` | Vòng lặp train local, hook FedProx, hook DP-SGD | Không import Flower → test được không cần server |
| `s10_client.py` | Adapter `NumPyClient` | Chỉ gửi weights + `num_examples`, không gửi dữ liệu |
| `s11_server.py` | `run_sequential()` (không Ray) + `run_simulation()` (Ray) | FedAvg/FedProx/FedAvgM; đo `round_times`, `total_seconds` |
| `s29_producer.py` | Kafka producer, fire-and-forget | Broker chết → log warning, training tiếp tục |
| `s30_consumer.py` | Kafka → postgres | Natural key `(round, client_id)` để replay không duplicate |

**Ranh giới bất biến:** gRPC/Flower là kênh **duy nhất** cho model weights.
Kafka chỉ chở metrics. Spark không bao giờ nằm trong vòng lặp FL.

---

## 4. Layer 3 — Đánh giá

| Module | Trách nhiệm |
|---|---|
| `s12_eval.py` | AUC, PR-AUC, F1, Recall@10%, bootstrap CI (1000 resample), per-client, per-reason recall |
| `s13_centralized.py` | Upper bound: train trên toàn bộ `synth_train` |
| `s13_local_only.py` | Lower bound: mỗi client tự train, không aggregate |
| `s14_mlflow.py` | Log run + JSON fallback khi server tắt |
| `s15_e1_fl_gap.py` | E1: 5-fold CV, so centralized/federated/local-only |
| `s16_e2_e3.py` | E2 partition impact + E3 aggregation robustness |
| `s17_e4_dp.py` | E4 DP-SGD sweep σ ∈ {0, 0.5, 1, 2} |
| `s18_e0_ibm_baseline.py` | E0: AUC univariate của `Churn Score` |
| `s18_e5_fairness.py` | E5.1 per-client + E5.2 recall theo lý do churn |
| `s19_e6_scalability.py` | E6: K ∈ {5, 10, 20, 50} |
| `s20_e7_communication.py` | E7: fraction ∈ {0.25, 0.5, 1.0} |
| `s26_backfill_mlflow.py` | Đọc `results/E*.json` → log lại MLflow |
| `s28_spark_eval.py` | D14: AUC bằng Spark SQL, so với pandas |

---

## 5. Layer 4 — Serving & BI

| Module | Trách nhiệm |
|---|---|
| `s21_export_model.py` | Train đúng cấu hình E1 fold 0, lưu `fl_fedavg.pt` + `preprocessing_iid.json` |
| `s22_api.py` | FastAPI: `/health`, `/model-info`, `/predict` |
| `s23_export_results.py` | `results/E*.json` → 3 bảng postgres cho Superset |
| `s25_superset_dashboards.py` | Tạo 3 dashboard + 6 chart qua REST API |
| `s27_export_predictions.py` | Xuất `y_true`/`y_score` per-row cho Spark eval |

---

## 6. Hạ tầng

| File | Nội dung |
|---|---|
| `docker-compose.yml` | postgres, minio, mlflow, kafka, spark-master/worker, superset, grafana |
| `dags/s24_fl_pipeline_dag.py` | Airflow DAG `data_prep → fl_train → evaluate → export_dashboards` (viết, không chạy) |
| `dashboards/superset/` | `superset_config.py` + `init_superset.sh` (4 bước, không fallback) |
| `dashboards/grafana/` | `live_telemetry.json` (5 panel) + datasource postgres |
| `scripts/` | `00_up.sh`, `01_verify_env.py`, `02_run_clean.sh`, `03_benchmark_spark.py` |

---

## 7. Bảng tra nhanh: muốn X thì đọc file nào

| Muốn biết | Đọc |
|---|---|
| Cột nào là feature, cột nào bị drop và vì sao | `src/data/s01_clean.py` |
| Synthetic data có đáng tin không | `results/quality_report.json` + `docs/06_synthetic_decision.md` |
| Silo có chồng lấn không | `results/silo_verification.json` |
| Model có bao nhiêu tham số | `src/models/s08_mlp.py` → 18,049 |
| FedAvg được implement thế nào | `src/fl/s11_server.py` → `run_sequential()` |
| Metric được tính thế nào | `src/evaluation/s12_eval.py` |
| Kết quả E4 (privacy) | `results/E4_dp_sgd.md` |
| Vì sao E3 FAIL | `docs/15_w6_results.md` §2 |
| Vì sao E5.2 dùng recall không dùng AUC | `docs/15_w6_results.md` §7 |
| API nhận input gì | `http://localhost:8000/docs` hoặc `src/serving/s22_api.py` |
| Dashboard lấy dữ liệu từ đâu | `src/serving/s23_export_results.py` |

---

## 8. Quy ước code

| Quy ước | Lý do |
|---|---|
| Không hardcode path | Mọi path qua `src/config.py`, resolve tuyệt đối từ `REPO_ROOT` |
| Mỗi module có `main()` | Chạy được độc lập: `python -m src.data.s01_clean` |
| Docstring nêu **vì sao**, không chỉ **làm gì** | Ví dụ: GroupNorm giải thích 3 lý do, không chỉ "dùng GroupNorm" |
| Test không cần service ngoài | Kafka/MLflow/postgres đều có fallback hoặc skip |
| Số liệu trong docs phải sinh lại được bằng lệnh | E0 từng chỉ có số trong docs, không tái lập được — đã sửa |
