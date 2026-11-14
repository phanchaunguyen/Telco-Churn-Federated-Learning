# Task Breakdown — Federated Learning for Telco Customer Churn

**Team:** 4 người | **Timeline:** 11 tuần | **Method:** Weekly sprints + Friday demo

---

## 👥 Vai trò & Workstream

| Thành viên | Vai trò | Workstream chính | Workstream phụ |
|---|---|---|---|
| **Person A** | Data Lead | Data amplification, partitioning, client silos, baseline evaluations | Hỗ trợ E1, E2 |
| **Person B** | FL Infra Lead | Flower server, aggregation strategies, Ray parallel, Airflow DAG | Hỗ trợ E3, E6, E7 |
| **Person C** | ML Lead | Local model, feature engineering, DP integration, per-client training | Hỗ trợ E4, E5 |
| **Person D** | MLOps & BI Lead | MLflow, Superset, FastAPI, Docker Compose, documentation | Hỗ trợ integration, báo cáo |

---

## 📅 Tuần 1 — Setup & Alignment

**Mục tiêu:** Repo sẵn sàng, môi trường chạy được, thống nhất scope.

- [ ] **All** — Kick-off, thống nhất scope, hypothesis, success criteria
  - Deliverable: `docs/kickoff_notes.md`
- [ ] **Person D** — Setup repo structure + Docker Compose skeleton
  - Deliverable: `docker-compose.yml`, `.gitignore`, `README.md`
- [ ] **Person D** — Setup MLflow server (local)
  - Deliverable: MLflow UI accessible at `localhost:5000`
- [ ] **Person A** — EDA IBM Telco gốc
  - Deliverable: `notebooks/01_eda.ipynb`
  - Nội dung: phân phối từng cột, missing rate, churn rate, correlation matrix
- [ ] **Person B** — Đọc paper FL cơ bản (McMahan 2017 FedAvg, Li 2020 FedProx)
  - Deliverable: `docs/reading_notes_fl.md`
- [ ] **Person C** — Đọc paper DP-SGD (Abadi 2016) + Opacus docs
  - Deliverable: `docs/reading_notes_dp.md`

**Friday demo:** Docker Compose up thành công, MLflow UI chạy, EDA notebook hoàn chỉnh.

---

## 📅 Tuần 2 — Data Amplification

**Mục tiêu:** Có synthetic dataset 1M+ dòng, chất lượng được đánh giá.

- [ ] **Person A** — Implement Gaussian Copula generator
  - Deliverable: `src/data/copula_generator.py`
  - Output: `data/synthetic/copula_1m.parquet`
- [ ] **Person A** — Implement CTGAN generator
  - Deliverable: `src/data/ctgan_generator.py`
  - Output: `data/synthetic/ctgan_1m.parquet`
- [ ] **Person A + C** — Quality evaluation (KS test, correlation, train-test gap)
  - Deliverable: `notebooks/02_synthetic_quality.ipynb`
  - So sánh: real (7K) vs copula vs CTGAN
- [ ] **All** — Review & chọn phương pháp chính
  - Deliverable: `docs/synthetic_decision.md`
- [ ] **Person B** — Setup Ray cluster (local multi-process)
  - Deliverable: `src/infra/ray_setup.py`, verification script
- [ ] **Person B** — Flower skeleton (server + client base class)
  - Deliverable: `src/fl/server.py`, `src/fl/client_base.py`

**Friday demo:** So sánh chất lượng 2 phương pháp synthetic, chọn 1 làm primary.

**⚠️ Risk:** Nếu synthetic data chất lượng kém (KS test fail nhiều cột) → fallback dùng data gốc 7K, giảm K xuống 5–10.

---

## 📅 Tuần 3 — Partitioning & Local Preprocessing

**Mục tiêu:** 10 client silo sẵn sàng, feature pipeline hoạt động.

- [ ] **Person A** — IID partitioning
  - Deliverable: `src/partitioning/iid_partitioner.py`
- [ ] **Person A** — Label skew partitioning (Dirichlet α)
  - Deliverable: `src/partitioning/label_skew_partitioner.py`
  - Support: α = 0.5 và α = 0.1
- [ ] **Person A** — Partition metadata & visualization
  - Deliverable: `notebooks/03_partition_analysis.ipynb`
  - Vẽ: churn ratio per client, data size per client
- [ ] **Person C** — Feature engineering pipeline
  - Deliverable: `src/features/feature_engineering.py`
  - Encoding, derived features, feature selection
- [ ] **Person C** — Local normalization (per-client stats only)
  - Deliverable: `src/features/local_preprocessor.py`
  - **Quan trọng:** không dùng global stats (privacy violation)
- [ ] **Person A + D** — Client silo setup (MinIO buckets)
  - Deliverable: 10 buckets `s3://clients/client_{i}/`
  - Verification: mỗi client có data riêng, không overlap

**Friday demo:** Show 10 client silos với phân phối label khác nhau (IID vs α=0.5 vs α=0.1).

---

## 📅 Tuần 4 — FL Core: FedAvg Baseline

**Mục tiêu:** End-to-end FL chạy được 1 round, không crash.

- [ ] **Person C** — MLP model definition (PyTorch)
  - Deliverable: `src/models/mlp.py`
  - Architecture: 30 → 128 → 64 → 32 → 1
- [ ] **Person C** — Local training loop
  - Deliverable: `src/fl/local_trainer.py`
  - Support: batch size, epochs, early stopping
- [ ] **Person B** — Flower client wrapper (tích hợp local_trainer)
  - Deliverable: `src/fl/flower_client.py`
- [ ] **Person B** — Flower server + FedAvg strategy
  - Deliverable: `src/fl/fl_server.py`
  - Config: 50 rounds, fraction_fit=0.5, fraction_evaluate=0.5
- [ ] **Person B + C** — **End-to-end 1 round chạy được**
  - Deliverable: `scripts/run_fl_baseline.sh`
  - Verification: global model weights updated sau 1 round
- [ ] **Person D** — MLflow logging integration
  - Deliverable: `src/tracking/mlflow_logger.py`
  - Log: global_auc, per_client_auc, communication_bytes, round_duration

**Friday demo:** Chạy 1 round FL end-to-end, show MLflow log.

**⚠️ Risk:** Flower integration phức tạp → B dành buffer time, fallback dùng FedML hoặc custom gRPC.

---

## 📅 Tuần 5 — FL Round Loop & E1 (FL Gap)

**Mục tiêu:** Chạy đủ 50 rounds trên IID, có kết quả E1.

- [ ] **Person B** — Full 50-round training (IID partition)
  - Deliverable: Run log + model checkpoint
- [ ] **Person C** — Convergence analysis (AUC vs round)
  - Deliverable: `notebooks/04_convergence_iid.ipynb`
- [ ] **Person A** — Centralized baseline
  - Deliverable: `src/baselines/centralized.py`
  - Train trên toàn bộ 1M dòng
- [ ] **Person A** — Local-only baseline
  - Deliverable: `src/baselines/local_only.py`
  - Train riêng từng client, không aggregate
- [ ] **All** — **E1 — FL gap analysis**
  - Deliverable: `results/E1_fl_gap.md` + `results/E1_fl_gap.png`
  - Bảng: Centralized vs FL vs Local-only (AUC, F1, Recall@10%)

**Friday demo:** Bảng E1 hoàn chỉnh — chứng minh FL gap < 5% AUC (IID).

---

## 📅 Tuần 6 — Non-IID & Aggregation Strategies (E2 + E3)

**Mục tiêu:** Kết quả E2 (partition impact) và E3 (aggregation robustness).

- [ ] **Person B** — Chạy FL trên α=0.5 và α=0.1
  - Deliverable: 2 run logs + MLflow experiments
- [ ] **Person B** — FedProx implementation (μ=0.01)
  - Deliverable: `src/fl/strategies/fedprox.py`
- [ ] **Person B** — FedAvgM implementation (β=0.9)
  - Deliverable: `src/fl/strategies/fedavgm.py`
- [ ] **Person B** — Chạy 3 strategies trên α=0.1
  - Deliverable: 3 run logs
- [ ] **Person C** — **E2 — Partition impact analysis**
  - Deliverable: `results/E2_partition_impact.md` + plot
  - Bảng: IID vs α=0.5 vs α=0.1 (AUC, convergence rounds)
- [ ] **Person C** — **E3 — Aggregation comparison**
  - Deliverable: `results/E3_aggregation.md` + plot
  - Bảng: FedAvg vs FedProx vs FedAvgM trên α=0.1

**Friday demo:** E2 + E3 hoàn chỉnh — chứng minh non-IID làm giảm AUC, FedProx robust hơn.

---

## 📅 Tuần 7 — Privacy (E4) & Fairness (E5)

**Mục tiêu:** Đường cong privacy-utility + phân tích fairness.

- [ ] **Person C** — Opacus DP-SGD integration
  - Deliverable: `src/fl/dp_trainer.py`
  - Config: noise_multiplier, max_grad_norm=1.0, delta=1e-5
- [ ] **Person C** — Chạy với noise_multiplier ∈ {0, 0.5, 1, 2}
  - Deliverable: 4 run logs + ε tracking
- [ ] **Person C** — **E4 — Privacy-utility tradeoff**
  - Deliverable: `results/E4_privacy_utility.md` + curve (AUC vs ε)
- [ ] **Person B** — **E5 — Fairness analysis**
  - Deliverable: `results/E5_fairness.md` + boxplot
  - Metric: per-client AUC (mean, std, min) cho FedAvg vs FedProx

**Friday demo:** Đường cong AUC vs ε + fairness boxplot.

**⚠️ Risk:** DP-SGD làm AUC giảm quá nhiều → chấp nhận và báo cáo như finding (H3 có thể bị reject — đó cũng là kết quả nghiên cứu).

---

## 📅 Tuần 8 — Scalability (E6) & Communication (E7)

**Mục tiêu:** Kết quả scalability + communication cost.

- [ ] **Person B** — Chạy K ∈ {5, 10, 20, 50}
  - Deliverable: 4 run logs
- [ ] **Person B** — Chạy fraction_sampled ∈ {0.25, 0.5, 1.0}
  - Deliverable: 3 run logs
- [ ] **Person B** — **E6 — Scalability analysis**
  - Deliverable: `results/E6_scalability.md` + plot (time vs K)
- [ ] **Person B** — **E7 — Communication cost**
  - Deliverable: `results/E7_communication.md` + plot (bytes vs round)
- [ ] **Person A** — Hỗ trợ: kiểm tra data consistency khi K thay đổi
  - Deliverable: verification script

**Friday demo:** E6 + E7 hoàn chỉnh.

---

## 📅 Tuần 9 — Serving & BI

**Mục tiêu:** Dashboard + API hoạt động.

- [ ] **Person D** — FastAPI prediction endpoint
  - Deliverable: `src/serving/prediction_api.py`
  - Endpoint: `POST /predict` → `{churn_prob, risk_segment, top_factors}`
- [ ] **Person D** — Superset Dashboard 1 (Convergence)
  - Deliverable: `dashboards/convergence.json`
  - Chart: AUC vs round, per-client AUC
- [ ] **Person D** — Superset Dashboard 2 (Fairness)
  - Deliverable: `dashboards/fairness.json`
  - Chart: boxplot per-client AUC, min/std
- [ ] **Person D** — Superset Dashboard 3 (Privacy)
  - Deliverable: `dashboards/privacy.json`
  - Chart: AUC vs ε curve
- [ ] **Person B + D** — Airflow DAG end-to-end
  - Deliverable: `dags/fl_pipeline_dag.py`
  - Tasks: prepare → partition → FL train → eval → dashboard refresh

**Friday demo:** 3 dashboards + API demo.

---

## 📅 Tuần 10 — Integration, Testing & Báo cáo (Draft)

**Mục tiêu:** Pipeline end-to-end ổn định, báo cáo draft đầy đủ.

- [ ] **All** — End-to-end integration test
  - Deliverable: `docs/integration_log.md`
  - Chạy full pipeline từ data → FL → dashboard
- [ ] **All** — Cross code review (mỗi PR cần 1 approval)
  - Deliverable: Review comments trên GitHub
- [ ] **Person D + C** — Viết Ch. 1–2 (Introduction, Related Work)
  - Deliverable: `report/ch1_ch2_draft.md`
- [ ] **Person A + B** — Viết Ch. 3–4 (Methodology, Architecture)
  - Deliverable: `report/ch3_ch4_draft.md`
- [ ] **All** — Viết Ch. 5–6 (Experiments, Results)
  - Deliverable: `report/ch5_ch6_draft.md`
  - Mỗi người viết phần experiment mình phụ trách
- [ ] **Person C** — Viết Ch. 7 (Conclusion, Future Work, Limitations)
  - Deliverable: `report/ch7_draft.md`

**Friday demo:** Full pipeline chạy end-to-end từ 1 command.

**⚠️ Risk:** Báo cáo dồn vào 2 tuần cuối rất căng → khuyến nghị bắt đầu viết từ tuần 6 (mỗi experiment xong → viết section tương ứng).

---

## 📅 Tuần 11 — Finalization

**Mục tiêu:** Báo cáo final, slide, demo video sẵn sàng.

- [ ] **Person D (lead) + All** — Slide thuyết trình 15–20 phút
  - Deliverable: `presentation/final_slides.pdf`
  - Structure: Motivation → Hypothesis → Method → Results (E1–E7) → Conclusion
- [ ] **Person D** — Demo video 5 phút
  - Deliverable: `presentation/demo_video.mp4`
  - Nội dung: Docker up → run FL → show dashboard → API predict
- [ ] **All** — Final report proofread
  - Deliverable: `report/final_report.pdf` (30–50 trang)
- [ ] **All** — Rehearsal (2 lần)
  - Deliverable: Practice runs, Q&A prep

**Final deliverable:** Báo cáo PDF + Slide + Video + GitHub repo public.

---

## 🔄 Cross-cutting Responsibilities

| Task | Owner | Frequency |
|---|---|---|
| **Weekly sync** | Rotating facilitator | Monday 30 phút |
| **Mid-week check-in** | Rotating | Wednesday 15 phút |
| **Friday demo + retro** | Rotating | Friday 45 phút |
| **Code review** | Cross-review (1 approval required) | Per PR |
| **Experiment log** | Người chạy experiment | Per run → MLflow + shared sheet |
| **Documentation** | D (lead), mỗi người viết doc cho module mình | Continuous |
| **Report sections** | Mỗi người viết phần liên quan | Continuous từ tuần 6 |

---

## 📊 Dependency Graph

```
Week 1: Setup (D + A)
   │
   ▼
Week 2: Amplification (A) ──► Week 3: Partitioning (A) ──► Week 4: FL Core (B + C)
   │                                                              │
   └──► Week 3: Feature Eng (C) ──────────────────────────► Week 5: E1 (A+B+C)
                                                                  │
                                              Week 6: E2 + E3 (B) ──► Week 7: E4 + E5 (C+B)
                                                                  │
                                              Week 8: E6 + E7 (B) ──► Week 9: Serving (D)
                                                                  │
                                              Week 10–11: Integration + Report (All)
```

**Critical path:** A (data) → B (FL infra) → experiments → D (serving) → report.
**Parallel track:** C (ML) chạy song song với B sau week 4.

---

## ⚠️ Risk Management

| Risk | Xác suất | Impact | Mitigation |
|---|---|---|---|
| Synthetic data chất lượng kém | Trung bình | Cao | Validate KS test; fallback dùng data gốc 7K, giảm K xuống 5–10 |
| Flower integration phức tạp | Trung bình | Cao | Buffer time week 2; fallback dùng FedML hoặc custom gRPC |
| DP-SGD làm AUC giảm quá nhiều | Thấp | Trung bình | Chấp nhận và báo cáo như finding |
| Client fail/dropout trong Ray | Thấp | Thấp | Implement retry + log (đây là behavior FL thực tế) |
| Team member unavailable | Trung bình | Cao | Cross-training; mỗi module có ≥ 2 người hiểu |
| Report deadline trượt | Trung bình | Cao | Viết báo cáo song song từ week 6 |

---

## 🎯 Success Criteria

Project thành công nếu đạt **≥ 6/8** tiêu chí:

| # | Tiêu chí | Mức độ |
|---|---|---|
| 1 | FL chạy end-to-end 50 rounds trên IID, không crash | **Bắt buộc** |
| 2 | E1: FL gap < 5% AUC so với centralized (IID) | **Bắt buộc** |
| 3 | E2: non-IID làm giảm AUC (có số liệu) | **Bắt buộc** |
| 4 | E3: FedProx/FedAvgM robust hơn FedAvg (có số liệu) | **Bắt buộc** |
| 5 | E4: đường cong privacy-utility đầy đủ | Nên có |
| 6 | E5: phân tích fairness (per-client AUC) | Nên có |
| 7 | Superset dashboard hiển thị kết quả | Nên có |
| 8 | FastAPI demo prediction hoạt động | Nice-to-have |

**Pass condition:** ≥ 4 bắt buộc + ≥ 2 "nên có" → đủ để bảo vệ.

---

## 📌 Weekly Ritual

| Day | Activity | Duration | Output |
|---|---|---|---|
| **Monday** | Weekly sync — review tuần trước, plan tuần này | 30 phút | Meeting notes |
| **Wednesday** | Mid-week check-in — blocker? | 15 phút | Blocker list |
| **Friday** | Demo + Retrospective | 45 phút | Demo recording + retro notes |

**Nguyên tắc:**
- Mỗi task phải có **deliverable cụ thể** (file, notebook, bảng kết quả) — không có "task mơ hồ".
- Mỗi người tự log progress vào shared sheet.
- Mỗi PR cần 1 approval từ người khác trước khi merge.