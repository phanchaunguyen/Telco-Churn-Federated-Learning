# 01 — Kickoff Notes

**Ngày:** W1 · **Người tham gia:** A, B, C, D · **Trạng thái:** ✅ Đã thống nhất

---

## 1. Project identity

**Title:** Federated Learning for Telecom Customer Churn Prediction: A Systematic Study of
Non-IID Robustness, Aggregation Strategies, and Privacy-Utility Tradeoff

**Problem statement**

Các nhà mạng viễn thông có dữ liệu khách hàng phân tán ở nhiều chi nhánh và **không thể tập trung**
vì quy định privacy (GDPR, PDPA) và cạnh tranh. Làm sao huấn luyện model churn chính xác **mà không
tập trung dữ liệu**, trong điều kiện dữ liệu giữa các chi nhánh không đồng nhất (non-IID)?

**Core premise:** dữ liệu không bao giờ rời silo — chỉ model weights được truyền.

---

## 2. Hypotheses

| # | Hypothesis | Experiment |
|---|---|---|
| **H1** (main) | FL đạt hiệu năng gần centralized (< 5% AUC gap dưới IID; < 10% dưới non-IID) mà không chia sẻ dữ liệu | E1, E2 |
| **H2** | Dưới label skew nghiêm trọng (Dirichlet α ≤ 0.1), FedProx và FedAvgM vượt FedAvg về global AUC và per-client fairness | E3 |
| **H3** | DP-SGD với ε ∈ [1, 10] giảm AUC ≤ 5%; tồn tại "sweet spot" ε | E4 |
| **H4** | Client ít dữ liệu hoặc phân phối label cực đoan bị thiệt; FedProx giảm chênh lệch so với FedAvg | E5.1, E5.2 |

---

## 3. Research questions

| RQ | Câu hỏi | Hypothesis | Experiment |
|---|---|---|---|
| RQ1 | FL đạt bao nhiêu % hiệu năng centralized dưới IID vs non-IID? | H1 | E1, E2 |
| RQ2 | Aggregation strategy nào robust nhất với label skew? | H2 | E3 |
| RQ3 | Privacy-utility tradeoff (ε vs AUC) cho churn là gì? | H3 | E4 |
| RQ4 | FL có gây unfairness giữa các client không? Đo và giảm thế nào? | H4 | E5.1, E5.2 |
| RQ5 | FL có scale được khi K tăng từ 5 lên 50? | — | E6 |
| RQ6 | Chi phí communication của FL có khả thi thực tế? | — | E7 |

---

## 4. Scope — trong và ngoài

### ✅ Trong scope

- Dataset: IBM Telco **EXTENDED** (7,043 × 50) → clean → amplify 1M+ → partition
- **Spark batch pipeline**: clean → amplify → partition → eval aggregation
- **Kafka telemetry bus**: `fl_telemetry` + `fl_round_summary` → MLflow + live dashboard
- K = 10 client silos (MinIO), benchmark K ∈ {5, 10, 20, 50}
- 3 aggregation strategies: FedAvg, FedProx (μ=0.01), FedAvgM (β=0.9)
- DP-SGD qua Opacus, sweep noise_multiplier ∈ {0, 0.5, 1, 2}
- 8 experiments: E0, E1, E2, E3, E4, E5.1, E5.2, E6, E7
- Serving: MLflow + Superset (3 static + 1 live) + FastAPI

### ❌ Ngoài scope (đã cân nhắc và loại)

| Loại | Lý do |
|---|---|
| Dataset "Temporal Churn & LTV" | Closed cohort, right-censored, panel 1–11 cycles, không join được |
| Dùng 7K gốc làm training set duy nhất | Phải amplify lên 1M+ trước |
| Join dữ liệu giữa các silo | Vi phạm nguyên tắc cô lập dữ liệu — đây là điểm cốt lõi của FL |
| K-means / PCA / unsupervised | Đây là bài toán supervised classification trên tabular |
| TensorFlow | Stack là PyTorch + Flower + Ray |
| "Centralize dữ liệu" như giải pháp | Phá vỡ tiền đề privacy-preserving |
| `Churn Score` làm feature | Label leakage — chỉ dùng làm E0 baseline |
| **Kafka làm kênh truyền model weights** | **gRPC/Flower là kênh DUY NHẤT cho weights** |
| **Spark Structured Streaming / Kafka Streams** | **Pipeline là batch; Kafka chỉ là metrics bus** |
| **Spark dùng để train model** | **Spark chỉ làm data pipeline + eval aggregation** |

---

## 5. Success criteria

Dự án pass nếu đạt **≥ 4 bắt buộc + ≥ 3 "nên có"** (tổng 11 tiêu chí).

### Bắt buộc (4)

| # | Tiêu chí | Trạng thái |
|---|---|---|
| 1 | FL chạy end-to-end 50 rounds IID, không crash | ⬜ |
| 2 | E1: FL gap < 5% AUC vs centralized (IID) | ⬜ |
| 3 | E2: non-IID làm giảm AUC (có số liệu) | ⬜ |
| 4 | E3: FedProx/FedAvgM vượt FedAvg (có số liệu) | ⬜ |

### Nên có (5)

| # | Tiêu chí | Trạng thái |
|---|---|---|
| 5 | E4: đường cong privacy-utility đầy đủ | ⬜ |
| 6 | E5: fairness analysis (E5.1 per-client + E5.2 stratified) | ⬜ |
| 7 | Superset dashboards hoạt động | ⬜ |
| 8 | **Spark pipeline xử lý 1M rows end-to-end** (clean → amplify → partition), tổng thời gian **< 15 phút** trên `local[*]` — cần bảng benchmark | ⬜ |
| 9 | **Kafka telemetry hiển thị real-time** per-round metrics trên Superset/Grafana trong lúc FL training | ⬜ |

### Nice-to-have (2)

| # | Tiêu chí | Trạng thái |
|---|---|---|
| 10 | FastAPI prediction demo hoạt động | ⬜ |
| 11 | **Spark-based evaluation cho AUC khớp pandas** (sai số ≤ 0.001) trên 10K sample | ⬜ |

---

## 6. RACI

| Hoạt động | A (Data) | B (FL Infra) | C (ML) | D (MLOps/BI) |
|---|---|---|---|---|
| Clean + EDA (W1) | **R** | C | C | A |
| **Spark clean + amplify (W2)** | **R** | C | C | I |
| **Spark partitioning (W3)** | **R** | C | I | A |
| **Kafka topic setup (W3)** | I | **R** | I | C |
| Feature engineering (W3) | C | I | **R** | I |
| MLP + local trainer (W4) | I | C | **R** | I |
| Flower server/client (W4) | I | **R** | C | I |
| **Kafka producer trong Flower (W4)** | I | **R** | C | I |
| **Kafka → MLflow consumer (W4)** | I | C | I | **R** |
| MLflow logging (W4) | I | C | I | **R** |
| E1 baselines (W5) | **R** | C | C | I |
| **Spark eval aggregation (W5–W8)** | **R** | C | C | I |
| E2 + E3 (W6) | I | **R** | **R** | I |
| E4 + E5 (W7) | I | C | **R** | I |
| E6 + E7 (W8) | C | **R** | I | I |
| Serving + BI (W9) | I | C | I | **R** |
| **Live dashboard từ Kafka (W9)** | I | C | I | **R** |
| Integration (W10) | C | C | C | **R** |
| Report | **R** | **R** | **R** | **R** |

`R` = Responsible · `A` = Accountable · `C` = Consulted · `I` = Informed

---

## 7. Timeline (11 tuần)

| Tuần | Nội dung | Deliverable chốt |
|---|---|---|
| W1 | Setup + EDA + **CLEAN** + reading | D0.5, D0.6, docs 00–04 |
| W2 | **Spark** clean + **Spark** amplify + quality gate | D1, **D13**, `05_spark_pipeline.md`, `06_synthetic_decision.md` |
| W3 | **Spark** partitioning + **Kafka** topic setup | D2, `07_partition_design.md`, `08_kafka_telemetry.md` |
| W4 | FL core (gRPC) + **Kafka telemetry hookup** | D3, **D15**, `09_fl_architecture.md` |
| W5 | 50 rounds IID + E1 + **Spark eval aggregation** | `results/E1_fl_gap.md` |
| W6 | Non-IID + FedProx + FedAvgM + E2 + E3 | `results/E2_*.md`, `E3_*.md` |
| W7 | DP-SGD + E4 + E5.1 + E5.2 | `results/E4_*.md`, `E5_*.md` |
| W8 | Scalability + communication + E6 + E7 | `results/E6_*.md`, `E7_*.md` |
| W9 | Serving + **live dashboard từ Kafka** | D8, D9, **D15** |
| W10 | Integration + report draft | `11_integration_log.md` |
| W11 | Slides + demo video + final report | D10, D11, D12 |

**Critical path:** A (data) → B (FL infra) → experiments → D (serving) → report
**Parallel track:** C (ML) chạy song song với B sau W4
**Writing policy:** viết báo cáo song song từ W6 — **không** dồn W10

---

## 8. Communication norms

| Ngày | Hoạt động | Thời lượng | Output |
|---|---|---|---|
| Thứ 2 | Weekly sync | 30 phút | Meeting notes |
| Thứ 4 | Mid-week check-in | 15 phút | Blocker list |
| Thứ 6 | Demo + Retrospective | 45 phút | Demo recording + retro notes |

**Nguyên tắc:**
- Mỗi task phải có deliverable cụ thể (file, notebook, bảng kết quả) — không có task mơ hồ.
- Mỗi PR cần 1 approval từ người khác trước khi merge.
- Mọi experiment log lên MLflow với tags `{experiment, partition, alpha, strategy, K, noise_multiplier}`.
- Mọi kết quả phải reproducible (fixed seed = 42).

---

## 9. Risks & fallbacks

| Risk | Xác suất | Impact | Fallback |
|---|---|---|---|
| Synthetic data chất lượng kém | Trung bình | Cao | Dùng 7K gốc, giảm K xuống 5 |
| Flower integration phức tạp | Trung bình | Cao | FedML hoặc custom gRPC |
| DP-SGD giảm AUC quá nhiều | Thấp | Trung bình | Báo cáo như finding (H3 reject vẫn hợp lệ) |
| Client dropout trong Ray | Thấp | Thấp | Retry + log (đây là behavior FL thực tế) |
| Thành viên không available | Trung bình | Cao | Cross-training, mỗi module ≥ 2 người hiểu |
| Report deadline trượt | Trung bình | Cao | Viết song song từ W6 |
| **Spark local mode chậm / ngốn RAM** | Trung bình | Trung bình | `driver.memory=6g`, `shuffle.partitions=8`; benchmark trước khi chạy full 1M |
| **Docker Compose quá nhiều service → hết RAM** | Trung bình | Cao | Skip Grafana ở dev (`--profile full`), dùng Superset polling; cap memory container |

**Fallback bổ sung:**
- **Spark:** dùng pandas cho clean/amplify (7K đủ nhỏ), Spark **chỉ** cho eval aggregation 1M test set.
- **Kafka:** thay topic bằng file `results/telemetry.jsonl`, Airflow đọc — mất "real-time" nhưng giữ metrics pipeline. Báo cáo như limitation §7.4.

---

## 10. Quyết định đã chốt ở W1

| # | Quyết định | Lý do |
|---|---|---|
| 1 | Dùng bản **50 cột** thực tế, cập nhật `promt.md` | Data thật khác spec 21 cột |
| 2 | Bỏ `Joined` **trước** amplification | Khách mới chưa đủ cửa sổ quan sát |
| 3 | **29 features** (không phải 27/28) | `Number of Referrals` bị bỏ sót trong patch gốc |
| 4 | `clean_7043.parquet` = **31 cột** | ID + 29 features + target |
| 5 | Giữ cả 3 cột billing | MLP tự xử lý collinearity |
| 6 | `Offer = "None"` giữ làm category riêng | Không gộp, giữ thông tin |
| 7 | `Churn Score` chỉ dùng cho **E0** | Không bao giờ là feature |
| 8 | Docs đánh số 2 chữ số | Thứ tự đọc/thực thi rõ ràng |
| 9 | **gRPC là kênh DUY NHẤT cho model weights** | Chuẩn FL, không thay đổi |
| 10 | **Kafka chỉ chở telemetry** | Tách observability khỏi transport |
| 11 | **Spark chỉ làm data + eval, không train** | Tránh overlap trách nhiệm |
| 12 | **Giữ `src/data/clean.py` (pandas) làm reference** | Test nhanh + sanity check cho Spark output |
