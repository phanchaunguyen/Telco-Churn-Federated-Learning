# 00 — Documentation Index

> **Quy ước:** mọi file trong `docs/` có tiền tố **2 chữ số** = thứ tự đọc/thực thi.
> Không đánh số lại khi thêm doc mới — chỉ cấp số tiếp theo.
> Số chẵn chục (00, 10, 20…) dành cho doc tổng hợp/index; số lẻ cho doc chuyên đề.

---

## Đọc theo thứ tự này

| # | File | Owner | Tuần | Trạng thái | Nội dung |
|---|---|---|---|---|---|
| **00** | `00_index.md` | D | W1 | ✅ | Mục lục, thứ tự đọc, trạng thái |
| **01** | `01_kickoff_notes.md` | All | W1 | ✅ | Scope, H1–H4, RQ1–RQ6, success criteria, RACI |
| **02** | `02_reading_notes_fl.md` | B | W1 | ✅ | McMahan 2017 (FedAvg), Li 2020 (FedProx) |
| **03** | `03_reading_notes_dp.md` | C | W1 | ✅ | Abadi 2016 (DP-SGD) + Opacus |
| **04** | `04_data_decisions.md` | A | W1 | ✅ | Leakage taxonomy, 29 features, số liệu EDA |
| **05** | `05_spark_pipeline.md` | A | W2 | ✅ | Spark clean/amplify/partition/eval + benchmark |
| **06** | `06_synthetic_decision.md` | A+C | W2 | ✅ | Copula vs CTGAN, quality gate PASS (TSTR gap 3.7%) |
| **07** | `07_partition_design.md` | A | W3 | ✅ | IID vs Dirichlet, số liệu thực, silo gate PASS |
| **08** | `08_kafka_telemetry.md` | B+D | W3–W4 | ✅ | Kafka topics, schema, consumers, live dashboard |
| **09** | `09_fl_architecture.md` | B | W4 | ✅ | **GroupNorm rationale, FL stack, 1 round milestone** |
| **10** | `10_experiment_protocol.md` | B | W5 | ✅ | **Protocol E0–E7, quyết định scaler, fold protocol** |
| 11 | `11_integration_log.md` | All | W10 | ⬜ | Log tích hợp end-to-end |
| **12** | `12_architecture.md` | All | W1 | ✅ | **Sơ đồ Mermaid: 4 layers, data pipeline, FL round, experiment protocol, non-overlap** |
| **13** | `13_layer1_plan.md` | A | W2–W3 | 🚧 | **Kế hoạch hoàn thành Layer 1: 9 phases, 12 artifacts, 8 commits** |
| **14** | `14_naming_convention.md` | All | W1 | ✅ | **Quy ước đánh số file/thư mục theo thứ tự pipeline** |
| **15** | `15_w6_results.md` | B | W6 | ✅ | **E2 partition impact (PASS) + E3 aggregation (FAIL), 2 lỗi metric, FedAvgM divergence** |
| **16** | `16_w8_results.md` | C | W8 | ✅ | **E6 scalability (PASS, cost phẳng) + E7 communication (PASS, 0.18% AUC cho 5× băng thông)** |

---

## Đường đọc gợi ý theo vai trò

| Vai trò | Đọc |
|---|---|
| **Person A** (Data) | 01 → 04 → **05** → 06 → 07 → **12** |
| **Person B** (FL Infra) | 01 → 02 → **08** → 09 → 10 → **12** |
| **Person C** (ML) | 01 → 03 → 04 → 06 → **12** |
| **Person D** (MLOps/BI) | 01 → 04 → **08** → 10 → 11 → **12** |
| **Người mới vào dự án** | 00 → 01 → **12** → 04 → 05 → 08 → 02 → 03 |

---

## Số liệu then chốt (từ `notebooks/01_eda.ipynb`)

| Metric | Value |
|---|---|
| Raw shape | 7,043 × 50 |
| `Joined` rows dropped | 454 (6.45%) |
| Cohort sau khi bỏ `Joined` | 6,589 |
| Churn rate (clean cohort) | **28.37%** |
| Churned / Stayed | 1,869 / 4,720 |
| Features giữ lại | **29** |
| Cột bị drop | 20 (Loại 1–6) |
| **E0 — IBM Churn Score AUC** | **0.9421** |
| `Churn Category` strata | 5 |
| Zero-variance columns | 3 |

---

## Cập nhật index này khi nào?

Mỗi khi thêm một file mới vào `docs/`:

1. Cấp số tiếp theo (không tái sử dụng số cũ).
2. Thêm dòng vào bảng "Đọc theo thứ tự này".
3. Cập nhật cột trạng thái: ⬜ chưa có → 🚧 đang viết → ✅ hoàn thành.
