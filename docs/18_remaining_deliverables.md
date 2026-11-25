# 18 — D15 Kafka, D14 Spark eval, D7 MLflow, D4 E0

> **Trạng thái:** ✅ cả 4 hạng mục hoàn thành
> **Ngày:** 2026-11-25

---

## 1. D4 — E0 IBM Churn Score baseline

**Vấn đề:** số 0.9421 xuất hiện trong `docs/04_data_decisions.md` và
`docs/12_architecture.md` nhưng **không có file kết quả nào** và không có lệnh
nào tái lập được. Một con số trong báo cáo phải sinh lại được bằng lệnh.

**Sửa:** `src/experiments/s18_e0_ibm_baseline.py` đọc `eda_7043.parquet`, lấy
`Churn Score` làm ranking, tính AUC univariate, ghi
`results/E0_ibm_baseline.{json,md}`.

| Metric | Value |
|---|---|
| AUC | **0.9421** |
| Samples | 6,589 |
| Churned | 1,869 |
| Score range | 5 – 96 |

Khớp chính xác con số trong docs.

**Lỗi đã sửa:** `eda_7043.parquet` giữ 50 cột gốc nên target là `Churn Label`
(Yes/No), không phải cột `churn` 0/1 (chỉ có trong artifact đã clean). Module
chấp nhận cả hai.

**Guard:** AUC ngoài band [0.90, 0.97] → log warning, vì đó là dấu hiệu cột bị
đọc sai kiểu.

---

## 2. D7 — MLflow backfill

**Vấn đề:** `log_run()` tồn tại trong `src/tracking/s14_mlflow.py` nhưng
**không experiment nào gọi nó** — hàm chỉ được gọi từ `main()` của chính nó.
Kết quả: MLflow UI trống, chỉ có 1 file `results/runs/smoke_test-*.json`.

**Sửa:** `src/tracking/s26_backfill_mlflow.py` đọc `results/E*.json` và log lại.
Chạy lại E1–E7 chỉ để populate MLflow sẽ tốn ~2 giờ cho những con số đã có sẵn.

| Experiment | Runs |
|---|---|
| E0 | 1 |
| E1 | 5 (mỗi fold) |
| E2 | 3 (mỗi partition) |
| E3 | 3 (mỗi strategy) |
| E4 | 4 (mỗi noise level) |
| E5 | 2 (mỗi strategy) |
| E6 | 4 (mỗi K) |
| E7 | 3 (mỗi fraction) |
| **Tổng** | **25** |

Mỗi run có đủ 6 required tags: `experiment, partition, alpha, strategy, K,
noise_multiplier`.

### Hạ tầng MLflow — 3 lỗi phải sửa

1. **`minio/minio:latest` không còn pull được** (Docker Hub retired repo).
   Đổi sang `bitnamilegacy/minio:latest` (cùng server, khác tag).
2. **MinIO không tạo được `/data/.minio.sys`** — image chạy uid 1001 nhưng
   volume mới thuộc root. Thêm `user: root` (chấp nhận được cho dev stack).
3. **MLflow 2.10 bind `127.0.0.1` bất kể `--host`** → port publish không tới
   được. Dùng `network_mode: host` để server listen trên loopback của host.

Cũng đổi port host MinIO sang 9100/9101 vì 9000/9001 bị Jupyter kernel chiếm.

---

## 3. D14 — Spark evaluation parity

**Vấn đề:** Layer 3 nói evaluation aggregation chạy trên Spark, nhưng mọi metric
đến từ scikit-learn trong một process Python. Nửa Spark của kiến trúc chưa được
kiểm chứng. Và **không experiment nào lưu predictions** — chỉ lưu metrics tổng,
nên Spark không có gì để đọc.

**Sửa hai bước:**

1. `src/serving/s27_export_predictions.py` — load model đã export, score
   `test_real` với canonical scaler, ghi
   `results/predictions/fl_fedavg.parquet` (`y_true`, `y_score`).
2. `src/evaluation/s28_spark_eval.py` — đọc parquet bằng Spark, tính AUC bằng
   **Spark SQL thuần**, so với pandas.

| Metric | Value |
|---|---|
| Spark AUC | 0.894901 |
| pandas AUC | 0.894901 |
| \|Δ\| | **4.74e-07** |
| Ngưỡng | 0.001 |

**Success criterion #11: PASS** — sai số nhỏ hơn ngưỡng 2000 lần.

### Vì sao viết công thức thay vì gọi MLlib

Spark MLlib có `BinaryClassificationEvaluator`, nhưng gọi nó là test MLlib chứ
không phải test logic aggregation. AUC có dạng đóng theo hạng (Mann-Whitney U):

$$\text{AUC} = \frac{\sum \text{rank}(\text{pos}) - \frac{n_{pos}(n_{pos}+1)}{2}}{n_{pos} \cdot n_{neg}}$$

Viết bằng SQL mới chứng minh được aggregation đúng. Ties xử lý bằng cách lấy
trung bình hạng mà nhóm tie chiếm — `RANK` một mình không làm được, phải
`AVG(rank)` trên nhóm.

---

## 4. D15 — Kafka telemetry

**Vấn đề:** `src/telemetry/` không tồn tại. Kafka service có trong compose
nhưng không có code publish/consume. Success criterion #9 chưa đạt.

### Kiến trúc

```
run_sequential(telemetry=True)
    │
    ├─ mỗi client xong round → fl_telemetry
    └─ mỗi round xong aggregate → fl_round_summary
                │
                ▼
        s30_consumer (postgres)
                │
                ▼
        Grafana / Superset
```

**Kafka chỉ chở metrics. Model weights đi qua gRPC/Flower và không gì khác.**
Nếu weights đi qua Kafka thì protocol không còn là FL chuẩn.

### Fire-and-forget, có chủ ý

Một run mất vài phút đến vài giờ. Nếu Kafka down, run **vẫn phải xong** —
telemetry là observability, không phải dependency. Mọi send được bọc để lỗi
broker chỉ log warning và trả `False`, không raise.

Đây là ngược lại với đường gRPC: transport fail ở đó **phải** abort round, vì
weights chính là training còn metrics thì không.

### Kết quả end-to-end

Chạy 10 rounds FedAvg IID với `telemetry=True`:

| Metric | Value |
|---|---|
| Messages gửi | **60** (50 client + 10 round) |
| Failed | **0** |
| Client rows trong postgres | 50 |
| Round rows trong postgres | 10 |
| Round 1 AUC | 0.7847 — khớp chính xác E1 |

### Idempotency

Cả hai bảng có natural key — `(round, client_id)` và `round` — khai báo là
primary key. Replay topic sẽ update chứ không duplicate, quan trọng vì consumer
chỉ commit offset sau khi ghi thành công.

### Hạ tầng Kafka — 2 lỗi phải sửa

1. **`bitnami/kafka:3.6` không còn tồn tại** (Bitnami retired tags). Đổi sang
   `apache/kafka:latest` — cùng broker, chỉ khác prefix biến (`KAFKA_` thay vì
   `KAFKA_CFG_`) và data dir (`/var/lib/kafka/data`).
2. **`PLAINTEXT_HOST` phải có trong `KAFKA_LISTENERS`**, không chỉ trong
   `ADVERTISED_LISTENERS`. Một địa chỉ advertised không phục vụ được cả compose
   network lẫn localhost, nên tách 2 listener: `kafka:9092` (nội bộ) và
   `localhost:29092` (host).

---

## 5. Artifacts

| File | Nội dung |
|---|---|
| `src/experiments/s18_e0_ibm_baseline.py` | E0 baseline |
| `src/tracking/s26_backfill_mlflow.py` | MLflow backfill |
| `src/serving/s27_export_predictions.py` | Xuất predictions |
| `src/evaluation/s28_spark_eval.py` | Spark AUC + parity check |
| `src/telemetry/s29_producer.py` | Kafka producer |
| `src/telemetry/s30_consumer.py` | Kafka → postgres |
| `results/E0_ibm_baseline.{json,md}` | AUC 0.9421 |
| `results/predictions/fl_fedavg.parquet` | 1,318 rows |
| `results/spark_eval.{json,md}` | \|Δ\| 4.74e-07 |
| `tests/test_29_telemetry.py` | 30 test |

---

## 6. Tái lập

```bash
# D4 — E0 baseline (~5 giây)
python -m src.experiments.s18_e0_ibm_baseline

# D7 — MLflow backfill (~30 giây, cần MLflow chạy)
docker compose up -d postgres minio mlflow
python -m src.tracking.s26_backfill_mlflow

# D14 — Spark eval (~30 giây)
python -m src.serving.s27_export_predictions
python -m src.evaluation.s28_spark_eval

# D15 — Kafka end-to-end
docker compose up -d kafka
python -c "
from src.fl.s11_server import run_sequential
h = run_sequential(n_rounds=10, telemetry=True)
print(h['telemetry'])
"
python -m src.telemetry.s30_consumer --max-messages 60
```

**Tests:** 272 passing (242 trước, +30 cho telemetry).
