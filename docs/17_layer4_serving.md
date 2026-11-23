# 17 — Layer 4: Serving & BI

> **Trạng thái:** ✅ FastAPI chạy · ✅ Superset 3 dashboard · ✅ Grafana live · ✅ Airflow DAG
> **Ngày:** 2026-11-23

---

## 1. Vấn đề chặn đã phát hiện

Trước Layer 4, **không có model weights nào được lưu trên đĩa**. `run_sequential`
trả `final_weights` trong memory, các experiment chỉ ghi metrics ra JSON. FastAPI
không có gì để load.

Đã sửa bằng `src/serving/s21_export_model.py`: train đúng cấu hình E1 fold 0 rồi
lưu

| Artifact | Nội dung |
|---|---|
| `results/models/fl_fedavg.pt` | `state_dict` + kiến trúc (input_dim, hidden_dims, dropout, norm, num_groups) |
| `results/models/model_metadata.json` | metrics trên `test_real`, để API báo cáo mà không cần load checkpoint |
| `data/processed/preprocessing_iid.json` | vocabulary + scaler từng client |

**Kết quả export:** AUC **0.8949**, PR-AUC 0.8121, Recall@10% 0.3289 — khớp
chính xác E1 fold 0, nên model được serve chính là model trong báo cáo.

### Vì sao lưu kiến trúc cùng weights

`state_dict` một mình không đủ để reload: kiến trúc phụ thuộc `input_dim`,
`hidden_dims`, `dropout`, `norm`, `num_groups`. Lưu kèm để `load_model()` dựng
lại đúng module mà người gọi không phải nhớ config.

---

## 2. FastAPI — `src/serving/s22_api.py`

| Route | Chức năng |
|---|---|
| `GET /health` | Liveness + model đã load chưa |
| `GET /model-info` | Model đang serve + metrics hold-out |
| `POST /predict` | 29 features thô → xác suất churn |

**Kiểm chứng thật** (không chỉ unit test):

```
row 0: actual_churn=0  prob=0.3859  pred=False   ✓
row 1: actual_churn=1  prob=0.6784  pred=True    ✓
```

### Scaler: canonical `train_real`, không phải scaler của client

Mỗi silo có `LocalScaler` riêng fit trên thống kê silo đó. Một khách hàng mới
đến API **không thuộc silo nào**, nên không có scaler cục bộ để áp. Dùng scaler
canonical fit trên `train_real` — đúng cách E1–E7 đánh giá global model. Serving
và evaluation tiền xử lý giống hệt nhau.

Nếu dùng scaler khác, mọi input sẽ bị dịch âm thầm và chất lượng dự đoán giảm
mà **không có lỗi nào được ném ra**. Đó là lý do scaler được load từ artifact
export chứ không fit lại.

### Validation

`extra: "forbid"` trong Pydantic schema: gửi thừa field → 422, không bị bỏ qua
âm thầm. Thiếu field → 422. Categorical phải là string (encoder so sánh với
vocabulary, float sẽ không bao giờ khớp).

---

## 3. Superset — 3 dashboard

### Vì sao Superset chứ không chỉ Grafana

| Tiêu chí | Superset | Grafana |
|---|---|---|
| Biểu đồ BI (bar, pie, pivot) | ✅ 50+ loại, trau chuốt | 🟡 ~15 loại, thiên monitoring |
| Time-series | 🟡 poll SQL | ✅ native |
| Live Kafka | ❌ | ✅ native |
| Ad-hoc SQL | ✅ SQL Lab | ❌ |

→ **Superset cho 3 dashboard tĩnh** (đẹp hơn, đúng D8), **Grafana cho live**
(native streaming, đúng success criterion #9).

### Kiến trúc

```
results/E*.json
      │  s23_export_results.py
      ▼
postgres (mlflow db)
  ├── e1_convergence  150 rows  (3 partition × 50 rounds)
  ├── e5_fairness      30 rows  (2 strategy × (10 client + 5 reason))
  └── e4_privacy        4 rows  (4 noise level)
      │  s25_superset_dashboards.py (REST API)
      ▼
Superset :8088 — 3 dashboard, 6 chart
```

### Vì sao mỗi dashboard một bảng riêng

Mỗi dashboard hỏi một câu khác nhau và cần grain khác nhau. Gộp vào một bảng
rộng sẽ khiến mọi chart phải lọc bỏ phần lớn rows, và SQL đằng sau mỗi chart
khó đọc hơn chính cái chart.

### Dashboard đã tạo

| Dashboard | Chart | Nguồn |
|---|---|---|
| FL Telco — Convergence | AUC/round IID, AUC/round non-IID | `e1_convergence` |
| FL Telco — Fairness | Per-client AUC, Recall by reason | `e5_fairness` |
| FL Telco — Privacy | Privacy-utility curve, AUC by σ | `e4_privacy` |

**Kiểm chứng:** `GET /api/v1/dashboard/` trả 3 dashboard, `GET /api/v1/chart/`
trả 6 chart. Dữ liệu query được:

```
       run       | rounds | min_auc | max_auc
-----------------+--------+---------+---------
 iid             |     50 |  0.7847 |  0.8033
 label_skew_a0.1 |     50 |  0.7800 |  0.8004
 label_skew_a0.5 |     50 |  0.7815 |  0.8033
```

### Superset init — 4 bước, không fallback

`dashboards/superset/init_superset.sh` chạy tuần tự, `set -euo pipefail`:

1. `superset db upgrade` — tạo/migrate metadata tables
2. `superset fab create-admin` — tạo tài khoản (idempotent: kiểm tra tồn tại trước)
3. `superset init` — load roles, permissions, datasource mặc định
4. `superset run` — webserver foreground

Bước nào fail → container dừng. **Không có fallback**: Superset nửa vời còn tệ
hơn Superset dừng, vì dashboard sẽ âm thầm thiếu mà báo cáo vẫn tuyên bố có.

---

## 4. Grafana — live telemetry

`dashboards/grafana/live_telemetry.json` — 5 panel, auto-refresh 5s:

1. AUC/round IID baseline
2. AUC/round non-IID partitions
3. Per-client AUC (FedAvg vs FedProx)
4. Recall by churn reason
5. Privacy-utility table

**Đây là near-real-time, không phải native streaming.** Kafka consumer ghi vào
postgres, Grafana poll. Native Kafka streaming cần plugin Grafana Kafka — chưa
cài. Ghi rõ để không ai hiểu nhầm là streaming thật.

### Panel 1 đã sửa giữa đường

Ban đầu panel 1 query `source = 'E1'`, nhưng **E1 JSON không lưu AUC curve** —
chỉ có metrics cuối mỗi fold. Panel sẽ trống. Đã đổi sang query E2 (run duy
nhất có curve per-round) và ghi rõ lý do trong description.

---

## 5. Airflow DAG — `dags/s24_fl_pipeline_dag.py`

```
data_prep → fl_train → evaluate → export_dashboards
```

**Viết đúng chuẩn Airflow 2.x nhưng KHÔNG chạy.** Airflow cần metadata DB +
scheduler + webserver (~2 GB container), trong khi pipeline đủ ngắn để chạy tay.

Dùng `BashOperator` chứ không `PythonOperator` vì mỗi stage đã là CLI entry
point (`python -m src...`). DAG chỉ mô tả thứ tự và phụ thuộc, không lặp lại
business logic. Cùng lệnh đó chạy được ngoài Airflow — đúng cách project này
thực sự chạy.

Import an toàn khi không có Airflow: `_HAS_AIRFLOW` guard, `dag = None`.

---

## 6. Artifacts

| File | Nội dung |
|---|---|
| `src/serving/s21_export_model.py` | Train + export model |
| `src/serving/s22_api.py` | FastAPI |
| `src/serving/s23_export_results.py` | E*.json → postgres |
| `src/serving/s25_superset_dashboards.py` | Tạo dashboard qua REST API |
| `dashboards/superset/superset_config.py` | Config Superset |
| `dashboards/superset/init_superset.sh` | Entrypoint 4 bước |
| `dashboards/grafana/live_telemetry.json` | 5 panel live |
| `dashboards/grafana/datasources/postgres.yml` | Datasource |
| `dags/s24_fl_pipeline_dag.py` | DAG (không chạy) |
| `results/models/fl_fedavg.pt` | Model weights |
| `tests/test_22_api.py` | 10 test |
| `tests/test_23_export_results.py` | 16 test |

---

## 7. Tái lập

```bash
# 1. Export model (~6 phút)
python -m src.serving.s21_export_model

# 2. Chạy API
python -m src.serving.s22_api --port 8000
curl http://localhost:8000/health

# 3. Khởi động hạ tầng BI
docker compose --profile full up -d postgres superset grafana

# 4. Đẩy kết quả vào postgres
python -m src.serving.s23_export_results

# 5. Tạo dashboard
python -m src.serving.s25_superset_dashboards

# Truy cập
#   Superset : http://localhost:8088  (admin / admin)
#   Grafana  : http://localhost:3000  (admin / admin)
```

**Tests:** 242 passing (216 trước Layer 4).
