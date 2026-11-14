# 08 — Kafka Telemetry

**Owner:** Person B + D · **Tuần:** W3–W4 · **Trạng thái:** ⬜ Chưa bắt đầu
**Deliverables:** D15 (Kafka producer/consumer + docker-compose full-stack)

---

## 1. Vai trò của Kafka (và giới hạn)

Kafka là **observability bus**. Nó KHÔNG chở model weights.

> ⚠️ **Nếu ai đề xuất dùng Kafka thay gRPC để truyền weights → SAI nguyên tắc cốt lõi.**
> gRPC/Flower là kênh DUY NHẤT cho model weights. Kafka chỉ chở metrics.

| Kafka LÀM | Kafka KHÔNG LÀM |
|---|---|
| Chở per-client metrics | Chở model weights |
| Chở per-round summary | Chở training data |
| Feed MLflow (batch) | Thay thế gRPC |
| Feed live dashboard | Streaming training |

---

## 2. Topics

| Topic | Producer | Payload | Consumer |
|---|---|---|---|
| `fl_telemetry` | Mỗi client, cuối mỗi round | `{round, client_id, local_auc, local_loss, n_samples, bytes_sent, duration_sec}` | MLflow logger + Superset live |
| `fl_round_summary` | FL server, cuối mỗi round | `{round, global_auc, global_loss, n_clients_selected, total_bytes, round_duration_sec}` | Grafana / Superset live |

### 2.1 Cấu hình topic

| Tham số | Giá trị | Lý do |
|---|---|---|
| Partitions | 3 | Đủ cho 10 client, cho phép scale |
| Replication factor | 1 | Single broker (dev) |
| Retention | 7 days | Đủ cho toàn bộ 50 rounds + debug |
| Schema validation | JSON Schema | Đơn giản, dễ debug (Avro nếu cần strict hơn) |
| Mode | **KRaft** (no ZooKeeper) | Kafka 3.5+ mặc định, ít service hơn |

### 2.2 JSON Schema — `fl_telemetry`

```json
{
  "type": "object",
  "required": ["round", "client_id", "local_auc", "local_loss", "n_samples", "bytes_sent", "duration_sec"],
  "properties": {
    "round":        {"type": "integer", "minimum": 1},
    "client_id":    {"type": "integer", "minimum": 0},
    "local_auc":    {"type": "number", "minimum": 0, "maximum": 1},
    "local_loss":   {"type": "number", "minimum": 0},
    "n_samples":    {"type": "integer", "minimum": 1},
    "bytes_sent":   {"type": "integer", "minimum": 0},
    "duration_sec": {"type": "number", "minimum": 0}
  }
}
```

### 2.3 JSON Schema — `fl_round_summary`

```json
{
  "type": "object",
  "required": ["round", "global_auc", "global_loss", "n_clients_selected", "total_bytes", "round_duration_sec"],
  "properties": {
    "round":               {"type": "integer", "minimum": 1},
    "global_auc":          {"type": "number", "minimum": 0, "maximum": 1},
    "global_loss":         {"type": "number", "minimum": 0},
    "n_clients_selected":  {"type": "integer", "minimum": 1},
    "total_bytes":         {"type": "integer", "minimum": 0},
    "round_duration_sec":  {"type": "number", "minimum": 0}
  }
}
```

---

## 3. Integration pattern

### 3.1 Flower client wrapper (`src/fl/flower_client.py`)

Trong `fit()`:
1. `t0 = time.time()`
2. `self.model.set_weights(parameters)`
3. `metrics = self.model.train(self.local_loader)` → `{loss, auc}`
4. `weights = self.model.get_weights()`
5. `producer.send("fl_telemetry", {...})`
6. `return weights, len(self.local_loader.dataset), {"local_auc": metrics["auc"]}`

**Lưu ý:** Kafka send là **fire-and-forget** (không block training). Nếu Kafka down,
training vẫn phải chạy — log warning, không raise.

### 3.2 Flower server (`src/fl/fl_server.py`)

Sau mỗi round aggregate:
`producer.send("fl_round_summary", {...})`

### 3.3 Consumers

#### MLflow consumer (`src/telemetry/kafka_to_mlflow.py`)

- Micro-batch mỗi **10 giây**
- `mlflow.log_metric("global_auc", val, step=round)`
- `mlflow.log_metric("per_client_auc", val, step=round)` với tag `client_id`
- Chạy như Airflow task `flush_kafka_to_mlflow`

#### Superset live dashboard

- Poll interval: **5–10 giây**
- 3 chart:
  1. **Global AUC vs round** — line chart, auto-refresh
  2. **Per-client AUC** — scatter, màu theo client
  3. **Communication bytes cumulative** — area chart

#### Grafana (optional)

- Chỉ bật khi `docker compose --profile full up -d`
- Dùng Kafka datasource plugin
- Cùng 3 chart như Superset

---

## 4. Docker Compose

```yaml
kafka:
  image: bitnami/kafka:3.6
  environment:
    KAFKA_CFG_NODE_ID: 0
    KAFKA_CFG_PROCESS_ROLES: controller,broker
    KAFKA_CFG_CONTROLLER_QUORUM_VOTERS: 0@kafka:9093
    KAFKA_CFG_LISTENERS: PLAINTEXT://:9092,CONTROLLER://:9093
    KAFKA_CFG_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
    KAFKA_CFG_CONTROLLER_LISTENER_NAMES: CONTROLLER
  ports: ["9092:9092"]
```

**Resource note:** Grafana dùng `profiles: ["full"]` để dev không bị ngốn RAM.

---

## 5. Fallback

Nếu Kafka gây phức tạp:
- Ghi `results/telemetry.jsonl` (append-only, mỗi dòng 1 JSON)
- Airflow đọc file thay vì consume topic
- Mất "real-time" nhưng giữ metrics pipeline
- **Báo cáo như limitation §7.4**

---

## 6. Checklist W3–W4

- [ ] Kafka service chạy trong Docker Compose (KRaft mode)
- [ ] 2 topic được tạo với đúng partitions/retention
- [ ] JSON Schema validation hoạt động
- [ ] Flower client gửi `fl_telemetry` mỗi round
- [ ] Flower server gửi `fl_round_summary` mỗi round
- [ ] MLflow consumer flush mỗi 10s
- [ ] Superset live dashboard hiển thị 3 chart
- [ ] Test: Kafka down → training vẫn chạy (không crash)
