# 05 — Spark Data Pipeline

**Owner:** Person A · **Tuần:** W2–W3 · **Trạng thái:** ⬜ Chưa bắt đầu
**Deliverables:** D13 (`spark_clean.py`, `spark_amplify.py`), D14 (`spark_eval.py`)

---

## 1. Vai trò của Spark (và giới hạn)

Spark là **batch data processing layer**. Nó KHÔNG train model và KHÔNG nằm trong FL loop.

| Spark LÀM | Spark KHÔNG LÀM |
|---|---|
| Clean 7K×50 → 6,589×31 | Train model |
| Amplify → 1M rows | Chở model weights |
| Partition → 10 silos | Tham gia FL round |
| Aggregate evaluation results | Streaming (không dùng Structured Streaming) |

> ⚠️ **Nguyên tắc bất biến:** gRPC/Flower là kênh DUY NHẤT cho model weights.
> Spark chỉ là plumbing cho data — không bao giờ là research variable.

---

## 2. Ba module

### 2.1 `src/data/spark_clean.py` (W2)

**Input:** `data/raw/telco.csv` (7,043 × 50)
**Output:** `data/processed/clean_7043.parquet` (6,589 × 31)

**Các bước:**
1. Đọc CSV với **explicit schema** (không `inferSchema` cho production — chậm và dễ sai dtype)
2. `filter(F.col("Customer Status") != "Joined")`
3. `drop(*LEAKAGE, *CONSTANTS, *DERIVED, *GEO)` — dùng chung danh sách với `src/data/clean.py`
4. `withColumn("churn", (F.col("Customer Status") == "Churned").cast("int"))`
5. Ghi Parquet

**Assertions bắt buộc:**
- Không còn cột nào trong `DROPPED_COLS`
- Không còn row `Joined`
- `churn` ∈ {0, 1}
- 31 cột
- Row count == 6,589
- Churn rate == 0.2837 (khớp pandas reference)

**Cạm bẫy phải xử lý (giống pandas):**
- Header lỗi `"Total Refu    nds"` → rename về `"Total Refunds"`
- `"None"` là category hợp lệ của `Offer` / `Internet Type` → **không** set `nullValue="None"`

### 2.2 `src/data/spark_amplify.py` (W2)

**Input:** `clean_7043.parquet` (6,589 × 31)
**Output:** `data/synthetic/synthetic_1M.parquet` (1,000,000 × 30)

**Hai lựa chọn:**

| Option | Cách làm | Ưu | Nhược |
|---|---|---|---|
| **A (khuyến nghị)** | Fit SDV Gaussian Copula trên pandas sample 5K rows → Spark apply sampled indices song song | Chất lượng cao, giữ correlation | Cần pandas ở driver |
| **B** | Spark-native per-partition sampling từ fitted distribution (không SDV) | Nhanh, thuần Spark | Chất lượng thấp hơn |

**Quality gate — chạy trên PANDAS, không trên Spark:**
- KS test từng cột (α = 0.05)
- Correlation matrix diff
- Train-on-synthetic / test-on-real AUC gap < 5%

> Quality gate là **validation step**, không phải data processing → dùng pandas cho đơn giản.

### 2.3 `src/partitioning/spark_partition.py` (W3)

**Input:** `synthetic_1M.parquet`
**Output:** 10 Parquet files trên MinIO (`s3a://clients/client_{i}/features.parquet`)

**Các bước:**
1. `assign_client_id(df, strategy, alpha, K)` → thêm cột `client_id`
   - `iid`: shuffle + chia đều
   - `label_skew`: Dirichlet(α) trên phân phối nhãn
2. Với mỗi `cid` ∈ [0, K):
   `df.filter(client_id == cid).drop("client_id").write.parquet(f"s3a://clients/client_{cid}/")`

**Assertions (success criterion — data isolation):**
- Mỗi silo có đúng **29 features + target**
- Không silo nào chứa cột leakage
- Các silo **disjoint** theo `Customer ID`
- Tổng rows các silo == 1,000,000

### 2.4 `src/evaluation/spark_eval.py` (W5–W8)

**Input:** per-client predictions (`s3a://eval/client_*/predictions.parquet`)
**Output:** `results/E*.parquet` + `results/E*.csv`

**Aggregations:**
- Global AUC
- Per-client AUC (E5.1)
- Per-churn-category AUC (E5.2) — join `Churn Category` từ `eda_7043.parquet` **chỉ trên test set**
- Communication bytes cumulative (E7)

---

## 3. Benchmark protocol (success criterion #8)

| Dataset size | Mode | Driver mem | Shuffle partitions | Wall-clock |
|---|---|---|---|---|
| 10K | `local[*]` | 6g | 8 | must measure |
| 100K | `local[*]` | 6g | 8 | must measure |
| 1M | `local[*]` | 6g | 8 | **< 15 min (target)** |

Bảng này vào report **§4.7**.

**Cấu hình Spark:**
```yaml
spark:
  master: "local[*]"
  driver_memory: 6g
  executor_memory: 4g
  shuffle_partitions: 8
```

### 3.1 Kết quả đo được (2026-11-19)

Chạy `python scripts/benchmark_spark.py` trên máy 16 GB, Java 21, PySpark 4.3.0.

| n_rows | mode | driver | shuffle | clean | amplify | partition | **total** | under 15 min |
|---|---|---|---|---|---|---|---|---|
| 10,000 | `local[*]` | 6g | 8 | 3.25s | 3.48s | 3.49s | **10.22s** | ✅ |
| 100,000 | `local[*]` | 6g | 8 | 0.22s | 1.99s | 4.85s | **7.06s** | ✅ |
| 1,000,000 | `local[*]` | 6g | 8 | 0.15s | 8.52s | 14.53s | **23.20s (0.39 min)** | ✅ |

**1M end-to-end: 0.39 phút** — nhanh hơn ngưỡng 15 phút **38×**.

Ghi chú đọc bảng:
- 10K chậm hơn 100K ở cột `clean` vì lần chạy đầu bao gồm JVM warm-up và
  đọc CSV lần đầu. Đây là hiện tượng bình thường, không phải lỗi.
- `partition` là stage tốn thời gian nhất ở 1M (14.53s) vì phải ghi 10 thư mục
  parquet riêng biệt.
- Toàn bộ nằm trong `local[*]` — không có cluster thật.

### 3.2 Parity với pandas

| Artifact | pandas | Spark | Khớp |
|---|---|---|---|
| `clean_7043` rows | 6,589 | 6,589 | ✅ |
| `clean_7043` churned | 1,869 | 1,869 | ✅ |
| `clean_7043` churn rate | 0.2837 | 0.2837 | ✅ (Δ < 1e-9) |
| `synthetic_1m` rows | 1,000,000 | 1,000,000 | ✅ |
| Partition sizes (IID) | 90,000 × 10 | 90,000 × 10 | ✅ |
| Partition churn (α=0.1) | 0.0087–0.9123 | 0.0087–0.9123 | ✅ |

### 3.3 Ghi chú về MinIO

`spark_partition.py` mặc định ghi **local parquet** (`use_minio=False`).

Lý do: ghi lên MinIO qua `s3a://` cần `hadoop-aws` + AWS SDK v2 bundle
(**533 MB**). Nạp bộ jar đó làm mỗi lần khởi động Spark mất thêm vài phút,
trong khi layout local **giống hệt** về nội dung.

Bật MinIO khi cần demo end-to-end:

```python
build_partitions_spark("iid", use_minio=True)
```

Jars đã tải sẵn trong `jars/` (49 MB, không gồm AWS SDK).

---

## 4. Spark vs pandas — phân chia trách nhiệm

| Bước | Công cụ | Lý do |
|---|---|---|
| Clean | **Spark** (primary) + pandas (reference) | Spark cho scale, pandas cho test nhanh |
| Amplify | **Spark** (parallel sampling) | 1M rows |
| Partition | **Spark** | repartition + write song song |
| Quality gate | **pandas** | Validation, không phải processing |
| FL training | **KHÔNG dùng Spark** | Flower + PyTorch |
| Eval aggregation | **Spark** | 1M test set |

**`src/data/clean.py` (pandas) được GIỮ làm reference implementation:**
1. Chạy `pytest tests/test_clean.py` nhanh (không cần Spark)
2. Sanity check: Spark output phải khớp pandas output (shape, column set, churn rate)

---

## 5. Fallback

Nếu Spark gây chậm/treo:
- Dùng **pandas** cho clean + amplify (7K đủ nhỏ)
- Spark **chỉ** cho evaluation aggregation của 1M test set — đó là nơi thực sự cần
- Báo cáo như limitation §7.4

---

## 6. Checklist W2–W3

- [ ] `spark_clean.py` chạy được, output khớp pandas reference
- [ ] `spark_amplify.py` sinh 1M rows
- [ ] Quality gate pass (KS, correlation, AUC gap)
- [ ] `spark_partition.py` sinh 10 silos trên MinIO
- [ ] Assertion: không silo nào chứa leakage
- [ ] Benchmark table 10K/100K/1M hoàn chỉnh
- [ ] `docs/06_synthetic_decision.md` ghi quyết định Copula vs CTGAN
