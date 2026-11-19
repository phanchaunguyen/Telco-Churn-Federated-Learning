# 07 — Partition Design

**Owner:** Person A · **Tuần:** W3 · **Trạng thái:** ✅ Đã chốt
**Nguồn số liệu:** `results/partition_metadata.json`, `results/silo_verification.json`

---

## 1. Quyết định

| Tham số | Giá trị |
|---|---|
| Số client K | **10** |
| Strategies | **IID**, **Dirichlet α=0.5**, **Dirichlet α=0.1** |
| Input | `synth_train.parquet` (900,000 × 30) |
| Output | 10 parquet/client × 3 strategies |
| Seed | 42 |

---

## 2. Thuật toán

### 2.1 IID

```
order = rng.permutation(n)
assignment[order] = arange(n) % k
```

Mỗi client nhận `n/k` rows ngẫu nhiên. Churn rate kỳ vọng = global rate.

### 2.2 Dirichlet label skew

Với mỗi class `c`:

```
idx_c = các row có label c, shuffle
p ~ Dirichlet(α · 1_k)              # tỉ lệ class c chia cho k client
p = max(p, MIN_CLIENT_SHARE)        # floor 0.02
p = p / p.sum()                     # renormalize
counts = floor(p · |idx_c|)
counts = max(counts, MIN_CLASS_COUNT)   # floor 20 rows/class/client
phân phối phần dư theo largest fractional part
```

**Hai guard và lý do:**

| Guard | Giá trị | Lý do |
|---|---|---|
| `MIN_CLIENT_SHARE` | 0.02 | Dirichlet draw có thể cho client 0.001 → silo 285 rows, không train được |
| `MIN_CLASS_COUNT` | 20 | Client chỉ có 1 class → BCE loss không xác định, model không học được decision boundary |

**Tại sao floor + renormalize thay vì cap cứng:**

Lần thử đầu dùng cap `MAX_CLIENT_SHARE = 0.25`. Kết quả: α=0.1 cho churn rate
`0.2282` ở 7/10 client — **skew bị triệt tiêu hoàn toàn**, không còn là non-IID
nữa. Floor + renormalize giữ được skew tương đối (client draw 0.001 vẫn thấp
hơn nhiều so với client draw 0.6) trong khi đảm bảo mọi silo dùng được.

---

## 3. Kết quả đo được

### 3.1 IID

| Client | Rows | Churn rate |
|---|---|---|
| 0–9 | 90,000 mỗi client | 0.2819 – 0.2872 |

- **std = 0.0018** — nhiễu lấy mẫu thuần túy
- Tổng: 900,000 rows

### 3.2 Label skew α = 0.5 (moderate non-IID)

| Client | Rows | Churn rate |
|---|---|---|
| 0 | 31,445 | 0.1783 |
| 1 | 89,767 | 0.0548 |
| 2 | 34,359 | 0.6464 |
| 3 | 65,032 | 0.0756 |
| 4 | 356,622 | 0.0138 |
| 5 | 38,278 | 0.6826 |
| 6 | 101,224 | 0.6323 |
| 7 | 93,679 | 0.8703 |
| 8 | 41,280 | 0.7056 |
| 9 | 48,314 | 0.2513 |

- **churn range: 0.0138 – 0.8703**, std = 0.3254
- Client 4 gần như toàn "stayed", client 7 gần như toàn "churned"

### 3.3 Label skew α = 0.1 (severe non-IID)

| Client | Rows | Churn rate |
|---|---|---|
| 0 | 15,955 | 0.2912 |
| 1 | 31,885 | 0.1457 |
| 2 | 15,955 | 0.2912 |
| 3 | 17,891 | 0.3679 |
| 4 | 15,955 | 0.2912 |
| 5 | 531,450 | 0.0087 |
| 6 | 22,293 | 0.4927 |
| 7 | 93,254 | 0.8787 |
| 8 | 128,994 | 0.9123 |
| 9 | 26,368 | 0.5711 |

- **churn range: 0.0087 – 0.9123**, std = 0.2939
- Client 5 chiếm **59% dữ liệu** và gần như toàn "stayed" (0.87% churn)
- Client 8 gần như toàn "churned" (91.2%)

Đây là **severe label skew đúng nghĩa** — chính là regime mà H2 dự đoán FedProx
và FedAvgM sẽ vượt FedAvg.

---

## 4. Silo verification gate

`python -m src.partitioning.s06_verify_silos` → **PASS**

| Check | IID | α=0.5 | α=0.1 |
|---|---|---|---|
| 10 client files tồn tại | ✅ | ✅ | ✅ |
| Không cột leakage | ✅ | ✅ | ✅ |
| 29 features mỗi silo | ✅ | ✅ | ✅ |
| Disjoint theo Customer ID | ✅ | ✅ | ✅ |
| Tổng rows = 900,000 | ✅ | ✅ | ✅ |
| Cả 2 class mỗi silo | ✅ | ✅ | ✅ |
| Không NaN | ✅ | ✅ | ✅ |

**Unique IDs:** 900,000 / 900,000 cho cả 3 strategies — không overlap.

---

## 5. Artifacts

| Path | Nội dung |
|---|---|
| `data/synthetic/partitions_iid/client_{0..9}.parquet` | 10 × 90,000 rows |
| `data/synthetic/partitions_label_skew_a0.5/client_{0..9}.parquet` | 10 silo, skew vừa |
| `data/synthetic/partitions_label_skew_a0.1/client_{0..9}.parquet` | 10 silo, skew mạnh |
| `results/partition_metadata.json` | size + churn rate per client |
| `results/silo_verification.json` | kết quả gate |

Mỗi parquet: `Customer ID` (SYN-0000001…) + 29 features + `churn` = **31 cột**.

---

## 6. Bàn giao cho Layer 2

| Cho | Artifact |
|---|---|
| FL client | `partitions_{tag}/client_{i}.parquet` |
| MLflow tags | `partition`, `alpha` từ `partition_metadata.json` |
| E2 (partition impact) | so sánh 3 strategies |
| E3 (aggregation) | dùng `label_skew_a0.1` |
| E5.1 (fairness) | per-client AUC trên từng silo |

---

## 7. Tái lập

```bash
python -m src.partitioning.s05_partition    # sinh 3 bộ partition
python -m src.partitioning.s06_verify_silos   # gate
pytest tests/test_05_partition.py -v         # 20 tests
```

---

## 8. Ghi chú cho báo cáo

- **α=0.1 có một client chiếm 59% dữ liệu.** Đây là hệ quả tự nhiên của
  Dirichlet với α nhỏ, không phải lỗi. Cần nêu trong §7 Limitations rằng
  FedAvg sẽ bị chi phối bởi client lớn — đúng như H4 dự đoán.
- **Client 5 (α=0.1) có 0.87% churn** — gần như single-class. Đây là stress test
  tốt nhất cho FedProx: nếu proximal term hoạt động, client này không được kéo
  global model lệch.
- **IID std = 0.0018** chứng minh partitioner đúng — sai lệch chỉ là nhiễu
  binomial với n = 90,000.
