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

## 5. Đánh giá chất lượng partition — 7 chiều

Partition không chỉ cần "chạy được". Một partition tồi có thể khiến toàn bộ thí
nghiệm FL vô nghĩa (ví dụ: client chỉ có 1 class → BCE loss không xác định).
Dưới đây là 7 phép kiểm tra độc lập, mỗi phép bắt một loại lỗi khác nhau.

| # | Chiều | Câu hỏi | Bắt được lỗi gì |
|---|---|---|---|
| 1 | **Completeness** | Mọi row đều được gán? | Row bị mất hoặc gán 2 lần |
| 2 | **Disjointness** | Các silo không overlap? | Rò rỉ dữ liệu giữa client |
| 3 | **Schema integrity** | Đúng 31 cột, 29 features? | Cột leakage lọt vào |
| 4 | **Class presence** | Mỗi silo có cả 2 class? | BCE loss undefined |
| 5 | **Size floor** | Không silo nào quá nhỏ? | Client không đủ 1 batch |
| 6 | **Skew fidelity** | Skew có đúng mức yêu cầu? | α=0.1 vô tình thành IID |
| 7 | **Reproducibility** | Cùng seed → cùng kết quả? | Kết quả không tái lập |

### 5.1 Chiều 1 — Completeness

**Cách làm:** `sizes.sum() == len(df)` và `set(unique(assignment)) == set(range(k))`.

**Kết quả:** cả 3 strategies, tổng rows = **900,000** = `synth_train` rows. Không
row nào bị mất, không client nào rỗng.

### 5.2 Chiều 2 — Disjointness (đo trên file thực tế)

**Cách làm:** đọc 10 file parquet từ disk, lấy `set(Customer ID)` mỗi silo, kiểm
tra pairwise intersection.

**Kết quả:**

| Strategy | Tổng IDs các silo | Union (unique) | Pairwise overlap | Disjoint |
|---|---|---|---|---|
| IID | 900,000 | 900,000 | **0** | ✅ |
| α=0.5 | 900,000 | 900,000 | **0** | ✅ |
| α=0.1 | 900,000 | 900,000 | **0** | ✅ |

Đây là kiểm tra **trên artifact thực tế**, không chỉ trên array in-memory — bắt
được lỗi ghi file (ví dụ ghi nhầm client_id).

### 5.3 Chiều 3 — Schema integrity

**Cách làm:** mỗi silo phải có đúng `['Customer ID', *29 features, 'churn']` = 31
cột, và **không** cột nào trong `DROPPED_COLS` (20 cột leakage/constant/geo/dup).

**Kết quả:**

| Strategy | Số silo có leakage | Số silo sai schema | Tổng NaN |
|---|---|---|---|
| IID | **0** | **0** | **0** |
| α=0.5 | **0** | **0** | **0** |
| α=0.1 | **0** | **0** | **0** |

Đây là **hard requirement** của dự án: nếu một silo chứa `Churn Score` thì toàn
bộ thí nghiệm FL mất giá trị.

### 5.4 Chiều 4 — Class presence

**Cách làm:** mỗi silo phải có ≥ `MIN_CLASS_COUNT` (20) rows của **cả hai** class.
Nếu thiếu, BCE loss không xác định và model không học được decision boundary.

**Kết quả — min class count mỗi silo:**

| Strategy | Min churn count | Min stay count | Pass |
|---|---|---|---|
| IID | 25,373 | 64,151 | ✅ |
| α=0.5 | **4,915** | **12,151** | ✅ |
| α=0.1 | **4,646** | **11,308** | ✅ |

Ngay cả ở α=0.1 (severe skew), silo nghèo nhất vẫn có 4,646 churned và 11,308
stayed — dư sức train.

### 5.5 Chiều 5 — Size floor

**Cách làm:** mỗi silo phải có ≥ `MIN_CLIENT_SHARE × n_rows / k` = 0.02 × 900,000
/ 10 = **1,800 rows**.

**Kết quả:**

| Strategy | Min silo size | Max silo size | Ratio max/min |
|---|---|---|---|
| IID | 90,000 | 90,000 | **1.0×** |
| α=0.5 | 31,445 | 356,622 | 11.3× |
| α=0.1 | 15,955 | 531,450 | **33.3×** |

α=0.1 có một client chiếm 59% dữ liệu — đây là **hệ quả tự nhiên** của Dirichlet
với α nhỏ, không phải lỗi. Cần nêu trong Limitations.

### 5.6 Chiều 6 — Skew fidelity ⚠️ **phát hiện quan trọng**

**Cách làm:** so sánh churn-rate spread giữa các strategy, và kiểm tra floor có
đang "nuốt" mất skew không.

**Kết quả spread:**

| Strategy | churn min | churn max | spread | std |
|---|---|---|---|---|
| IID | 0.2819 | 0.2872 | **0.0053** | 0.0018 |
| α=0.5 | 0.0138 | 0.8703 | **0.8565** | 0.3254 |
| α=0.1 | 0.0087 | 0.9123 | **0.9036** | 0.2939 |

**IID spread 0.0053** — đúng bằng 4× binomial SE (0.0013 × 4 = 0.0052). Đây là
nhiễu lấy mẫu thuần túy, chứng minh partitioner IID đúng.

**α=0.1 spread 0.9036 > α=0.5 spread 0.8565** — skew tăng đúng hướng khi α giảm.

#### ⚠️ Nhưng floor đang binding mạnh

Đo số client nằm **đúng tại ngưỡng floor**:

| Strategy | Client ở min churn count | Client ở min stay count |
|---|---|---|
| α=0.5 | **3/10** (4,915) | **4/10** (12,151) |
| α=0.1 | **5/10** (4,646) | **2/10** (11,308) |

**Nguyên nhân:** raw Dirichlet draw ở α=0.1 cho kết quả cực đoan:

```
alpha=0.1 raw proportions:
  [0.9024, 0.0588, 0.0004, 0.0000, 0.0378, 0.0000, 0.0004, 0.0000, 0.0000, 0.0002]
  → 7/10 client dưới floor 0.02
  → 4/10 client nhận đúng 0.0000 (không có row nào)
```

Sau floor 0.02 + renormalize, 5/10 client bị đẩy lên cùng một mức tối thiểu →
mất phân biệt giữa chúng.

**Đánh giá:** đây là **đánh đổi có chủ ý**, không phải bug:

| Không floor | Có floor 0.02 |
|---|---|
| 4 client có 0 rows → không train được | Mọi client ≥ 1,800 rows |
| Skew cực đoan nhất có thể | Skew vẫn rất mạnh (spread 0.90) |
| Không dùng được cho FL | Dùng được, vẫn stress-test FedProx |

**Cần nêu trong Limitations:** *"At α=0.1, the 0.02 floor binds for 5 of 10
clients, so the realized skew is slightly less extreme than a raw Dirichlet draw
would produce. This is deliberate: a raw draw leaves 4 clients with zero rows,
which cannot be trained. The realized spread (0.0087–0.9123) still represents
severe label skew."*

### 5.7 Chiều 7 — Reproducibility

**Cách làm:** chạy partitioner hai lần với cùng seed, so sánh assignment array.

**Kết quả:** `np.testing.assert_array_equal` pass cho cả IID và Dirichlet
(`tests/test_05_partition.py`). Cùng seed 42 → cùng partition bit-for-bit.

### 5.8 Bảng tổng hợp

| # | Chiều | Ngưỡng | IID | α=0.5 | α=0.1 |
|---|---|---|---|---|---|
| 1 | Completeness | 900,000 rows | ✅ | ✅ | ✅ |
| 2 | Disjointness | 0 overlap | ✅ | ✅ | ✅ |
| 3 | Schema | 31 cột, 0 leakage | ✅ | ✅ | ✅ |
| 4 | Class presence | ≥ 20 mỗi class | ✅ | ✅ | ✅ |
| 5 | Size floor | ≥ 1,800 rows | ✅ | ✅ | ✅ |
| 6 | Skew fidelity | spread tăng khi α giảm | ✅ 0.005 | ✅ 0.857 | ✅ 0.904 |
| 7 | Reproducibility | cùng seed → cùng kết quả | ✅ | ✅ | ✅ |

**RESULT: PASS cả 7 chiều.**

### 5.9 Parity pandas vs Spark

Cả hai đường (pandas `s05_partition.py` và Spark `s05_partition_spark.py`) dùng
**chung** hàm `assign_iid` / `assign_dirichlet`, nên kết quả khớp tuyệt đối:

| Strategy | pandas sizes | Spark sizes | Khớp |
|---|---|---|---|
| IID | 90,000 × 10 | 90,000 × 10 | ✅ |
| α=0.5 | 31,445 … 356,622 | 31,445 … 356,622 | ✅ |
| α=0.1 | 15,955 … 531,450 | 15,955 … 531,450 | ✅ |

Churn rates cũng khớp đến 6 chữ số thập phân.

---

## 6. Artifacts

| Path | Nội dung |
|---|---|
| `data/synthetic/partitions_iid/client_{0..9}.parquet` | 10 × 90,000 rows |
| `data/synthetic/partitions_label_skew_a0.5/client_{0..9}.parquet` | 10 silo, skew vừa |
| `data/synthetic/partitions_label_skew_a0.1/client_{0..9}.parquet` | 10 silo, skew mạnh |
| `results/partition_metadata.json` | size + churn rate per client |
| `results/silo_verification.json` | kết quả gate |

Mỗi parquet: `Customer ID` (SYN-0000001…) + 29 features + `churn` = **31 cột**.

---

## 7. Bàn giao cho Layer 2

| Cho | Artifact |
|---|---|
| FL client | `partitions_{tag}/client_{i}.parquet` |
| MLflow tags | `partition`, `alpha` từ `partition_metadata.json` |
| E2 (partition impact) | so sánh 3 strategies |
| E3 (aggregation) | dùng `label_skew_a0.1` |
| E5.1 (fairness) | per-client AUC trên từng silo |

---

## 8. Tái lập

```bash
python -m src.partitioning.s05_partition      # sinh 3 bộ partition
python -m src.partitioning.s06_verify_silos   # gate
pytest tests/test_05_partition.py -v          # 20 tests
```

---

## 9. Ghi chú cho báo cáo

- **α=0.1 có một client chiếm 59% dữ liệu.** Đây là hệ quả tự nhiên của
  Dirichlet với α nhỏ, không phải lỗi. Cần nêu trong §7 Limitations rằng
  FedAvg sẽ bị chi phối bởi client lớn — đúng như H4 dự đoán.
- **Client 5 (α=0.1) có 0.87% churn** — gần như single-class. Đây là stress test
  tốt nhất cho FedProx: nếu proximal term hoạt động, client này không được kéo
  global model lệch.
- **IID std = 0.0018** chứng minh partitioner đúng — sai lệch chỉ là nhiễu
  binomial với n = 90,000.
- **Floor 0.02 binding ở 5/10 client khi α=0.1** — cần nêu trong Limitations
  rằng realized skew nhẹ hơn raw Dirichlet draw, và giải thích đây là đánh đổi
  có chủ ý (raw draw để 4 client 0 rows).
