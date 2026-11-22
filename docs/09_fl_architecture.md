# 09 — FL Architecture

**Owner:** Person B · **Tuần:** W4 · **Trạng thái:** ✅ Milestone 1 round đạt
**Tham chiếu:** `12_architecture.md` §1, §3 · `03_reading_notes_dp.md`

---

## 1. Thành phần

| Stage | File | Vai trò |
|---|---|---|
| 07 | `src/features/s07_preprocess.py` | One-hot + scaler cục bộ |
| 08 | `src/models/s08_mlp.py` | MLP với GroupNorm |
| 09 | `src/fl/s09_trainer.py` | Training loop + FedProx hook |
| 10 | `src/fl/s10_client.py` | Flower `NumPyClient` wrapper |
| 11 | `src/fl/s11_server.py` | FedAvg / FedProx / FedAvgM |

---

## 2. Quyết định: GroupNorm thay BatchNorm

`configs/base.yaml` đổi từ `batch_norm: true` sang `norm: groupnorm`.

### 2.1 Ba lý do, mỗi lý do đủ để quyết định

**Lý do 1 — Opacus không wrap được BatchNorm (blocker cứng)**

DP-SGD cần per-sample gradient:

$$\tilde{g}_i = \frac{g_i}{\max(1, \|g_i\|_2 / C)}$$

Nhưng với BatchNorm, output của sample $i$ phụ thuộc vào **toàn bộ batch** qua
$\mu_B$:

$$\ell(w, x_i) = f\left(x_i, \mu_B(x_1,\dots,x_B)\right)$$

→ $g_i$ không xác định độc lập được. Opacus raise `UnsupportedModuleError`.
E4 (privacy-utility) **không thể chạy** nếu giữ BatchNorm.

**Lý do 2 — So sánh công bằng E1–E7**

Nếu E1–E3 dùng BatchNorm còn E4 dùng GroupNorm:

```
E1 AUC = 0.89  (BatchNorm)
E4 AUC = 0.85  (GroupNorm + DP)
```

Chênh 0.04 nhưng **không biết bao nhiêu do DP, bao nhiêu do đổi kiến trúc**.
Kết luận H3 mất giá trị vì confounded.

**Lý do 3 — BatchNorm hỏng dưới non-IID FedAvg**

`state_dict()` của PyTorch bao gồm buffer `running_mean`, `running_var`. FedAvg
trung bình **cả buffer**:

$$W_{\text{global}} = \sum_k \frac{n_k}{n} W_k \quad\text{(bao gồm buffers)}$$

Với partition α=0.1 của dự án:

| Client | Churn rate | `running_mean` của Tenure |
|---|---|---|
| Client 5 | 0.87% | ~40 tháng (khách ở lâu) |
| Client 8 | 91.2% | ~10 tháng (khách mới bỏ) |

FedAvg trung bình → `running_mean` ≈ 25 tháng — **không khớp client nào**.
Global model có normalization statistics vô nghĩa.

GroupNorm **không có buffer** → không có vấn đề này.

Tham khảo: Li et al. (2021), *FedBN: Federated Learning on Non-IID Features via
Local Batch Normalization*.

### 2.2 Kiến trúc

```
Input (D=56) → Linear(56,128) → GroupNorm(8,128) → ReLU → Dropout(0.3)
             → Linear(128,64) → GroupNorm(8,64)  → ReLU → Dropout(0.3)
             → Linear(64,32)  → ReLU
             → Linear(32,1)   → Sigmoid
```

**Parameters:** 18,049 · **Weight tensors:** 12 · **Bytes/round:** 72,196

---

## 3. Preprocessing — bất biến privacy

### 3.1 Scaler cục bộ

Mỗi client fit scaler trên **silo của chính nó**:

```python
mean = X_client.mean(axis=0)
std  = X_client.std(axis=0)
```

**Không** dùng global mean/std — đó là rò rỉ thông tin giữa các silo.

Hệ quả: cùng một feature có scale khác nhau ở mỗi silo. Đây là **đúng**: model
học weights phù hợp với biểu diễn đã chuẩn hoá, và FedAvg trung bình **weights**,
không phải dữ liệu thô.

### 3.2 Vocabulary chia sẻ

One-hot width phụ thuộc category nào xuất hiện trong silo. Nếu silo A không có
`Offer = "None"` thì ma trận hẹp hơn silo B → FedAvg không average được tensor
khác shape.

**Giải pháp:** fit vocabulary trên **union** category của tất cả silo, áp dụng
cố định cho mọi silo.

Vocabulary là **metadata** (tên category nào tồn tại), không phải **data** (bao
nhiêu khách hàng có category đó) → chia sẻ không rò rỉ gì.

### 3.3 Kết quả đo được

| Metric | Value |
|---|---|
| Categorical columns | 19 |
| Numeric columns | 10 |
| **Input dim** | **56** |
| X shape (client 0) | (90,000, 56) |
| NaN | 0 |

---

## 4. Local training

| Tham số | Giá trị |
|---|---|
| Optimizer | Adam |
| Loss | BCE |
| `local_epochs` | 1 |
| `batch_size` | 256 |
| `learning_rate` | 1e-3 |

### 4.1 FedProx hook

```python
loss = BCE(proba, y) + (mu/2) * Σ ||w - w_global||²
```

`mu = 0.01` khi strategy là `fedprox`, `0.0` cho FedAvg/FedAvgM.

### 4.2 Kết quả smoke test (client 0, IID)

| Metric | Before | After 1 epoch |
|---|---|---|
| Loss | 0.7031 | **0.5172** |
| AUC | 0.5318 | **0.7828** |
| Duration | — | 4.85s |

---

## 5. Server — quyết định defer Ray

### 5.1 Ray là gì

Framework distributed computing. Trong dự án dùng để chạy client training
**song song** thay vì tuần tự.

| | Tuần tự | Ray |
|---|---|---|
| 5 client/round | 10s | ~2.5s |
| 50 rounds | ~8 phút | ~2 phút |

### 5.2 Tại sao defer

**Lý do 1 — Milestone đầu cần debug dễ, không cần nhanh**

Với Ray, một `KeyError` trong `fit()` hiện ra dưới dạng `RayTaskError` với
stack trace phân mảnh qua nhiều process. Tuần tự cho traceback Python bình thường.

**Lý do 2 — 8 phút cho 50 rounds là chấp nhận được**

W5–W8 chạy ~20–30 runs → 3–4 giờ. Không lý tưởng nhưng khả thi.

**Lý do 3 — Ray chỉ thực sự cần ở E6**

| Experiment | K | Tuần tự | Cần Ray? |
|---|---|---|---|
| E1–E5, E7 | 10 | ~8 min | Không |
| **E6** | **50** | **~40 min** | **Có** |

**Lý do 4 — Flower 1.44 yêu cầu Ray cho `start_simulation`**

`fl.simulation.start_simulation` raise `ImportError: Unable to import module ray`.
Nên ta implement FedAvg loop trực tiếp trong `run_sequential()` — logic giống hệt
Flower's `FedAvg` strategy.

### 5.3 Hai entry point

| Function | Dùng khi | Cần Ray |
|---|---|---|
| `run_sequential()` | E1–E5, E7, debug | ❌ |
| `run_simulation()` | E6 (K=50) | ✅ |

CLI: `--ray` flag chọn `run_simulation`.

---

## 6. Milestone: 1 round end-to-end

### 6.1 Lệnh

```bash
python -m src.fl.s11_server --rounds 1 --clients 10 --strategy fedavg --partition iid
```

### 6.2 Kết quả

```
Sequential simulation: strategy=fedavg partition=iid rounds=1 clients=10
                       selected_per_round=5

Client 0 fit: round=1 loss=0.5172 auc=0.7828
Client 3 fit: round=1 loss=0.5159 auc=0.7844
Client 4 fit: round=1 loss=0.5155 auc=0.7864
Client 5 fit: round=1 loss=0.5139 auc=0.7862
Client 7 fit: round=1 loss=0.5174 auc=0.7836

Round 1/1: clients=[0, 3, 4, 5, 7] loss=0.5160 auc=0.7847
```

### 6.3 Verification

| Check | Kỳ vọng | Đo được |
|---|---|---|
| Server khởi động | OK | ✅ |
| 10 client load | OK | ✅ |
| `fraction_fit=0.5` | 5/10 client | ✅ `[0,3,4,5,7]` |
| Weighted aggregation | `Σ(n_i/n)·W_i` | ✅ |
| Global weights thay đổi | Có | ✅ |
| AUC > random | > 0.5 | ✅ **0.7847** |

**Đây là success criterion #1 (mandatory):** *"FL runs end-to-end 50 rounds IID
without crash"* — bước đầu đã đạt, còn 50 rounds ở W5.

---

## 7. Tests

`tests/test_09_fl.py` — **32 tests**

| Nhóm | Số test | Nội dung |
|---|---|---|
| Vocabulary | 4 | Coverage, uniqueness, width, roundtrip |
| Encoding | 4 | Shape, dtype, no NaN, one-hot binary |
| Scaler | 4 | Standardise, zero-variance, roundtrip, **local-only** |
| Model | 7 | Shape, probability, **GroupNorm default**, **no BN buffers**, runtime dim |
| Weights | 2 | Roundtrip, wrong count |
| Trainer | 7 | Dim mismatch, length mismatch, loss improves, fields, proximal term |
| Client | 2 | fit returns n, evaluate returns AUC |
| Server | 3 | proximal_mu mapping, **1 round runs**, weights change |

**Tổng suite: 97 tests pass** (65 Layer 1 + 32 Layer 2).

---

## 8. Bàn giao cho W5

| Cho | Artifact |
|---|---|
| E1 (FL gap) | `run_sequential()` với 50 rounds |
| E2 (partition impact) | `--partition iid / label_skew_a0.5 / label_skew_a0.1` |
| E3 (aggregation) | `--strategy fedavg / fedprox / fedavgm` |
| E4 (privacy) | Cần thêm Opacus vào `s09_trainer.py` |
| E5.1 (fairness) | `client.evaluate()` per client |
| E6 (scalability) | `--ray` với K=50 |
| E7 (communication) | `result.bytes_sent` |

---

## 9. Tái lập

```bash
# Preprocess
python -m src.features.s07_preprocess

# Model smoke test
python -m src.models.s08_mlp

# Trainer smoke test
python -m src.fl.s09_trainer

# Client smoke test
python -m src.fl.s10_client

# 1 round end-to-end
python -m src.fl.s11_server --rounds 1 --clients 10 --strategy fedavg --partition iid

# Tests
pytest tests/test_09_fl.py -v
```
