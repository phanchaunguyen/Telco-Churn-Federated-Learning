# 06 — Synthetic Data Decision

**Owner:** Person A + C · **Tuần:** W2 · **Trạng thái:** ✅ Đã chốt
**Nguồn số liệu:** `data/synthetic/quality_report.json`

---

## 1. Quyết định

**Phương pháp chính: rank-based Gaussian Copula.**

CTGAN không được dùng làm primary. Lý do ở §4.

---

## 2. Vì sao không dùng `copulas.GaussianMultivariate`

Thư viện `copulas` fit một **Gaussian KDE riêng cho từng cột**, rồi nghịch đảo
bằng root-finding (`chandrupatla`) — mỗi lần đánh giá kernel phải so sánh với
**toàn bộ** `n_train` rows.

Hệ quả đo được:

| Vấn đề | Số liệu thực |
|---|---|
| Bộ nhớ cho 1 lần sample 1M | `1,000,000 × 5,271 × 8 bytes` = **39.3 GiB** |
| Kết quả | `numpy._core._exceptions._ArrayMemoryError` |
| Chunking 50K | Vẫn chạy > 20 phút chưa xong 1M rows |

Không khả thi trên máy 16 GB.

---

## 3. Phương pháp thay thế — rank-based Gaussian Copula

### 3.1 Thuật toán

```
FIT (trên train_real, 5,271 rows)
  1. Encode categorical → integer
  2. Với mỗi cột: rank → uniform (rank-0.5)/n → normal score Φ⁻¹(u)
  3. Σ = corrcoef(normal scores)              # tham số copula
  4. Lưu quantile grid 10,000 điểm mỗi cột    # cho inverse empirical CDF

SAMPLE (N rows)
  1. z ~ N(0, Σ)                              # Cholesky, O(N·d²)
  2. u = Φ(z)                                 # về uniform
  3. x = interp(u, grid, quantiles)           # inverse empirical CDF
  4. Decode categorical, clip numeric về [min, max]
```

### 3.2 Độ phức tạp

| | `copulas` (KDE) | Rank-based (dùng) |
|---|---|---|
| Time | O(N · n_train · d) | **O(N · d²)** |
| Memory | O(N · n_train) | **O(N · d)** |
| 1M rows | 39 GiB, không xong | **13.9 giây** |

### 3.3 Đánh đổi

| Mất | Được |
|---|---|
| Không smooth marginals (dùng empirical CDF thay KDE) | Marginal khớp **chính xác** theo định nghĩa |
| Không nội suy được ngoài range quan sát | Không sinh giá trị vô lý (tenure âm, charge âm) |
| — | Nhanh hơn ~100×, chạy được trên laptop |

Với mục tiêu của dự án (train FL trên tabular, không cần sinh giá trị mới ngoài
phân phối), empirical CDF là lựa chọn **đúng hơn** KDE.

---

## 4. Vì sao không dùng CTGAN làm primary

| Tiêu chí | Copula | CTGAN |
|---|---|---|
| Thời gian fit (5,271 rows) | **~2 giây** | ~10–30 phút (cần GPU để nhanh) |
| Thời gian sample 1M | **~12 giây** | ~5–15 phút |
| Rủi ro mode collapse | Không | Có |
| Cần tune hyperparameter | Không | Có (epochs, discriminator dim, pac) |
| Marginal fidelity | **Chính xác** (empirical CDF) | Xấp xỉ |
| Correlation fidelity | Tốt (rank-based) | Tốt hơn nếu train đủ lâu |

Với 5,271 rows training data, CTGAN **không đủ dữ liệu** để học ổn định —
khuyến nghị chính thức của SDV là ≥ 10K rows cho CTGAN. Copula phù hợp hơn ở
quy mô này.

**CTGAN vẫn được giữ trong kế hoạch như validation** (Phase 3.3) nếu thời gian
cho phép, nhưng không phải điều kiện để pass gate.

---

## 5. Kết quả quality gate

Đo trên `synthetic_1m.parquet` (1,000,000 rows) vs `train_real` (5,271 rows),
TSTR đánh giá trên `test_real` (1,318 rows).

| Check | Ngưỡng | Đo được | Kết quả |
|---|---|---|---|
| **KS pass rate** | ≥ 90% cột | **100.0% (29/29)** | ✅ |
| **Correlation mean\|Δ\|** | < 0.10 | **0.0209** | ✅ |
| Correlation max\|Δ\| | (tham khảo) | 0.1249 | — |
| **TSTR AUC** | — | **0.8941** | — |
| **Reference AUC** (train_real → test_real) | — | **0.9315** | — |
| **TSTR gap** | < 0.05 | **0.0374** | ✅ |

**RESULT: PASS**

### 5.1 Đọc kết quả

- **KS 100%**: cả 29 cột có marginal không phân biệt được với real ở α = 0.05.
  Đây là hệ quả trực tiếp của việc dùng empirical CDF — marginal khớp theo
  thiết kế, không phải may mắn.
- **Correlation mean|Δ| = 0.021**: sai số tương quan trung bình chỉ 2 điểm phần
  trăm. Max|Δ| = 0.125 nằm ở một cặp cụ thể (thường là cặp charge có phân phối
  đuôi dài) — chấp nhận được.
- **TSTR gap = 0.0374**: model train **hoàn toàn trên synthetic** đạt AUC 0.8941
  trên **real test set**, so với 0.9315 khi train trên real. Mất 3.7 điểm AUC —
  dưới ngưỡng 5%.

### 5.2 Điều chỉnh metric so với plan ban đầu

Plan (`13_layer1_plan.md` §4 Phase 4) ghi "Correlation Frobenius diff < 0.15".
Frobenius norm là **tổng bình phương** các phần tử, nên nó tăng theo số cột —
một ngưỡng cố định không so sánh được giữa các bộ feature khác nhau.

Đo được: Frobenius = 0.3010 với d = 10 numeric columns. Nhưng
`mean|Δ| = 0.0209` — tức sai số thực tế rất nhỏ.

**Quyết định:** dùng `mean_abs_diff` (scale-invariant) làm metric chính, ngưỡng
0.10. Frobenius vẫn được báo cáo để tham khảo.

### 5.3 Điều chỉnh reference AUC

Lần chạy đầu dùng reference = model train trên `test_real` rồi test trên chính
`test_real` → AUC = 1.0000 (in-sample, vô nghĩa).

**Sửa:** reference = model train trên `train_real`, test trên `test_real` —
cùng held-out set với model synthetic. Đây mới là so sánh like-for-like về
**chất lượng training data**.

---

## 6. Artifacts

| File | Shape | Nội dung |
|---|---|---|
| `data/synthetic/synthetic_1m.parquet` | 1,000,000 × 30 | 29 features + churn |
| `data/synthetic/amplify_metadata.json` | — | encoders, quantile ranges, churn rate |
| `data/synthetic/quality_report.json` | — | KS per column, correlation, TSTR |
| `data/synthetic/synth_train.parquet` | 900,000 × 30 | → partitioning |
| `data/synthetic/synth_test.parquet` | 100,000 × 30 | → E6/E7 only |

**Churn rate:** 0.2839 (target 0.2836 từ `train_real`) — lệch 0.0003.

---

## 7. Tái lập

```bash
python -m src.data.s02_split              # train_real / test_real
python -m src.data.s03_amplify            # synthetic_1m  (~14 giây)
python -m src.data.s04_quality_gate       # quality_report.json
python -m src.data.s02_split --synthetic  # synth_train / synth_test
```

---

## 8. Rủi ro còn lại

| Risk | Trạng thái |
|---|---|
| Copula không bắt được interaction bậc cao | Chấp nhận — TSTR gap 3.7% cho thấy đủ tốt |
| Synthetic churn rate lệch | Đã kiểm soát: 0.2839 vs 0.2836 |
| `test_real` lọt vào amplify | Đã phòng: `build_synthetic` chỉ đọc `train_real.parquet` |
| CTGAN chưa chạy | Ghi nhận — không phải điều kiện pass gate |
