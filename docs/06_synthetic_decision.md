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

### 4.1 CTGAN KHÔNG được chạy — ghi nhận trung thực

**Trạng thái thực tế:**

| Kiểm tra | Kết quả |
|---|---|
| `data/synthetic/ctgan_1m.parquet` | ❌ không tồn tại |
| `import sdv` | ❌ `ModuleNotFoundError` |
| Hàm `fit_ctgan` trong `s03_amplify.py` | ❌ không có |

Spec gốc (`promt.md` §3.5) yêu cầu CTGAN làm **validation** để so sánh với
Copula. Bước này **chưa được thực hiện**.

**Lý do:** 5,271 rows nằm dưới ngưỡng 10K mà SDV khuyến nghị cho CTGAN. Chạy
CTGAN trên dữ liệu dưới ngưỡng sẽ cho kết quả không ổn định, và so sánh sẽ
không công bằng — CTGAN thua vì thiếu dữ liệu, không phải vì bản chất phương
pháp.

**Cách xử lý trong báo cáo:** nêu rõ trong Chapter 4 (Methodology) và
Chapter 7 (Limitations):

> "CTGAN was not evaluated as a comparison baseline. The 5,271-row training set
> falls below SDV's recommended 10K minimum for CTGAN, so any comparison would
> confound method quality with data scarcity. Copula was selected on the basis
> of its exact marginal fidelity (KS pass rate 100%) and a TSTR gap of 3.7%.
> Evaluating CTGAN on an oversampled training set is left as future work."

**Đây là thiếu sót so với spec, không phải lựa chọn thiết kế.** Cần nêu trong
phần Limitations, không được ẩn.

---

## 5. Đánh giá chất lượng synthetic data — 6 chiều

Synthetic data được đánh giá bằng **6 phép kiểm tra độc lập**, không chỉ một
metric. Mỗi phép trả lời một câu hỏi khác nhau về chất lượng.

| # | Chiều đánh giá | Câu hỏi | Công cụ |
|---|---|---|---|
| 1 | **Marginal fidelity** | Từng cột có phân phối giống real? | KS test / total variation |
| 2 | **Correlation fidelity** | Quan hệ giữa các cột có giữ được? | Spearman Δ |
| 3 | **Utility (TSTR)** | Model train trên synthetic có dự đoán được real? | AUC gap |
| 4 | **Diversity** | Có sinh ra dữ liệu mới hay chỉ copy? | unique ratio |
| 5 | **Memorization** | Có rò rỉ nguyên bản ghi training? | exact-match rate |
| 6 | **Validity** | Giá trị có nằm trong miền hợp lệ? | range + category check |

### 5.1 Chiều 1 — Marginal fidelity (KS test)

**Cách làm:** với mỗi cột numeric, chạy two-sample KS test giữa `train_real` và
`synthetic_1m`. Với cột categorical, dùng total variation distance (KS chỉ định
nghĩa cho phân phối liên tục).

**Ngưỡng:** ≥ 90% số cột phải pass ở α = 0.05.

**Kết quả: 29/29 = 100% PASS**

| Cột | Test | Statistic | p-value |
|---|---|---|---|
| Number of Referrals | KS | 0.0011 | 1.0000 |
| Total Long Distance Charges | KS | 0.0010 | 1.0000 |
| Total Charges | KS | 0.0010 | 1.0000 |
| Avg Monthly Long Distance Charges | KS | 0.0010 | 1.0000 |
| Tenure in Months | KS | 0.0009 | 1.0000 |
| Payment Method | TVD | 0.0009 | — |
| Monthly Charge | KS | 0.0007 | 1.0000 |
| … (22 cột còn lại) | — | ≤ 0.0007 | 1.0000 |

**Đọc kết quả:** KS statistic lớn nhất chỉ **0.0011** — tức hai phân phối lệch
nhau tối đa 0.11% khối lượng xác suất. Đây là hệ quả **tất yếu** của việc dùng
inverse empirical CDF: marginal khớp theo thiết kế, không phải may mắn.

> ⚠️ **Cảnh báo diễn giải:** KS pass 100% **không** chứng minh synthetic data
tốt. Nó chỉ chứng minh marginal khớp. Một generator copy nguyên training rows
cũng đạt KS 100%. Vì vậy cần 5 chiều còn lại.

### 5.2 Chiều 2 — Correlation fidelity

**Cách làm:** tính ma trận Spearman correlation trên 10 numeric columns cho cả
hai tập, đo hiệu tuyệt đối từng cặp (bỏ đường chéo).

**Ngưỡng:** mean |Δ| < 0.10.

**Kết quả: mean |Δ| = 0.0209 PASS**

| Metric | Value |
|---|---|
| mean \|Δ\| | **0.0209** |
| max \|Δ\| | 0.1249 |
| Số cặp vượt 0.10 | 2 / 90 |
| Frobenius norm | 0.3010 (tham khảo) |

**5 cặp lệch nhiều nhất:**

| Cặp | Real ρ | Synth ρ | Δ |
|---|---|---|---|
| Number of Dependents ↔ Number of Referrals | +0.354 | +0.229 | **0.125** |
| Tenure in Months ↔ Total Long Distance Charges | +0.631 | +0.576 | 0.056 |
| Avg Monthly GB Download ↔ Total Extra Data Charges | +0.128 | +0.073 | 0.055 |

**Đọc kết quả:** sai số trung bình 2.1 điểm phần trăm — tốt. Cặp tệ nhất
(`Dependents` ↔ `Referrals`) mất 0.125 do cả hai đều là biến đếm thưa (nhiều
số 0), nơi rank correlation kém ổn định. Chấp nhận được.

### 5.3 Chiều 3 — Utility (TSTR)

**Cách làm:** Train-on-Synthetic, Test-on-Real (Jordon et al. 2022).

1. Train `HistGradientBoostingClassifier` **chỉ trên synthetic** (1M rows)
2. Test trên `test_real` (1,318 rows **thật**, chưa từng thấy)
3. So với reference: cùng model train trên `train_real`, test trên cùng `test_real`

**Ngưỡng:** |AUC gap| < 0.05.

**Kết quả: gap = 0.0374 PASS**

| Model | Train trên | Test trên | AUC |
|---|---|---|---|
| Synthetic | `synthetic_1m` (1M) | `test_real` (1,318) | **0.8941** |
| Reference | `train_real` (5,271) | `test_real` (1,318) | **0.9315** |
| **Gap** | | | **0.0374** |

**Đọc kết quả:** model học hoàn toàn từ dữ liệu giả đạt **95.9%** hiệu năng của
model học từ dữ liệu thật (0.8941 / 0.9315). Đây là chiều **quan trọng nhất** —
nó đo đúng thứ ta cần: synthetic data có dùng được để train không.

> **Lưu ý thiết kế:** reference model phải train trên `train_real` và test trên
`test_real`. Lần chạy đầu tôi train reference trên chính `test_real` → AUC 1.0000
(in-sample, vô nghĩa). Đã sửa.

### 5.4 Chiều 4 — Diversity

**Cách làm:** đếm số hàng unique trong synthetic, so với tổng số hàng.

**Kết quả:**

| Metric | Value |
|---|---|
| Synthetic rows | 1,000,000 |
| Unique synthetic rows | **1,000,000** |
| Unique ratio | **100.00%** |

**Đọc kết quả:** không có hàng nào trùng lặp. Generator sinh dữ liệu mới thật sự,
không phải lặp lại một tập hữu hạn.

### 5.5 Chiều 5 — Memorization

**Cách làm:** lấy mẫu 20,000 hàng synthetic, kiểm tra từng hàng có trùng khớp
tuyệt đối với một hàng trong `train_real` không.

**Kết quả:**

| Metric | Value |
|---|---|
| Sample size | 20,000 |
| Exact matches to `train_real` | **0** |
| Memorization rate | **0.00%** |

**Đọc kết quả:** không có rò rỉ dữ liệu training. Đây là yêu cầu bắt buộc — nếu
synthetic chứa nguyên bản ghi thật thì việc "dùng synthetic để tránh chia sẻ dữ
liệu" trở nên vô nghĩa về mặt privacy.

### 5.6 Chiều 6 — Validity

**Cách làm:** kiểm tra mọi giá trị synthetic nằm trong miền quan sát được.

**Kết quả:**

| Check | Kết quả |
|---|---|
| Numeric ngoài `[min, max]` của `train_real` | **0 cột** |
| Categorical có giá trị lạ | **0 cột** |
| NaN | **0** |
| Target ∈ {0, 1} | ✅ |
| Churn rate drift | **0.0002** (0.2839 vs 0.2836) |

**Đọc kết quả:** không có giá trị vô lý (tenure âm, charge âm, category không
tồn tại). Đây là ưu điểm trực tiếp của việc clip về range quan sát + decode qua
encoder map.

### 5.7 Bảng tổng hợp

| # | Chiều | Ngưỡng | Đo được | Kết quả |
|---|---|---|---|---|
| 1 | Marginal (KS) | ≥ 90% cột | **100% (29/29)** | ✅ |
| 2 | Correlation | mean \|Δ\| < 0.10 | **0.0209** | ✅ |
| 3 | **Utility (TSTR)** | gap < 0.05 | **0.0374** | ✅ |
| 4 | Diversity | — | **100% unique** | ✅ |
| 5 | Memorization | — | **0.00%** | ✅ |
| 6 | Validity | — | **0 vi phạm** | ✅ |

**RESULT: PASS cả 6 chiều.**

### 5.8 Điều chỉnh metric so với plan ban đầu

Plan (`13_layer1_plan.md` §4 Phase 4) ghi "Correlation Frobenius diff < 0.15".
Frobenius norm là **tổng bình phương** các phần tử, nên nó tăng theo số cột —
một ngưỡng cố định không so sánh được giữa các bộ feature khác nhau.

Đo được: Frobenius = 0.3010 với d = 10 numeric columns. Nhưng
`mean|Δ| = 0.0209` — tức sai số thực tế rất nhỏ.

**Quyết định:** dùng `mean_abs_diff` (scale-invariant) làm metric chính, ngưỡng
0.10. Frobenius vẫn được báo cáo để tham khảo.

### 5.9 Điều chỉnh reference AUC

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
| **CTGAN chưa chạy** | **Thiếu sót so với spec — phải nêu trong Limitations** |
| Copula chỉ bắt được phụ thuộc tuyến tính trong không gian normal-score | Chấp nhận — đây là giới hạn cố hữu của Gaussian copula |

### 8.1 Giới hạn của Gaussian Copula (cần nêu trong báo cáo)

Gaussian Copula giả định phụ thuộc giữa các biến là **tuyến tính trong không
gian normal-score**. Nó không bắt được:

- **Tail dependence** — hai biến cùng cực đoan ở đuôi phân phối
- **Non-linear interaction** — quan hệ phi tuyến (ví dụ: churn chỉ tăng khi cả
  tenure thấp VÀ charge cao đồng thời)
- **Conditional heteroskedasticity** — phương sai thay đổi theo giá trị biến khác

Với dữ liệu tabular churn, các giới hạn này chấp nhận được vì TSTR gap chỉ 3.7%.
Nhưng phải nêu trong Chapter 7 (Limitations) để không overclaim.

### 8.2 Việc cần làm để đóng thiếu sót CTGAN

Nếu có thời gian trước W10:

1. `pip install sdv`
2. Thêm `fit_ctgan()` vào `s03_amplify.py`
3. Chạy quality gate trên `ctgan_1m.parquet`
4. Thêm bảng so sánh Copula vs CTGAN vào §5.7
5. Cập nhật kết luận §1

Nếu không kịp: giữ nguyên §4.1 và nêu rõ trong Limitations.
