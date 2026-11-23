# 10 — Experiment Protocol

**Owner:** Person B · **Tuần:** W5–W8 · **Trạng thái:** 🚧 Đang thực hiện
**Tham chiếu:** `12_architecture.md` §4 · `09_fl_architecture.md`

---

## 1. Nguyên tắc chung

### 1.1 Test set — chỉ `test_real`

| Test set | Rows | Dùng cho | KHÔNG dùng cho |
|---|---|---|---|
| **`test_real`** | **1,318** (374 churned) | **E0–E5.2** — mọi kết quả chính | — |
| `synth_test` | 100,000 | E6 (latency), E7 (sanity) | Abstract, conclusion |

**Lý do:** `test_real` là 20% thật, held out **trước** khi amplify. Synthetic test chỉ đo in-distribution fit, không đo generalization.

### 1.2 Protocol tiered

| Experiment | Test set | Protocol | Reporting |
|---|---|---|---|
| E0 | `test_real` | 1 lần | AUC + bootstrap 95% CI |
| **E1** | `test_real` | **5-fold CV** | **mean ± std + bootstrap CI** |
| E2 | `test_real` | fold 0 | AUC + bootstrap CI |
| E3 | `test_real` | fold 0 | AUC + per-client std + CI |
| E4 | `test_real` | fold 0 | AUC vs ε + CI |
| E5.1 | `test_real` | fold 0 | per-client AUC distribution |
| E5.2 | `test_real` + `eda_7043` | fold 0 | per-stratum AUC + caveat |
| E6 | `synth_test` | fold 0 | latency, AUC consistency |
| E7 | `test_real` (primary) + `synth_test` (sanity) | fold 0 | bytes, AUC |

**Lý do tiered:** 5-fold cho toàn bộ cần ~45 FL runs (2–4 giờ) và làm strata của E5.2 co lại còn ~75 churned/stratum.

---

## 2. Quyết định scaler — điểm dễ sai nhất

### 2.1 Vấn đề

Mỗi client scale bằng `(mean_k, std_k)` riêng (bắt buộc — dùng global stats là vi phạm privacy). Global model $W = \sum_k \frac{n_k}{n} W_k$ là trung bình của các weights học trên **các scale khác nhau**.

Vậy khi evaluate $W$ trên `test_real`, dùng scale nào?

### 2.2 Quyết định: `train_real` scaler

| Model | Train scale | Eval scale | Lý do |
|---|---|---|---|
| **Centralized** | `train_real` scaler | `train_real` scaler | Like-for-like |
| **FL global** | per-client scaler | **`train_real` scaler** | Canonical scale |
| **Local-only k** | client k scaler | **client k scaler** | Model k expects scale k |

**Cơ sở:**
- `test_real` là held-out từ cùng phân phối với `train_real`
- Centralized baseline dùng cùng scaler → so sánh E1 là like-for-like
- Đây là xấp xỉ **có chủ ý**, không phải lựa chọn hoàn hảo

**Caveat phải ghi trong báo cáo:**
> "The FL global model is evaluated with a scaler fitted on `train_real`, not on any client's local statistics. The global weights are an average of models trained on different local scales, so no single scale is exactly correct. Using the canonical `train_real` scale makes the comparison with the centralized baseline like-for-like, at the cost of a small mismatch between the scale the weights were trained on and the scale they are evaluated with."

### 2.3 Tại sao local-only dùng scaler riêng

Model của client $k$ chỉ từng thấy dữ liệu scale bằng `(mean_k, std_k)`. Evaluate bằng scale khác sẽ cho kết quả sai — model chưa từng thấy phân phối đó.

---

## 3. E1 — FL gap

### 3.1 Câu hỏi

**RQ1 / H1:** FL đạt bao nhiêu % hiệu năng centralized khi không bao giờ thấy dữ liệu ở một chỗ?

### 3.2 Ba model

| Model | Train trên | Rows | Vai trò |
|---|---|---|---|
| **Centralized** | `synth_train` | 900,000 | Upper bound |
| **Federated** | 10 silo, FedAvg | 900,000 tổng | Subject |
| **Local-only** | từng silo riêng | 90,000/client | Lower bound |

**Công bằng:** cả ba dùng cùng architecture (MLP GroupNorm), cùng optimizer (Adam, batch 256, lr 1e-3), cùng tổng epoch (50).

### 3.3 Fold protocol

Mỗi fold re-split `clean_7043` với seed khác, rồi re-amplify và re-partition.

**Không thể tái sử dụng synthetic data giữa các fold:**

```
Fold 0: train_real = rows {1..5271}, test_real = rows {5272..6589}
        Copula học từ rows {1..5271}

Fold 1: train_real = rows {1..1054, 2108..6589}, test_real = rows {1055..2107}
        Nếu dùng lại synthetic của fold 0:
        → Copula đã thấy rows {1055..2107} (chúng nằm trong fold 0's train_real)
        → test_real của fold 1 ĐÃ BỊ RÒ RỈ
```

| Fold | Seed | Artifacts |
|---|---|---|
| 0 | 42 | Tái sử dụng artifacts hiện có |
| 1 | 43 | `data/processed/fold_1/`, `data/synthetic/fold_1/` |
| 2 | 44 | `fold_2/` |
| 3 | 45 | `fold_3/` |
| 4 | 46 | `fold_4/` |

### 3.4 Chi phí

| Bước | Thời gian |
|---|---|
| Amplify 1M | 14s |
| Split + partition | 10s |
| FL 50 rounds × 5 client × 1.6s | 400s |
| Centralized 50 epoch × 900K | ~700s |
| Local-only 10 client × 50 epoch | ~800s |
| Eval + bootstrap | 5s |
| **Tổng/fold** | **~32 phút** |

**5 folds ≈ 2.7 giờ.** Chạy background.

### 3.5 Output

`results/E1_fl_gap.csv` + `results/E1_fl_gap.md`

| Model | AUC (mean ± std) | Gap vs centralized |
|---|---|---|
| centralized | 0.XXX ± 0.0YY | 0.00% |
| federated | 0.XXX ± 0.0YY | **Z.ZZ%** |
| local_only | 0.XXX ± 0.0YY | W.WW% |

**Success criterion #2:** FL gap < 5%.

---

## 4. E0 — External baseline

### 4.1 Câu hỏi

IBM `Churn Score` dùng thông tin post-hoc mà model của ta **không được phép** dùng. Nó đạt AUC bao nhiêu?

### 4.2 Đã đo

**AUC = 0.9421** (từ `notebooks/01_eda.ipynb` §3.5)

### 4.3 Reporting

> "IBM's precomputed Churn Score (which leverages post-hoc features) achieves AUC = 0.9421. Our centralized model without post-hoc features achieves AUC = Y. Our FL model achieves AUC = Z. Despite lacking post-hoc information, FL retains (Z/Y)×100% of centralized performance and (Z/0.9421)×100% of IBM's score."

---

## 5. E2 — Partition impact

### 5.1 Câu hỏi

**RQ1 / H1:** non-IID làm giảm AUC bao nhiêu?

### 5.2 Biến độc lập

| Partition | Churn range | std |
|---|---|---|
| IID | 0.2819–0.2872 | 0.0018 |
| α=0.5 | 0.0138–0.8703 | 0.3254 |
| α=0.1 | 0.0087–0.9123 | 0.2939 |

### 5.3 Protocol

Fold 0, FedAvg, 50 rounds, mỗi partition một lần chạy.

**Metrics:** AUC, số round để đạt 95% AUC cuối (convergence speed).

---

## 6. E3 — Aggregation robustness

### 6.1 Câu hỏi

**RQ2 / H2:** strategy nào robust nhất với label skew?

### 6.2 Biến độc lập

| Strategy | Cơ chế | Tham số |
|---|---|---|
| FedAvg | baseline | — |
| FedProx | `+ (μ/2)‖w − w^t‖²` | μ = 0.01 |
| FedAvgM | server momentum | β = 0.9 |

### 6.3 Protocol

Fold 0, partition α=0.1 (severe skew), 50 rounds, mỗi strategy một lần.

**Metrics:** AUC, per-client AUC std (fairness).

---

## 7. E4 — Privacy-utility

### 7.1 Câu hỏi

**RQ3 / H3:** DP-SGD với ε ∈ [1, 10] giảm AUC bao nhiêu?

### 7.2 Biến độc lập

| `noise_multiplier` | ε (kỳ vọng) |
|---|---|
| 0 | ∞ (no privacy) |
| 0.5 | must measure |
| 1.0 | must measure |
| 2.0 | must measure |

`max_grad_norm = 1.0`, `delta = 1e-5`.

### 7.3 Yêu cầu kỹ thuật

**GroupNorm bắt buộc** — Opacus không wrap được BatchNorm. Đã đổi ở `configs/base.yaml`.

### 7.4 Reporting

Đường cong AUC vs ε. **Phải ghi rõ ε báo cáo theo nghĩa nào:**
- Per-client per-round?
- Composed qua 50 rounds?

**Đề xuất:** báo cáo **composed ε sau 50 rounds** cho mỗi client, kèm ghi chú.

---

## 8. E5 — Fairness

### 8.1 E5.1 — Per-client

**RQ4 / H4:** FL có gây unfairness giữa các client?

**Metrics:** per-client AUC → mean, std, **min** (worst-client), boxplot.

**So sánh:** FedAvg vs FedProx.

### 8.2 E5.2 — Recall theo lý do churn

**Câu hỏi:** FedProx có giảm unfairness **giữa các lý do churn** tốt hơn FedAvg?

**Cách làm:**
1. Join `test_real` với `eda_7043.parquet` trên `Customer ID`
2. Lấy `Churn Category` (5 lý do: Price / Competitor / Dissatisfaction / Attitude / Other)
3. Tính **recall@top-10%** riêng từng lý do
4. Grouped bar chart

**⚠️ Vì sao KHÔNG dùng AUC (sửa so với bản protocol cũ):**

`Churn Category` chỉ được điền cho khách **đã churn**. Vì vậy mỗi stratum lý do là
**100% positive** — không có lớp âm để xếp hạng, AUC không xác định được. Bản
protocol đầu tiên ghi "tính AUC riêng từng stratum" là **sai về mặt toán học**;
lần chạy đầu tiên trả về `stratum AUC std = nan` đúng như dự đoán.

Số liệu thực tế trên `test_real` (1,318 dòng):

| Stratum | Số dòng | Churned |
|---|---|---|
| `None` (không churn) | 944 | 0 |
| `Competitor` | 170 | 170 |
| `Attitude` | 58 | 58 |
| `Dissatisfaction` | 56 | 56 |
| `Price` | 50 | 50 |
| `Other` | 40 | 40 |

**Metric thay thế:** recall@top-10% — *"trong số khách rời đi vì lý do R, bao
nhiêu % được model gắn cờ?"*

**Điểm kỹ thuật quan trọng:** ngưỡng top-10% tính **một lần trên toàn bộ test
set**, KHÔNG tính riêng từng stratum. Nếu tính riêng, mỗi stratum luôn bị gắn cờ
đúng 10% theo thiết kế → che mất chính sự chênh lệch cần đo.

**Caveat bắt buộc:**
> "With 374 churned customers spread across 5 reasons, each reason holds 40-170
> positives. Recall on 50 positives has a 95% CI of roughly ±0.13, so differences
> smaller than that between strategies should not be interpreted as meaningful.
> A PASS on this criterion means the alternative strategy reduced dispersion on
> at least one fairness view; it does not mean the model is fair."

---

## 9. E6 — Scalability

### 9.1 Câu hỏi

**RQ5:** FL có scale được khi K tăng từ 5 lên 50?

### 9.2 Biến độc lập

K ∈ {5, 10, 20, 50}

### 9.3 Test set

**`synth_test`** (100,000 rows) — đây là experiment duy nhất dùng synthetic test làm primary, vì mục đích là đo **latency**, không phải generalization.

### 9.4 Yêu cầu kỹ thuật

**Ray bắt buộc** cho K=50 (tuần tự ~40 phút → Ray ~10 phút).

```bash
python -m src.fl.s11_server --ray --clients 50 --rounds 50
```

---

## 10. E7 — Communication cost

### 10.1 Câu hỏi

**RQ6:** chi phí communication có khả thi thực tế?

### 10.2 Biến độc lập

`fraction_sampled` ∈ {0.25, 0.5, 1.0}

### 10.3 Metrics

- **Bytes/round** = `2 × |θ| × 4 bytes × n_clients_selected`
  - `|θ|` = 18,049 parameters
  - Hệ số 2 = upload + download
- **Cumulative bytes** qua 50 rounds
- AUC trên `test_real` (primary) + `synth_test` (sanity)

### 10.4 Kỳ vọng

| fraction | Clients/round | Bytes/round |
|---|---|---|
| 0.25 | 2–3 | ~0.3 MB |
| 0.5 | 5 | ~0.7 MB |
| 1.0 | 10 | ~1.4 MB |

---

## 11. MLflow tags bắt buộc

Mọi run phải log:

```python
{
    "experiment": "E1",
    "partition": "iid",
    "alpha": "n/a",
    "strategy": "fedavg",
    "K": 10,
    "noise_multiplier": 0.0,
}
```

**Fallback:** nếu MLflow server tắt → `results/runs/{run_id}.json`. Không được crash.

---

## 12. Tái lập

```bash
# E1 fold 0
python -m src.experiments.s15_e1_fl_gap --fold-ids 0 --rounds 50

# E1 folds 1-4 (background)
python -m src.experiments.s15_e1_fl_gap --fold-ids 1,2,3,4 --rounds 50 &

# Gộp
python -m src.experiments.s15_e1_fl_gap --fold-ids 0,1,2,3,4 --aggregate

# Baselines riêng lẻ
python -m src.baselines.s13_centralized --epochs 50
python -m src.baselines.s13_local_only --partition iid --epochs 50
```
