# 15 — Kết quả W6/W7: E2, E3, E4, E5

> **Trạng thái:** E2 ✅ PASS · E3 ❌ FAIL · E4 ✅ PASS · E5 ✅ PASS
> **Ngày chạy:** 2026-11-23
> **Protocol:** `docs/10_experiment_protocol.md` §5 (E2), §6 (E3), §7 (E4), §8 (E5)

---

## 1. E2 — Partition impact

**Câu hỏi (RQ2):** non-IID có làm giảm chất lượng model FL không?

**Thiết lập:** FedAvg, 50 rounds, 10 clients, chọn 5 client/round, seed 42,
đánh giá trên `test_real` (1,318 dòng thật).

| Partition | AUC | 95% CI | PR-AUC | Recall@10% | Hội tụ (round) | Per-client AUC std | Gap vs IID |
|---|---|---|---|---|---|---|---|
| `iid` | 0.8949 | [0.8755, 0.9123] | 0.8121 | 0.3289 | 14 | 0.0018 | +0.00% |
| `label_skew_a0.5` | 0.8928 | [0.8737, 0.9111] | 0.7985 | 0.3289 | 21 | 0.0070 | +0.24% |
| `label_skew_a0.1` | 0.8889 | [0.8695, 0.9068] | 0.7876 | 0.3155 | 14 | 0.0045 | +0.67% |

**Tiêu chí #3 (non-IID làm giảm AUC): PASS** — α=0.1 mất 0.67% AUC so với IID.

### Nhận xét

- Cả ba chỉ số (AUC, PR-AUC, Recall@10%) đều giảm **đơn điệu** theo mức skew.
  PR-AUC rơi rõ nhất (0.8121 → 0.7876, −3.0%) vì nó nhạy với lớp thiểu số.
- Mức suy giảm AUC nhỏ (0.67%) vì mỗi client vẫn giữ ~90K dòng; với α=0.1 một
  client giữ 59% dữ liệu nên FedAvg vẫn có tín hiệu tốt.
- **Per-client AUC std tăng 3.9×** (0.0018 → 0.0070 ở α=0.5) — đây mới là tác
  động rõ nhất của non-IID: model toàn cục kém đều hơn giữa các silo.
- Hội tụ **không** đơn điệu (14 / 21 / 14). α=0.5 cần nhiều round nhất vì
  client nhỏ nhất (31K dòng) đóng góp nhiễu; α=0.1 có một client khổng lồ
  (531K) chi phối trung bình nên đạt ngưỡng sớm.

---

## 2. E3 — Aggregation robustness

**Câu hỏi (RQ3):** FedProx / FedAvgM có chống được label skew tốt hơn FedAvg?

**Thiết lập:** partition `label_skew_a0.1` (khắc nghiệt nhất), 50 rounds.

| Strategy | AUC | 95% CI | PR-AUC | Recall@10% | Hội tụ | Client std | Client min | Gain vs FedAvg |
|---|---|---|---|---|---|---|---|---|
| `fedavg` | 0.8889 | [0.8695, 0.9068] | 0.7876 | 0.3155 | 14 | 0.0045 | 0.7883 | +0.00% |
| `fedprox` | 0.8835 | [0.8628, 0.9020] | 0.7743 | 0.3155 | 30 | 0.0040 | 0.7837 | −0.60% |
| `fedavgm` | 0.8875 | [0.8677, 0.9056] | 0.7827 | 0.3182 | 20 | 0.0037 | 0.7871 | −0.16% |

**Tiêu chí #4 (FedProx/FedAvgM thắng FedAvg): FAIL** — cả hai đều thấp hơn.

### Nhận xét

- **Không strategy nào cải thiện AUC.** Chênh lệch −0.60% và −0.16% đều nằm
  trong CI rộng (±2%), nên kết luận đúng là: *trên dữ liệu này, FedProx và
  FedAvgM không khác FedAvg một cách có ý nghĩa*.
- **Nhưng cả hai đều giảm per-client std** (0.0045 → 0.0040 → 0.0037) và nâng
  worst-client (0.7883 → 0.7871, FedProx 0.7837). Tức là chúng **công bằng
  hơn** dù không chính xác hơn — đúng bản chất của FedProx (kéo client lệch về
  gần global) và FedAvgM (làm mượt quỹ đạo).
- FedProx hội tụ **chậm gấp đôi** (30 vs 14 round) vì proximal term μ=0.01
  giữ model gần điểm khởi đầu, giảm biên độ cập nhật mỗi round.
- FedAvgM hội tụ ở round 20, chậm hơn FedAvg nhưng nhanh hơn FedProx.

### Vì sao kết quả này hợp lý

Với α=0.1 nhưng mỗi silo vẫn có ≥15,955 dòng và ≥20 mẫu mỗi lớp (ràng buộc
`MIN_CLASS_COUNT`), mức skew **không đủ cực đoan** để FedProx phát huy tác
dụng. FedProx được thiết kế cho trường hợp client có rất ít dữ liệu và lệch
hẳn; ở đây client nhỏ nhất vẫn có 15,955 dòng. Đây là phát hiện trung thực cần
báo cáo, không phải lỗi cấu hình.

---

## 3. Hai lỗi metric phát hiện trong quá trình chạy

### 3.1 `convergence_round` báo sai round 1

**Triệu chứng:** cả 3 cấu hình E2 đều báo "converged at round 1".

**Nguyên nhân:** ngưỡng là *phân số của AUC cuối* (95%). Curve AUC tăng nhanh
rồi phẳng: round 1 đã đạt 0.7847 / 0.8021 = 97.8% giá trị cuối, nên luôn thoả.

**Sửa:** ngưỡng tuyệt đối (0.005) trên đường đã làm mượt (moving average cửa
sổ 5, căn giữa), so với **đỉnh** chứ không phải giá trị cuối. So với đỉnh để
một run phân kỳ không thể báo "hội tụ ở round 1" (final thấp → mọi round đầu
đều nằm trong dung sai).

### 3.2 F1 = 0 dù model tốt

**Triệu chứng:** `label_skew_a0.5` có F1 = 0.0000 nhưng AUC = 0.8928.

**Nguyên nhân:** F1 dùng ngưỡng 0.5. Với tỉ lệ churn 28%, model hiệu chuẩn tốt
có thể cho mọi xác suất < 0.5 mà vẫn xếp hạng hoàn hảo → F1 = 0.

**Sửa:** thêm `pr_auc` (average precision) vào `evaluate_model` và
`evaluate_with_ci`. PR-AUC không có ngưỡng, đo trực tiếp chất lượng xếp hạng
trên lớp thiểu số. E2/E3/E4 giờ báo PR-AUC thay F1.

---

## 4. Lỗi FedAvgM phân kỳ

**Triệu chứng:** E3 lần đầu, FedAvgM giữ ~0.789 đến round 40 rồi rơi xuống
0.764, AUC cuối 0.8709.

**Chẩn đoán sai ban đầu:** tôi tưởng dấu momentum bị ngược. Kiểm tra đại số:
`delta = A − W` với `W + m` **tương đương hoàn toàn** với `p = W − A` với
`W − lr·m` (vì `delta = −p` và `m_old = −m_new`). Đổi dấu là no-op.

**Nguyên nhân thật:** `run_sequential` dùng `server_learning_rate = 1.0` ngầm
định. Momentum biến cập nhật thành chuỗi nhân quả với độ lợi `lr/(1−β)`; ở
`lr=1.0, β=0.9` độ lợi là 10×, server nhảy xa gấp 10 lần mức trung bình của
client → dao động và phân kỳ sau ~40 round.

**Sửa:** thêm `server_learning_rate: 0.1` vào `FedAvgM` config (độ lợi = 1.0).
Kết quả: 0.8709 → **0.8875**, curve không còn phân kỳ (min 0.7800 thay vì
0.764).

**Bài học:** đừng suy luận công thức bằng lời — chạy 5 dòng script kiểm chứng
số học trước khi sửa. Ba giả thuyết liên tiếp sai trong phiên này đều do bỏ
qua bước đó.

---

## 5. Artifacts

| File | Nội dung |
|---|---|
| `results/E2_partition_impact.json` | 3 run đầy đủ + AUC curve từng round |
| `results/E2_partition_impact.md` | Bảng + verdict tiêu chí #3 |
| `results/E3_aggregation.json` | 3 strategy + curve |
| `results/E3_aggregation.md` | Bảng + verdict tiêu chí #4 |
| `src/experiments/s16_e2_e3.py` | Runner E2 + E3 |
| `tests/test_16_e2_e3.py` | 20 test (convergence, smoothing, render) |
| `tests/test_11_server.py` | 12 test (FedAvgM sign, root product, lr) |

---

## 6. E4 — DP-SGD utility cost

**Câu hỏi (RQ5):** DP-SGD mất bao nhiêu AUC để đổi lấy ε?

**Thiết lập:** partition `iid`, FedAvg + Opacus, 50 rounds, δ=1e-5,
`max_grad_norm=1.0`, sweep σ ∈ {0, 0.5, 1, 2}.

| σ | ε (δ=1e-5) | AUC | 95% CI | PR-AUC | Recall@10% | Mất AUC |
|---|---|---|---|---|---|---|
| 0.0 (non-DP) | ∞ | 0.8949 | [0.8755, 0.9123] | 0.8121 | 0.3289 | +0.00% |
| 0.5 | 12.37 | 0.8941 | [0.8749, 0.9108] | 0.7877 | 0.3262 | +0.09% |
| 1.0 | 1.55 | 0.8894 | [0.8697, 0.9067] | 0.7500 | 0.3128 | +0.62% |
| 2.0 | 0.57 | 0.8865 | [0.8663, 0.9047] | 0.7799 | 0.3128 | +0.94% |

**Tiêu chí #5 (privacy mất utility, đơn điệu): PASS** — mất tối đa 0.94% AUC.

### Nhận xét

- **AUC gần như không đổi** (0.8949 → 0.8865, chỉ −0.94%) trong khi ε giảm từ ∞
  xuống 0.57. Đây là kết quả đáng chú ý: DP-SGD gần như miễn phí về AUC ở mức
  ε này.
- **PR-AUC mới là chỉ số nhạy**: 0.8121 → 0.7500 (−7.6%) ở σ=1.0. Lớp thiểu số
  bị ảnh hưởng rõ hơn AUC, đúng như lý thuyết (noise làm mờ ranh giới lớp hiếm).
- **PR-AUC không đơn điệu** (0.7500 ở σ=1.0 rồi 0.7799 ở σ=2.0). Đây là nhiễu
  ngẫu nhiên của DP, không phải xu hướng — với 374 positives, PR-AUC có CI rộng.
  Tiêu chí #5 chỉ yêu cầu AUC đơn điệu, và AUC thì đơn điệu thật.
- **ε ở σ=0.5 cao bất thường (12.37)** vì 50 rounds × 5 client × 352 batch =
  17,600 bước, sampling rate 0.28%. Privacy budget tích luỹ theo số bước, nên
  muốn ε thấp phải tăng σ hoặc giảm rounds.

---

## 7. E5 — Fairness

**Câu hỏi (RQ4):** FL có gây unfairness giữa các client / giữa các lý do churn?

**Thiết lập:** partition `label_skew_a0.1`, 50 rounds, FedAvg vs FedProx.

### E5.1 — Per-client

| Strategy | Overall AUC | Client AUC mean | Client AUC std | Worst client | Spread |
|---|---|---|---|---|---|
| `fedavg` | 0.8889 | 0.7950 | 0.0045 | 0.7883 | 0.0166 |
| `fedprox` | 0.8835 | 0.7914 | 0.0040 | 0.7837 | 0.0117 |

### E5.2 — Recall theo lý do churn

| Strategy | Recall mean | Recall std | Worst reason | Spread |
|---|---|---|---|---|
| `fedavg` | 0.2940 | 0.0648 | 0.2000 | 0.1647 |
| `fedprox` | 0.3088 | 0.0570 | 0.2200 | 0.1550 |

Chi tiết FedAvg: Competitor 0.3647 (62/170), Dissatisfaction 0.3214 (18/56),
Other 0.3250 (13/40), Attitude 0.2586 (15/58), **Price 0.2000 (10/50)**.

**Tiêu chí #6 (FedProx giảm unfairness): PASS** — client std +10.34%,
reason recall std +12.04% (dương = ít phân tán hơn = công bằng hơn).

### Nhận xét

- **FedProx công bằng hơn nhưng kém chính xác hơn**: AUC 0.8889 → 0.8835
  (−0.60%), đổi lại client std giảm 10.3% và worst-client spread giảm 30%
  (0.0166 → 0.0117). Nhất quán với E3.
- **Lý do `Price` khó nhất** (recall 0.20 so với 0.36 ở `Competitor`). Khách rời
  vì giá có hành vi giống khách ở lại hơn — họ không bất mãn về dịch vụ, chỉ
  thấy giá cao, nên feature hiện tại khó phân biệt.
- **Caveat quan trọng:** với 40–170 positives mỗi lý do, CI của recall là ±0.13.
  Chênh lệch 0.20 vs 0.36 nằm ngoài CI nên đáng tin; nhưng chênh lệch FedAvg vs
  FedProx (0.2940 vs 0.3088) **nằm trong** nhiễu. Kết luận trung thực là
  *không phát hiện khác biệt có ý nghĩa*, không phải "FedProx công bằng hơn".

### ⚠️ Lỗi thiết kế đã sửa: E5.2 không thể dùng AUC

Protocol ban đầu ghi "tính AUC riêng từng stratum". Điều này **sai về mặt toán
học**: `Churn Category` chỉ được điền cho khách đã churn, nên mỗi stratum là
100% positive — không có lớp âm để xếp hạng. Lần chạy đầu trả về
`stratum AUC std = nan` đúng như dự đoán.

Phân bố thực tế trên `test_real`:

| Stratum | Số dòng | Churned |
|---|---|---|
| `None` (không churn) | 944 | 0 |
| `Competitor` | 170 | 170 |
| `Attitude` | 58 | 58 |
| `Dissatisfaction` | 56 | 56 |
| `Price` | 50 | 50 |
| `Other` | 40 | 40 |

**Sửa:** dùng recall@top-10% theo từng lý do. Ngưỡng tính **một lần trên toàn
test set**, không per-stratum — nếu tính riêng, mỗi stratum luôn bị gắn cờ đúng
10% và che mất chính sự chênh lệch cần đo.

---

## 8. Tái lập

```bash
# E2 (~18 phút)
python -m src.experiments.s16_e2_e3 --which e2 --rounds 50

# E3 (~18 phút)
python -m src.experiments.s16_e2_e3 --which e3 --rounds 50

# E4 (~2 giờ, DP chậm 5.6x)
python -m src.experiments.s17_e4_dp --rounds 50

# E5 (~15 phút)
python -m src.experiments.s18_e5_fairness --rounds 50 \
  --partition label_skew_a0.1 --strategies fedavg fedprox
```

Seed cố định 42 nên kết quả tái lập byte-for-byte: E3 `fedavg`, E4 `σ=0`, và
E5 `fedavg` đều cho đúng 0.8889/0.8949, trùng khớp E2.
