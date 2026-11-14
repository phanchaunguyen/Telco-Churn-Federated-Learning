# 03 — Reading Notes: Differential Privacy

**Owner:** Person C · **Tuần:** W1 · **Trạng thái:** ✅

---

## 1. Abadi et al. (2016) — *Deep Learning with Differential Privacy* (DP-SGD)

**Venue:** CCS 2016 · **Vai trò:** nền tảng cho E4 (H3)

### 1.1 Định nghĩa (ε, δ)-Differential Privacy

Một cơ chế ngẫu nhiên $\mathcal{M}$ là $(\varepsilon, \delta)$-DP nếu với mọi cặp dataset
$D, D'$ khác nhau **đúng 1 bản ghi**, và mọi tập output $S$:

$$
\Pr[\mathcal{M}(D) \in S] \le e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta
$$

**Trực giác:**
- $\varepsilon$ nhỏ → privacy mạnh (output gần như không phụ thuộc vào 1 cá nhân)
- $\delta$ = xác suất "vỡ" guarantee (thường đặt $\delta \ll 1/n$, dự án dùng **δ = 1e-5**)

### 1.2 DP-SGD — 4 bước mỗi minibatch

```
1. Sample minibatch L (random)
2. Với mỗi mẫu i:  g_i ← ∇_w L(w, x_i)          # gradient riêng từng mẫu
3. Clip:           ḡ_i ← g_i / max(1, ‖g_i‖₂ / C)   # C = max_grad_norm = 1.0
4. Cộng nhiễu:     g̃ ← (1/L)(Σ_i ḡ_i + N(0, σ²C²I))  # σ = noise_multiplier
5. Update:         w ← w - η·g̃
```

**Hai siêu tham số then chốt:**

| Tham số | Ý nghĩa | Giá trị dự án |
|---|---|---|
| `max_grad_norm` (C) | Ngưỡng clip gradient | **1.0** |
| `noise_multiplier` (σ) | Độ lớn nhiễu Gaussian | sweep {0, 0.5, 1, 2} |

### 1.3 Tại sao phải clip?

Không clip → một mẫu có gradient lớn có thể "át" toàn bộ minibatch, khiến nhiễu không đủ che.
Clip đảm bảo **sensitivity** của tổng gradient bị chặn trên bởi $C$ → biết chính xác cần bao nhiêu nhiễu.

### 1.4 Privacy accounting — RDP

Không thể cộng ε tuyến tính qua các bước. Abadi dùng **moments accountant** (RDP):

- Mỗi bước DP-SGD tiêu tốn một lượng privacy budget
- Sau T bước, tổng ε được tính bằng **RDP → (ε, δ)-DP conversion**
- Opacus cung cấp `RDPAccountant` và `get_epsilon(delta)`

**Quan hệ định tính:** σ ↑ → ε ↓ (privacy tốt hơn) nhưng utility ↓ (AUC giảm).

### 1.5 Kết quả chính của paper

- DP-SGD train được deep net (MNIST, CIFAR) với ε ≈ 1–10 mà accuracy giảm vừa phải.
- **Trade-off không tuyến tính**: từ σ=0 đến σ=1 mất nhiều accuracy hơn từ σ=1 đến σ=2.
- Cần **train lâu hơn** (nhiều epoch hơn) để bù nhiễu.

---

## 2. Opacus — thư viện DP-SGD cho PyTorch

### 2.1 Ba thành phần cốt lõi

| Thành phần | Vai trò |
|---|---|
| `PrivacyEngine` | Gắn DP vào optimizer + dataloader |
| `GradSampleModule` | Chuyển model sang chế độ per-sample gradient |
| `RDPAccountant` | Tính ε từ σ, δ, số bước |

### 2.2 API cơ bản

```python
from opacus import PrivacyEngine

model = MLP(input_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
data_loader = DataLoader(dataset, batch_size=256)

privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)

# ... training loop như bình thường ...

epsilon = privacy_engine.get_epsilon(delta=1e-5)
```

### 2.3 Ràng buộc kỹ thuật quan trọng

| Ràng buộc | Chi tiết |
|---|---|
| **BatchNorm không tương thích** | BatchNorm tính thống kê trên cả batch → vi phạm per-sample gradient. Phải thay bằng **GroupNorm** hoặc **LayerNorm** |
| **Batch size phải chia hết** | `len(dataset) % batch_size` phải xử lý (drop_last hoặc pad) |
| **Không dùng `loss.backward()` trực tiếp** | Phải qua `optimizer.step()` của Opacus |
| **Model phải ở train mode** | `model.train()` bắt buộc |

> ⚠️ **Ảnh hưởng tới kiến trúc MLP:** spec dùng `BatchNorm`. Khi bật DP-SGD phải chuyển sang
> `GroupNorm`. Cần quyết định: (a) dùng GroupNorm cho **tất cả** thí nghiệm để so sánh công bằng,
> hay (b) chỉ đổi khi bật DP. **Đề xuất (a)** — tránh confound giữa kiến trúc và privacy.

### 2.4 Đo ε trong dự án

```python
# Sau mỗi round FL, log ε hiện tại
epsilon = privacy_engine.get_epsilon(delta=1e-5)
mlflow.log_metric("epsilon", epsilon, step=round_num)
```

**Lưu ý:** trong FL, mỗi client có privacy engine riêng. ε của **một client** là ε của nó;
ε toàn cục phụ thuộc vào composition qua các round. Cần ghi rõ trong báo cáo đang báo ε theo nghĩa nào.

---

## 3. Mapping vào dự án

| Khái niệm | Mapping |
|---|---|
| `noise_multiplier` sweep | E4: {0, 0.5, 1, 2} |
| `max_grad_norm` | 1.0 (cố định) |
| `delta` | 1e-5 |
| ε target | H3: ε ∈ [1, 10], AUC giảm ≤ 5% |
| Metric | AUC vs ε curve |
| Config | `configs/base.yaml` → `privacy.*` |

---

## 4. Câu hỏi mở cần giải quyết ở W7

1. **BatchNorm → GroupNorm:** đổi cho tất cả thí nghiệm hay chỉ DP? (đề xuất: tất cả)
2. **ε báo cáo theo nghĩa nào?** Per-client per-round, hay composed qua 50 rounds?
   → Đề xuất: báo cáo **composed ε sau 50 rounds** cho mỗi client, kèm ghi chú.
3. **σ = 0 có hợp lệ?** σ=0 → ε = ∞ (không privacy). Đây là baseline "DP tắt" trong sweep.
4. **Ảnh hưởng của `local_epochs=1`:** với 1 epoch, số bước DP ít → ε nhỏ hơn. Cần ghi rõ
   trong bảng kết quả E4.
5. **Nhiễu cộng ở client hay server?** Trong FL chuẩn, DP-SGD chạy ở **client** (local DP).
   Nếu muốn central DP cần secure aggregation — ngoài scope.

---

## 5. Tài liệu tham khảo

1. Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L. (2016).
   *Deep Learning with Differential Privacy.* ACM CCS.
2. Mironov, I. (2017). *Rényi Differential Privacy.* IEEE CSF.
3. Opacus documentation — <https://opacus.ai/>
4. Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy.*
   Foundations and Trends in Theoretical Computer Science.
