# 02 — Reading Notes: Federated Learning

**Owner:** Person B · **Tuần:** W1 · **Trạng thái:** ✅

---

## 1. McMahan et al. (2017) — *Communication-Efficient Learning of Deep Networks from Decentralized Data*

**Venue:** AISTATS 2017 · **Vai trò:** nền tảng của FedAvg

### 1.1 Bài toán

Huấn luyện model trên dữ liệu phân tán ở nhiều thiết bị/client mà **không** đưa dữ liệu về server.
Mục tiêu kép: (a) bảo vệ privacy, (b) giảm chi phí communication.

### 1.2 Thuật toán FedAvg

```
Server khởi tạo w_0
for t = 1..T:
    S_t ← sample ngẫu nhiên m clients (fraction_fit)
    for each client k ∈ S_t (song song):
        w_k ← LocalSGD(w_t, D_k, E epochs)
    w_{t+1} ← Σ_k (n_k / n) · w_k        # trung bình có trọng số theo số mẫu
```

**Điểm mấu chốt:** trung bình **có trọng số theo `n_k`** (số mẫu của client), không phải trung bình cộng đơn giản.

### 1.3 Ba tham số điều khiển

| Tham số | Ý nghĩa | Giá trị dự án |
|---|---|---|
| `C` (fraction_fit) | Tỉ lệ client tham gia mỗi round | 0.5 |
| `E` (local epochs) | Số epoch train cục bộ mỗi round | 1 |
| `B` (batch size) | Batch size cục bộ | 256 |

**Trade-off:** tăng `E` → giảm số round cần thiết nhưng tăng client drift (đặc biệt khi non-IID).

### 1.4 Kết quả chính

- FedAvg giảm communication rounds **10–100×** so với FedSGD (1 batch gradient/client/round).
- Trên MNIST/CIFAR, FedAvg gần đạt centralized performance.
- **Non-IID làm giảm đáng kể** chất lượng — đây chính là động lực cho H1/H2 của dự án.

### 1.5 Áp dụng vào dự án

| Khái niệm | Mapping |
|---|---|
| Client | 10 MinIO silo (`s3://clients/client_{i}/`) |
| `D_k` | Partition dữ liệu của client k (IID hoặc Dirichlet) |
| `n_k` | Số mẫu trong silo k |
| `w` | Trọng số MLP (PyTorch `state_dict`) |
| Communication | gRPC qua Flower |

---

## 2. Li et al. (2020) — *Federated Optimization in Heterogeneous Networks* (FedProx)

**Venue:** MLSys 2020 · **Vai trò:** giải pháp cho non-IID (H2)

### 2.1 Vấn đề với FedAvg trên non-IID

Khi phân phối label giữa các client khác nhau mạnh (α nhỏ), mỗi client tối ưu một local objective
rất khác nhau → **client drift**: local model bị kéo lệch khỏi global model, aggregation cho kết quả kém.

### 2.2 Giải pháp: proximal term

Thêm ràng buộc kéo local model về gần global model:

$$
\min_{w} \; F_k(w) + \frac{\mu}{2}\|w - w^t\|^2
$$

- $F_k(w)$: loss cục bộ của client k
- $w^t$: global model tại round t
- $\mu$: hệ số proximal (**μ = 0.01** trong dự án)

**Trực giác:** nếu local optimum quá xa global model, proximal term phạt nặng → giữ client "gần" server.

### 2.3 Gradient của FedProx

$$
\nabla \left[F_k(w) + \frac{\mu}{2}\|w - w^t\|^2\right] = \nabla F_k(w) + \mu(w - w^t)
$$

Trong PyTorch, thêm vào loss:

```python
proximal_term = 0.0
for name, param in local_model.named_parameters():
    proximal_term += ((param - global_params[name]) ** 2).sum()
loss = criterion(output, target) + (mu / 2) * proximal_term
```

### 2.4 Kết quả chính

- FedProx chịu được **heterogeneity** tốt hơn FedAvg (system + statistical).
- Cho phép **partial work** (client không hoàn thành đủ E epochs vẫn đóng góp).
- Cải thiện **fairness**: giảm phương sai per-client accuracy.

### 2.5 Áp dụng vào dự án

| Khái niệm | Mapping |
|---|---|
| μ = 0.01 | `configs/base.yaml` → `fl.strategies.fedprox.proximal_mu` |
| Đo hiệu quả | E3 (α=0.1): FedAvg vs FedProx vs FedAvgM |
| Đo fairness | E5.1 (per-client AUC), E5.2 (stratified by `Churn Category`) |

---

## 3. FedAvgM — Server-side momentum

**Nguồn:** Hsu et al. (2019), *Measuring the Effects of Non-Identical Data Distribution for
Federated Visual Classification* · **Vai trò:** strategy thứ 3 cho E3

### 3.1 Ý tưởng

Áp dụng momentum ở **server** thay vì client:

$$
v_{t+1} = \beta v_t + \Delta_t, \qquad w_{t+1} = w_t + v_{t+1}
$$

với $\Delta_t = \sum_k \frac{n_k}{n} w_k - w_t$ là "pseudo-gradient" từ aggregation.

- $\beta = 0.9$ (server momentum)
- Giảm dao động của global model qua các round khi client distribution không đồng nhất

### 3.2 So sánh 3 strategies

| Strategy | Cơ chế | Ưu điểm | Nhược điểm |
|---|---|---|---|
| **FedAvg** | Trung bình có trọng số | Đơn giản, baseline | Kém dưới non-IID |
| **FedProx** | Proximal term ở client | Chống client drift | Cần tune μ |
| **FedAvgM** | Momentum ở server | Giảm dao động | Cần tune β |

---

## 4. Bảng tổng hợp tham số FL của dự án

| Tham số | Giá trị | Nguồn |
|---|---|---|
| `n_rounds` | 50 | Spec |
| `fraction_fit` | 0.5 | Spec |
| `fraction_evaluate` | 0.5 | Spec |
| `local_epochs` | 1 | Spec |
| `batch_size` | 256 | Spec |
| `learning_rate` | 1e-3 | Spec |
| FedProx `μ` | 0.01 | Li 2020 |
| FedAvgM `β` | 0.9 | Hsu 2019 |
| K (clients) | 10 (benchmark 5/10/20/50) | Spec |

---

## 5. Câu hỏi mở cần giải quyết ở W4–W6

1. **Weighted vs unweighted aggregation:** Flower mặc định weighted theo `num_examples`. Cần xác nhận
   client trả đúng `num_examples` trong `fit()` metrics.
2. **Client drift đo thế nào?** Đề xuất: $\|w_k - w^t\|_2$ sau mỗi round, log lên MLflow.
3. **FedProx với `local_epochs=1`:** proximal term có tác dụng rõ không? Có thể cần tăng E lên 2–3
   cho thí nghiệm E3 để thấy khác biệt.
4. **FedAvgM initialization:** $v_0 = 0$ hay $v_0 = \Delta_0$? Ảnh hưởng round đầu.

---

## 6. Tài liệu tham khảo

1. McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017).
   *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
2. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020).
   *Federated Optimization in Heterogeneous Networks.* MLSys.
3. Hsu, T.-M. H., Qi, H., & Brown, M. (2019). *Measuring the Effects of Non-Identical Data
   Distribution for Federated Visual Classification.* arXiv:1909.06335.
4. Flower documentation — <https://flower.ai/docs/>
