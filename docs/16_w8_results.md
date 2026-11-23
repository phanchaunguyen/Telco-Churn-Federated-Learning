# 16 — Kết quả W8: E6 scalability + E7 communication cost

> **Trạng thái:** E6 ✅ PASS · E7 ✅ PASS
> **Ngày chạy:** 2026-11-23
> **Protocol:** `docs/10_experiment_protocol.md` §9 (E6), §10 (E7)

---

## 1. E6 — Scalability

**Câu hỏi (RQ5):** FL có scale được khi K tăng từ 5 lên 50?

**Thiết lập:** FedAvg, IID, 50 rounds, `fraction_fit=0.5`, seed 42.
Thời gian đo **chỉ vòng lặp federated** (không tính khởi tạo client).

| K | Clients/round | Tổng (s) | s/round | s/client-round | Sanity AUC (synth) |
|---|---|---|---|---|---|
| 5 | 2 | 354.4 | 7.09 | 3.544 | 0.8012 |
| 10 | 5 | 381.1 | 7.62 | 1.524 | 0.8007 |
| 20 | 10 | 370.7 | 7.41 | 0.741 | 0.7994 |
| 50 | 25 | 387.1 | 7.74 | 0.310 | 0.7960 |

**Tiêu chí (wall-clock phẳng khi K tăng): PASS** — tổng thời gian chỉ biến
động **9.2%** trên toàn dải K=5..50 (354s → 387s).

### Phát hiện chính: chi phí PHẲNG, không tuyến tính

Đây là điểm quan trọng nhất và dễ diễn giải sai. K tăng 10× (5 → 50) nhưng
thời gian gần như không đổi. Nguyên nhân:

`fraction_fit=0.5` giữ cố định, nên số client mỗi round tăng theo K **nhưng
mỗi silo nhỏ đi đúng tỉ lệ đó**. Tổng dữ liệu huấn luyện mỗi round luôn ~450K:

| K | Clients/round | Rows/client | Rows/round |
|---|---|---|---|
| 5 | 2 | 180,000 | 360,000 |
| 10 | 5 | 90,000 | 450,000 |
| 20 | 10 | 45,000 | 450,000 |
| 50 | 25 | 18,000 | 450,000 |

**Cách đọc đúng:** FL scale được về *số client* mà không scale về *thời gian*,
vì khối lượng mỗi round do sampling fraction quyết định, không phải K. Cái giá
của việc tăng K là **ít dữ liệu hơn trên mỗi client mỗi round**, không phải
nhiều compute hơn.

### Vì sao không dùng AUC làm kết quả

E6 đo **load**, không đo chất lượng. Cột AUC chỉ là sanity check trên
`synth_test` (100K): nếu nó sụp bất thường thì đó là dấu hiệu bug trong sweep,
không phải phát hiện về scaling. AUC giảm nhẹ 0.8012 → 0.7960 (−0.65%) khi K
tăng, phù hợp với việc mỗi client có ít dữ liệu hơn.

### Vì sao IID cho mọi K

Thay đổi K và label skew cùng lúc sẽ confound hai hiệu ứng. IID giữ phân phối
cố định để K là biến duy nhất thay đổi.

---

## 2. E7 — Communication cost

**Câu hỏi (RQ6):** chi phí communication có khả thi thực tế?

**Công thức:**

$$\text{bytes/round} = 2 \times |\theta| \times 4 \times n_{\text{selected}}$$

- $|\theta| = 18{,}049$ tham số
- $4$ = bytes mỗi float32
- $2$ = upload + download

**Hệ số 2 là phần hay bị bỏ sót.** Mỗi round vừa gửi global model xuống từng
client được chọn, vừa gửi weights cập nhật ngược lên server — nên chi phí
đường truyền gấp đôi kích thước model mỗi client.

**Kết quả:**

| Fraction | Clients/round | MB/round | Cumulative MB | AUC (real) | PR-AUC (real) | Recall@10% | AUC (synth) |
|---|---|---|---|---|---|---|---|
| 0.25 | 2 | 0.289 | 14.4 | 0.8942 | 0.8099 | 0.3289 | 0.7994 |
| 0.50 | 5 | 0.722 | 36.1 | 0.8949 | 0.8121 | 0.3289 | 0.8007 |
| 1.00 | 10 | 1.444 | 72.2 | 0.8958 | 0.8152 | 0.3316 | 0.8008 |

**Tiêu chí (cấu hình rẻ nhất nằm trong 2% AUC): PASS** — fraction=0.25 chỉ tốn
0.289 MB/round và mất **0.18%** AUC so với cấu hình tốt nhất.

### Nhận xét

- **Sampling ít gần như miễn phí.** Giảm từ 10 client xuống 2 client/round
  (giảm 5× băng thông) chỉ mất 0.18% AUC. Đây là kết quả thực dụng nhất của E7:
  với model nhỏ 18K tham số, chi phí communication không phải nút thắt.
- **Chi phí tuyến tính theo số client được chọn**, đúng như công thức:
  0.289 → 0.722 → 1.444 MB (tỉ lệ 1 : 2.5 : 5, khớp 2 : 5 : 10 client).
- **Tổng 72 MB cho 50 rounds** ở cấu hình đầy đủ — nhỏ hơn một bức ảnh. Với
  model lớn hơn (LLM), con số này sẽ khác hoàn toàn, nên kết luận phải giới hạn
  ở quy mô model của đồ án.
- **Recall@10% gần như không đổi** (0.3289 → 0.3316) trong khi AUC tăng nhẹ.
  Điều này cho thấy top-10% khách rủi ro nhất được xác định ổn định bất kể
  sampling fraction.

### Chỉ tính weights

Không tính features, labels, gradients, hay thống kê per-sample. `num_examples`
được gửi để server weight trung bình, nhưng chỉ là một số nguyên mỗi client nên
bỏ qua vì không đáng kể.

### Lưu ý về fraction=0.25

Với K=10, lựa chọn là `max(2, round(10 × 0.25)) = 2` client, không phải "2–3"
như protocol phác thảo. Ngưỡng sàn 2 tồn tại vì FedAvg cần ít nhất hai client
để lấy trung bình.

---

## 3. Bug đã sửa: ghi đè partition

`build_partitions(k=5)` ghi vào `partitions_iid` — **cùng thư mục với K=10**.
Hệ quả: 10 file K=10 bị ghi đè bởi 5 file K=5, và `client_5..9` cũ còn sót lại.
`fit_vocabulary` sau đó đọc 10 file trong thư mục lẽ ra chỉ có 5 → trộn hai
partitioning khác nhau.

**Sửa hai lớp:**

1. Thêm param `k_tag: bool = False`. Khi `True`, thư mục thành
   `partitions_iid_k{k}`. K=10 giữ tên cũ để E1–E5 vẫn tái lập.
2. Xoá `client_*.parquet` cũ trước khi ghi, để file thừa từ lần chạy K lớn hơn
   không sống sót.

**Test:** 3 test mới trong `tests/test_05_partition.py` — `k_tag` thêm hậu tố,
mặc định giữ tên cũ, và dọn file thừa.

---

## 4. Artifacts

| File | Nội dung |
|---|---|
| `results/E6_scalability.json` | 4 run + timing từng round |
| `results/E6_scalability.md` | Bảng + giải thích vì sao phẳng |
| `results/E7_communication.json` | 3 fraction + bytes |
| `results/E7_communication.md` | Bảng + công thức |
| `src/experiments/s19_e6_scalability.py` | Runner E6 |
| `src/experiments/s20_e7_communication.py` | Runner E7 |
| `tests/test_19_e6.py` | 12 test |
| `tests/test_20_e7.py` | 13 test |
| `data/synthetic/partitions_iid_k{5,20,50}/` | Partition cho K sweep |

---

## 5. Tái lập

```bash
# Sinh partition cho K sweep (chỉ cần một lần)
python -c "
from src.partitioning.s05_partition import build_partitions
for k in (5, 20, 50):
    build_partitions('iid', k=k, k_tag=True, write=True)
"

# E6 (~25 phút)
python -m src.experiments.s19_e6_scalability --rounds 50

# E7 (~18 phút)
python -m src.experiments.s20_e7_communication --rounds 50
```

Seed cố định 42. K=10 trong E6 cho sanity AUC 0.8007 trên `synth_test`, khớp
với E7 fraction=0.5 (cùng cấu hình).
