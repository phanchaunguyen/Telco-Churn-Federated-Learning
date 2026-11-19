# 14 — Naming Convention

**Owner:** All · **Trạng thái:** ✅ Đã áp dụng
**Áp dụng từ:** 2026-11-20

---

## 1. Nguyên tắc

Mọi file hoặc thư mục **có thứ tự thực thi** đều mang tiền tố số ở đầu tên, để
`ls` và `sort` tự động cho ra đúng thứ tự đọc/chạy.

Có **hai dạng tiền tố**, vì Python không cho phép module bắt đầu bằng chữ số:

| Loại | Dạng tiền tố | Ví dụ | Lý do |
|---|---|---|---|
| Module Python (`.py` được import) | `sNN_` | `s01_clean.py` | `from src.data.01_clean import ...` là **syntax error** |
| Mọi thứ còn lại (docs, scripts, tests, notebooks) | `NN_` | `02_run_clean.sh` | Không bị ràng buộc cú pháp |

`NN` là số 2 chữ số. Số chẵn chục (00, 10, 20) dành cho mục tổng hợp/index.

---

## 2. Thứ tự pipeline

Số thứ tự phản ánh **vị trí trong pipeline**, không phải thứ tự viết code.

```
CLEAN → SPLIT → AMPLIFY → QUALITY GATE → PARTITION → VERIFY
```

| # | Stage | Module pandas | Module Spark |
|---|---|---|---|
| 00 | Hạ tầng dùng chung | — | `s00_spark_session.py` |
| 01 | CLEAN | `s01_clean.py` | `s01_clean_spark.py` |
| 02 | SPLIT | `s02_split.py` | — |
| 03 | AMPLIFY | `s03_amplify.py` | `s03_amplify_spark.py` |
| 04 | QUALITY GATE | `s04_quality_gate.py` | — |
| 05 | PARTITION | `s05_partition.py` | `s05_partition_spark.py` |
| 06 | VERIFY | `s06_verify_silos.py` | — |

Quy ước: bản pandas và bản Spark của cùng một stage **dùng chung số**, phân biệt
bằng hậu tố `_spark`.

---

## 3. Cấu trúc hiện tại

### 3.1 `src/`

```
src/
  __init__.py
  config.py                      # không đánh số — là hạ tầng, không phải stage
  data/
    __init__.py
    s00_spark_session.py         # SparkSession factory
    s01_clean.py                 # CLEAN (pandas, reference implementation)
    s01_clean_spark.py           # CLEAN (Spark, parity-checked)
    s02_split.py                 # SPLIT 80/20 + 90/10
    s03_amplify.py               # AMPLIFY (rank-based Gaussian Copula)
    s03_amplify_spark.py         # AMPLIFY (Spark parallel sampling)
    s04_quality_gate.py          # KS + correlation + TSTR
  partitioning/
    __init__.py
    s05_partition.py             # IID + Dirichlet assignment
    s05_partition_spark.py       # PARTITION → parquet/MinIO
    s06_verify_silos.py          # GATE cuối Layer 1
```

### 3.2 `scripts/`

```
scripts/
  00_up.sh                       # docker compose up
  01_verify_env.py               # smoke test môi trường
  02_run_clean.sh                # chạy CLEAN
  03_benchmark_spark.py          # benchmark 10K/100K/1M
```

### 3.3 `tests/`

```
tests/
  __init__.py
  test_01_clean.py               # 29 tests
  test_02_split.py               # 16 tests
  test_05_partition.py           # 20 tests
```

Số trong tên test khớp số của module tương ứng.

### 3.4 `docs/`

```
docs/
  00_index.md
  01_kickoff_notes.md
  02_reading_notes_fl.md
  03_reading_notes_dp.md
  04_data_decisions.md
  05_spark_pipeline.md
  06_synthetic_decision.md
  07_partition_design.md
  08_kafka_telemetry.md
  12_architecture.md
  13_layer1_plan.md
  14_naming_convention.md        # file này
```

Số 09–11 đã reserve cho `09_fl_architecture`, `10_experiment_protocol`,
`11_integration_log`.

### 3.5 `notebooks/`

```
notebooks/
  01_eda.ipynb
```

---

## 4. Quy tắc khi thêm file mới

1. **Xác định stage** của file trong pipeline.
2. **Dùng số của stage đó.** Nếu là biến thể (Spark, GPU, …), thêm hậu tố.
3. **Không đánh số lại** file đã có — chỉ cấp số mới.
4. **Cập nhật `docs/00_index.md`** nếu là doc.
5. **Cập nhật file này** nếu thêm stage mới.

### Ví dụ

| File mới | Tên đúng | Lý do |
|---|---|---|
| Evaluation module | `src/evaluation/s07_eval.py` | stage 07 |
| Spark eval | `src/evaluation/s07_eval_spark.py` | cùng stage, hậu tố |
| Test cho eval | `tests/test_07_eval.py` | khớp số module |
| Script chạy eval | `scripts/04_run_eval.sh` | script dùng số riêng |
| Doc về eval | `docs/15_eval_design.md` | số doc tiếp theo |

---

## 5. Ngoại lệ

Các file sau **không** đánh số vì không có thứ tự thực thi:

| File | Lý do |
|---|---|
| `src/config.py` | Hạ tầng dùng chung, không phải stage |
| `src/__init__.py`, `*/__init__.py` | Package marker |
| `tests/__init__.py` | Package marker |
| `README.md`, `promt.md`, `Task_breakdown.md` | Tài liệu gốc, không theo pipeline |
| `docker-compose.yml`, `pyproject.toml`, `.gitignore` | Cấu hình dự án |
| `configs/base.yaml` | Cấu hình, không phải stage |

---

## 6. Lịch sử đổi tên (2026-11-20)

| Tên cũ | Tên mới |
|---|---|
| `src/data/spark_session.py` | `src/data/s00_spark_session.py` |
| `src/data/clean.py` | `src/data/s01_clean.py` |
| `src/data/spark_clean.py` | `src/data/s01_clean_spark.py` |
| `src/data/split.py` | `src/data/s02_split.py` |
| `src/data/amplify.py` | `src/data/s03_amplify.py` |
| `src/data/spark_amplify.py` | `src/data/s03_amplify_spark.py` |
| `src/data/quality_gate.py` | `src/data/s04_quality_gate.py` |
| `src/partitioning/partitioner.py` | `src/partitioning/s05_partition.py` |
| `src/partitioning/spark_partition.py` | `src/partitioning/s05_partition_spark.py` |
| `src/partitioning/verify_silos.py` | `src/partitioning/s06_verify_silos.py` |
| `scripts/up.sh` | `scripts/00_up.sh` |
| `scripts/verify_env.py` | `scripts/01_verify_env.py` |
| `scripts/run_clean.sh` | `scripts/02_run_clean.sh` |
| `scripts/benchmark_spark.py` | `scripts/03_benchmark_spark.py` |
| `tests/test_clean.py` | `tests/test_01_clean.py` |
| `tests/test_split.py` | `tests/test_02_split.py` |
| `tests/test_partition.py` | `tests/test_05_partition.py` |

Toàn bộ import, docstring, và tài liệu đã được cập nhật. 65 tests vẫn pass.

---

## 7. Lệnh chạy sau khi đổi tên

```bash
# Layer 1 — pandas path
python -m src.data.s01_clean
python -m src.data.s02_split
python -m src.data.s03_amplify
python -m src.data.s04_quality_gate
python -m src.data.s02_split --synthetic
python -m src.partitioning.s05_partition
python -m src.partitioning.s06_verify_silos

# Layer 1 — Spark path
python -m src.data.s01_clean_spark
python -m src.data.s03_amplify_spark
python -m src.partitioning.s05_partition_spark
python scripts/03_benchmark_spark.py

# Tests
pytest tests/ -v
```
