# 04 — Data Decisions

**Owner:** Person A · **Tuần:** W1 · **Trạng thái:** ✅ Đã chốt
**Nguồn số liệu:** `notebooks/01_eda.ipynb` → `results/eda_*.csv`

---

## 1. Dataset thực tế

| Thuộc tính | Giá trị đo được |
|---|---|
| File | `data/raw/telco.csv` |
| Shape | **7,043 rows × 50 columns** |
| `Total Charges` dtype | **float64** — không cần cast |
| `Customer Status` | 3 giá trị: Churned / Stayed / **Joined** |
| Context | US / California / Q3 — single-quarter snapshot |

### 1.1 Hai cạm bẫy của file gốc (đã xử lý trong `load_raw()`)

| # | Vấn đề | Hậu quả nếu bỏ qua | Cách xử lý |
|---|---|---|---|
| 1 | Header lỗi: `"Total Refu    nds"` (4 dấu cách chèn giữa) | Cột này **lọt vào feature matrix** vì không khớp tên trong drop list | `HEADER_FIXES` map về `"Total Refunds"` |
| 2 | `"None"` là category hợp lệ của `Offer` / `Internet Type` | Pandas mặc định coi `"None"` là NaN → mất category, sai phân phối | `pd.read_csv(..., keep_default_na=False, na_values=[""])` |

### 1.2 Schema corrections (khác bản IBM 21 cột cổ điển)

- ❌ Không có `Churn Value` (1/0) — chỉ có `Churn Label` (Yes/No)
- ❌ Không có `Churn Score Category`, `Satisfaction Score Label`, `Count`
- ✅ Không có sentinel `"No phone service"` / `"No internet service"`

---

## 2. Cohort: bỏ nhóm `Joined`

| Metric | Value |
|---|---|
| Tổng rows | 7,043 |
| `Joined` rows | **454 (6.45%)** |
| Cohort sau khi bỏ | **6,589** |
| Churned / Stayed | 1,869 / 4,720 |
| **Churn rate (clean cohort)** | **28.37%** |

> **Lưu ý:** spec ghi "≈ 26.5%" — đó là con số của bản IBM 21 cột. Trên dataset EXTENDED sau khi
> bỏ `Joined`, churn rate thực đo là **28.37%**. Con số đo được là con số có thẩm quyền.

**Lý do bỏ `Joined`:** khách mới chưa có đủ cửa sổ quan sát → nhãn churn không so sánh được với
khách đã ở lâu. Bỏ **trước** amplification để synthetic data không học từ nhóm này.

**Kiểm tra consistency:** `Churn Label` khớp 100% với `Customer Status` trên cả 6,589 rows
(0 mismatch) → `churn = (Customer Status == "Churned")`.

---

## 3. Leakage taxonomy

| Loại | Tên | Cột | Số cột |
|---|---|---|---|
| **1** | TARGET LEAKAGE (direct) | `Churn Label`, `Customer Status` | 2 |
| **2** | POST-HOC DERIVED | `Churn Score`, `CLTV`, `Satisfaction Score` | 3 |
| **3** | INDIRECT POST-HOC | `Total Revenue`, `Total Refunds`, `Churn Category`, `Churn Reason` | 4 |
| **4** | ZERO-VARIANCE | `Country`, `State`, `Quarter` | 3 |
| **5** | GEO HIGH-CARDINALITY | `City`, `Zip Code`, `Latitude`, `Longitude`, `Population` | 5 |
| **6** | DERIVED DUPLICATES | `Under 30`, `Referred a Friend`, `Dependents` | 3 |

**Tổng drop: 20 cột.**

### 3.1 Bằng chứng leakage — univariate AUC

Đo AUC khi dùng **một mình** cột đó để dự đoán churn (trên cohort 6,589 rows):

| Cột | Loại | Raw AUC | Hướng | Predictive power |
|---|---|---|---|---|
| `Satisfaction Score` | 2 | 0.0506 | inverse | **0.9494** |
| `Churn Score` | 2 | **0.9421** | direct | **0.9421** |
| `Total Revenue` | 3 | 0.2768 | inverse | 0.7232 |
| `CLTV` | 2 | 0.4099 | inverse | 0.5901 |
| `Total Refunds` | 3 | 0.4849 | inverse | 0.5151 |

> **Đọc bảng:** AUC < 0.5 nghĩa là quan hệ **nghịch** (satisfaction cao → churn thấp). Predictive
> power = `max(AUC, 1-AUC)`. Ba cột vượt ngưỡng 0.7 → xác nhận leakage.

### 3.2 E0 — External baseline

**`Churn Score` univariate AUC = 0.9421**

Đây là "unfair ceiling": IBM tính `Churn Score` bằng thông tin post-hoc (Satisfaction, CLTV,
Churn Category…) mà model của dự án **không được phép** dùng. E0 báo cáo:

> "IBM's precomputed Churn Score achieves AUC = 0.9421. Our centralized model without post-hoc
> features achieves AUC = Y. Our FL model achieves AUC = Z. Despite lacking post-hoc information,
> FL retains (Z/Y)×100% of centralized performance and (Z/0.9421)×100% of IBM's score."

`Churn Score` **không bao giờ** là feature.

### 3.3 Cột giữ lại cho EDA only

| Cột | Dùng cho |
|---|---|
| `Churn Category` | **E5.2** — stratify fairness theo 5 lý do churn |
| `Churn Reason` | Chapter 1 (motivation), Chapter 6 (root-cause) |
| `CLTV`, `Satisfaction Score` | EDA, phân tích exploratory |
| `Churn Score` | **E0** external baseline |
| `City`, `Zip Code`, `Latitude`, `Longitude`, `Population` | EDA địa lý |

Tất cả nằm trong `eda_7043.parquet` (D0.5), **không** nằm trong `clean_7043.parquet` (D0.6).

---

## 4. Redundancy check (Loại 6)

| Cột dẫn xuất | Cột gốc | Agreement | Quyết định |
|---|---|---|---|
| `Under 30` | `Age` | 100% | Drop `Under 30`, giữ `Age` |
| `Referred a Friend` | `Number of Referrals` | 100% | Drop `Referred a Friend`, giữ count |
| `Dependents` (Yes/No) | `Number of Dependents` | 100% | Drop binary, giữ count |

Cả 3 cột dẫn xuất là **hàm tất định** của cột gốc → giữ cột gốc (chi tiết hơn) là đủ.

---

## 5. Final feature list — **29 features**

| Nhóm | SL | Cột |
|---|---|---|
| **Demographics** | 5 | Gender, Age, Senior Citizen, Married, Number of Dependents |
| **Account** | 6 | Tenure in Months, Offer, Contract, Paperless Billing, Payment Method, **Number of Referrals** |
| **Services** | 12 | Phone Service, Multiple Lines, Internet Service, Internet Type, Online Security, Online Backup, Device Protection Plan, Premium Tech Support, Streaming TV, Streaming Movies, Streaming Music, Unlimited Data |
| **Usage & Billing** | 6 | Avg Monthly Long Distance Charges, Avg Monthly GB Download, Monthly Charge, Total Charges, Total Extra Data Charges, Total Long Distance Charges |

### 5.1 `Number of Referrals` — feature #29

Patch gốc liệt kê 28 features nhưng **bỏ sót** `Number of Referrals`, trong khi Loại 6 lại ghi
*"keep the count, drop the binary"* → ý định là giữ. Đã chốt **giữ** làm feature #29.

### 5.2 Column accounting — phủ đủ 50 cột

```
29 features
+  1  Customer ID
+  3  Loại 6 (Under 30, Referred a Friend, Dependents)
+  3  Loại 4 (Country, State, Quarter)
+  5  Loại 5 (City, Zip Code, Latitude, Longitude, Population)
+  2  Loại 1 (Churn Label, Customer Status)
+  3  Loại 2 (Churn Score, CLTV, Satisfaction Score)
+  4  Loại 3 (Total Revenue, Total Refunds, Churn Category, Churn Reason)
= 50 ✅
```

Assertion này được enforce trong `validate_column_accounting()` — nếu thêm/bớt cột mà quên cập nhật
danh sách, test sẽ fail.

### 5.3 Categorical vs numeric

| Loại | SL | Cột |
|---|---|---|
| Categorical (one-hot) | 19 | Gender, Senior Citizen, Married, Offer, Contract, Paperless Billing, Payment Method, Phone Service, Multiple Lines, Internet Service, Internet Type, Online Security, Online Backup, Device Protection Plan, Premium Tech Support, Streaming TV, Streaming Movies, Streaming Music, Unlimited Data |
| Numeric | 10 | Age, Number of Dependents, Tenure in Months, Number of Referrals, Avg Monthly Long Distance Charges, Avg Monthly GB Download, Monthly Charge, Total Charges, Total Extra Data Charges, Total Long Distance Charges |

Sau one-hot: **~42–48 numeric features** → MLP input dim đo runtime (`X_train.shape[1]`).

---

## 6. Quyết định về multicollinearity

Spearman correlation giữa các numeric feature được kiểm tra trong notebook §4.2.

**Quyết định: giữ cả 3 cột billing** (`Total Charges`, `Total Extra Data Charges`,
`Total Long Distance Charges`) dù `Total Charges` tương quan mạnh với `Tenure × Monthly Charge`.

**Lý do:**
1. MLP xử lý collinearity tốt hơn linear model (nhờ nonlinearity + regularization).
2. Drop bớt sẽ mất thông tin thật (ví dụ: khách trả nhiều hơn mức "bình thường" cho tenure đó).
3. Giữ nguyên giúp so sánh với literature dễ hơn.

---

## 7. Quyết định về `Offer = "None"`

`Offer` có giá trị `"None"` (chuỗi) cho khách không nhận offer nào. **Giữ làm category riêng**,
không gộp vào nhóm khác.

**Lý do:** "không nhận offer" là một trạng thái có ý nghĩa (khách tự đến, không qua khuyến mãi) —
gộp sẽ làm mất tín hiệu.

---

## 8. Artifacts

| Artifact | Shape | Nội dung | Dùng cho |
|---|---|---|---|
| `data/processed/eda_7043.parquet` (D0.5) | 6,589 × 50 | 50 cột gốc, bỏ `Joined`, giá trị nguyên vẹn | EDA, E0, E5.2 |
| `data/processed/clean_7043.parquet` (D0.6) | 6,589 × 31 | `Customer ID` + 29 features + `churn` | Amplification (W2) |

### 8.1 Assertions enforce trong `validate_clean()`

1. `df.shape[1] == 31`
2. Không còn cột nào trong `DROPPED_COLS`
3. `len(FEATURE_COLS) == 29`
4. Mọi feature khai báo đều có mặt
5. Target ∈ {0, 1}, không NaN
6. Không NaN trong features
7. Không còn row `Joined`
8. `Customer ID` unique

---

## 9. Bàn giao

| Cho | Nội dung |
|---|---|
| **W2 (Amplification)** | Amplify `clean_7043.parquet` (29 features + target), **không** amplify 50 cột |
| **W3 (Partitioning)** | Partition trên `synthetic_1m.parquet`; assertion: mỗi silo có đúng 29 features |
| **E0** | Dùng `e0_auc = 0.9421` làm mốc tham chiếu |
| **E5.2** | Dùng `Churn Category` từ `eda_7043.parquet` (5 strata) |

---

## 10. Tái lập

```bash
# Sinh lại cả 2 artifact
bash scripts/run_clean.sh

# Chạy test
pytest tests/test_clean.py -v

# Chạy lại EDA
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
```

---

## 11. Spark pipeline (Patch v2)

Từ W2, bước CLEAN và AMPLIFY được implement bằng **PySpark** (`src/data/spark_clean.py`,
`src/data/spark_amplify.py`) thay vì pandas thuần.

**Không thay đổi semantics:** cùng drop list, cùng 29 features, cùng 31 cột output.

`src/data/clean.py` (pandas) vẫn được **giữ làm reference implementation** để:
1. Chạy test nhanh (`pytest tests/test_clean.py`) — không cần Spark
2. Sanity check: Spark output phải khớp pandas output (row count, column set, churn rate)

**Assertion bổ sung:** `spark_clean.py` phải tái tạo đúng `clean_7043.parquet` mà
`clean.py` sinh ra — so sánh shape, column list, và churn rate (sai số 0).

**Hai cạm bẫy phải xử lý trong Spark (giống pandas):**
1. Header lỗi `"Total Refu    nds"` → rename về `"Total Refunds"`
2. `"None"` là category hợp lệ của `Offer` / `Internet Type` → **không** set `nullValue="None"`

Chi tiết: xem `docs/05_spark_pipeline.md`.
