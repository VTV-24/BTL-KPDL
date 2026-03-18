# BTL-KPDL - Phân tích doanh số siêu thị

Đồ án môn Khai phá dữ liệu, xây dựng pipeline phân tích doanh số siêu thị theo hướng end-to-end, bao gồm:

1. Tiền xử lý dữ liệu và tạo đặc trưng.
2. Market Basket Analysis (Association Rules).
3. Customer Segmentation (RFM + KMeans).
4. Supervised Modeling (phân loại segment).
5. Forecasting doanh thu theo thời gian.
6. Dashboard trình bày kết quả bằng Streamlit.

Thiết kế dự án theo triết lý:
- OOP + modular package trong `src/`.
- Notebook-per-task cho giải thích và báo cáo.
- Script-per-task cho chạy pipeline tự động và tái lập.

---

## Mục lục

- 1) Mục tiêu dự án
- 2) Dataset
- 3) Cấu trúc thư mục
- 4) Cài đặt môi trường
- 5) Chạy nhanh toàn bộ pipeline
- 6) Mô tả notebooks (Notebook-per-task)
- 7) Mô tả scripts CLI
- 8) Kiến trúc mã nguồn `src`
- 9) Cấu hình tham số
- 10) Output và artefacts
- 11) Chạy dashboard Streamlit
- 12) Kết quả mong đợi và hướng mở rộng
- 13) Tác giả
- 14) Tổng hợp công việc theo nhánh

---

## 1) Mục tiêu dự án

Dự án tập trung trả lời các bài toán kinh doanh cốt lõi trong phân tích bán lẻ:

- Sản phẩm nào thường được mua cùng nhau?
- Có thể chia khách hàng thành các nhóm hành vi nào để tối ưu marketing?
- Mô hình nào dự đoán nhóm khách hàng tốt nhất?
- Doanh thu trong các tháng tới có xu hướng như thế nào?

Giá trị thực tế của dự án:

- Hỗ trợ quyết định cross-sell/bundle.
- Ưu tiên chiến dịch giữ chân và tái kích hoạt khách hàng.
- Định hướng kế hoạch tồn kho, ngân sách marketing và mục tiêu doanh thu.

---

## 2) Dataset

- Nguồn dữ liệu: Superstore Sales Dataset (Kaggle).
- Input chính: `data/raw/train.csv`.
- Một số cột quan trọng: `Order ID`, `Customer ID`, `Product Name`, `Sub-Category`, `Sales`, `Order Date`, `Region`, `Segment`, ...

Sau tiền xử lý, dự án tạo các bảng/tệp trung gian:

- `data/processed/cleaned.parquet`: dữ liệu đã làm sạch.
- `data/processed/rfm.parquet`: bảng RFM theo khách hàng.
- `data/processed/basket.parquet`: dữ liệu giỏ hàng dạng long-format.
- `data/processed/cluster_input.parquet`: đầu vào cho clustering/modeling.
- `data/processed/timeseries_monthly.csv`: chuỗi thời gian doanh thu theo tháng.

---

## 3) Cấu trúc thư mục

```text
BTL-KPDL/
|-- README.md
|-- requirements.txt
|-- configs/
|   `-- params.yaml
|-- data/
|   |-- raw/
|   `-- processed/
|-- docs/
|   |-- data_dictionary.md
|   `-- eda_insights.md
|-- notebooks/
|   |-- 01_eda.ipynb
|   |-- 02_preprocess_feature.ipynb
|   |-- 03_association_rules.ipynb
|   |-- 04_clustering.ipynb
|   |-- 05_classification.ipynb
|   |-- 06_forecasting.ipynb
|   `-- 07_evaluation_report.ipynb
|-- scripts/
|   |-- run_pipeline.py
|   |-- run_association.py
|   |-- run_clustering.py
|   |-- run_modeling.py
|   `-- run_forecasting.py
|-- src/
|   |-- data/
|   |-- features/
|   |-- mining/
|   |-- models/
|   |-- evaluation/
|   |-- visualization/
|   `-- utils/
|-- outputs/
|   |-- figures/
|   |-- models/
|   `-- tables/
`-- app/
    `-- streamlit_app.py
```

---

## 4) Cài đặt môi trường

### 4.1 Cài đặt bằng Conda (khuyến nghị)

```bash
conda create -n BTL_KPDL_env python=3.10 -y
conda activate BTL_KPDL_env
pip install -r requirements.txt
```

### 4.2 Cài đặt bằng venv (nếu không dùng Conda)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 4.3 Kiểm tra nhanh

```bash
python -c "import pandas, sklearn, mlxtend, statsmodels; print('OK')"
```

---

## 5) Chạy nhanh toàn bộ pipeline

Thực hiện theo thứ tự sau để đảm bảo dữ liệu trung gian được tạo đúng:

```bash
# 1) Tiền xử lý + tạo feature
python scripts/run_pipeline.py

# 2) Market Basket Analysis
python scripts/run_association.py

# 3) Customer Segmentation
python scripts/run_clustering.py

# 4) Supervised Modeling
python scripts/run_modeling.py

# 5) Forecasting doanh thu
python scripts/run_forecasting.py
```

Luồng dữ liệu tổng quan:

```text
train.csv
  -> cleaned.parquet
    -> rfm.parquet + basket.parquet + timeseries_monthly.csv
      -> cluster_input.parquet
        -> clustering + classification outputs
      -> forecasting outputs
```

---

## 6) Mô tả notebooks (Notebook-per-task)

| Thứ tự | Notebook | Mục tiêu | Output chính |
|---:|---|---|---|
| 01 | `notebooks/01_eda.ipynb` | Khám phá dữ liệu, thống kê mô tả, kiểm tra giá trị thiếu/ngoại lệ | Insights ban đầu trong `docs/` |
| 02 | `notebooks/02_preprocess_feature.ipynb` | Làm sạch dữ liệu, tạo RFM, basket, time series | Tệp trong `data/processed/` |
| 03 | `notebooks/03_association_rules.ipynb` | Tìm frequent itemsets và association rules | `outputs/tables/top_products.csv`, `outputs/tables/top_rules.csv` |
| 04 | `notebooks/04_clustering.ipynb` | Phân khúc khách hàng bằng RFM + KMeans | `outputs/tables/cluster_stats.csv`, `outputs/models/kmeans.pkl` |
| 05 | `notebooks/05_classification.ipynb` | Huấn luyện mô hình phân loại segment | `outputs/models/best_model.pkl`, `outputs/tables/model_metrics.csv` |
| 06 | `notebooks/06_forecasting.ipynb` | Dự báo doanh thu theo tháng (Naive/ARIMA/Prophet nếu có) | `outputs/tables/forecast_metrics.csv` |
| 07 | `notebooks/07_evaluation_report.ipynb` | Tổng hợp kết quả và khuyến nghị kinh doanh | Báo cáo đánh giá tổng hợp |

---

## 7) Mô tả scripts CLI

| Script | Đầu vào | Chức năng | Đầu ra |
|---|---|---|---|
| `scripts/run_pipeline.py` | `data/raw/train.csv` | Load, clean, feature engineering (RFM/basket/time series) | `cleaned.parquet`, `rfm.parquet`, `basket.parquet`, `cluster_input.parquet`, `timeseries_monthly.csv` |
| `scripts/run_association.py` | `data/processed/cleaned.parquet` | FP-Growth + Association Rules | `outputs/tables/top_products.csv`, `outputs/tables/top_rules.csv`, biểu đồ liên quan |
| `scripts/run_clustering.py` | `data/processed/cleaned.parquet` | RFM scaling, Elbow/Silhouette, KMeans, gán nhãn segment | `outputs/tables/cluster_stats.csv`, `outputs/tables/rfm_clustered.csv`, `outputs/models/kmeans.pkl` |
| `scripts/run_modeling.py` | `data/processed/cluster_input.parquet` | Train/evaluate nhiều mô hình classification, chọn best model | `outputs/models/best_model.pkl`, `outputs/tables/model_metrics.csv`, `outputs/figures/confusion_matrix.png` |
| `scripts/run_forecasting.py` | `data/processed/timeseries_monthly.csv` | Dự báo chuỗi thời gian (Naive, ARIMA, Prophet nếu có) | `outputs/tables/forecast_metrics.csv`, `outputs/figures/forecast_plot.png`, `outputs/figures/actual_vs_pred.png` |

---

## 8) Kiến trúc mã nguồn `src`

| Module | Nội dung |
|---|---|
| `src/data/` | Nạp dữ liệu, thông tin cơ bản, làm sạch (`loader.py`, `cleaner.py`) |
| `src/features/` | Tạo đặc trưng RFM, basket matrix, đặc trưng thời gian (`rfm.py`, `basket.py`, `time_features.py`) |
| `src/mining/` | Thuật toán Association Rules và Clustering (`association.py`, `clustering.py`) |
| `src/models/` | Mô hình dự báo và phân loại (`forecasting.py`, `supervised.py`) |
| `src/evaluation/` | Metric, evaluator, report (`metrics.py`, `evaluator.py`, `report.py`) |
| `src/visualization/` | Hàm vẽ biểu đồ dùng lại (`plots.py`) |
| `src/utils/` | Cấu hình và logging (`config.py`, `logger.py`) |

Kiến trúc này giúp:

- Tách biệt rõ logic nghiệp vụ và notebook trình bày.
- Tái sử dụng code giữa notebook và script.
- Dễ mở rộng khi cần bổ sung mô hình mới.

---

## 9) Cấu hình tham số

Tất cả tham số được quản lý tại `configs/params.yaml`.

Ví dụ các nhóm tham số chính:

- `seed`: random seed toàn dự án.
- `paths`: đường dẫn raw/processed/output.
- `association`: `min_support`, `min_confidence`, `min_lift`.
- `clustering`: `n_clusters`.
- `modeling`: `target`, `algorithms`, `test_size`, `selection_criterion`.
- `forecasting`: `date_col`, `value_col`, `test_periods`, `forecast_horizon`, `arima_order`.

Khi đổi yêu cầu bài toán, ưu tiên chỉnh tham số trong file cấu hình thay vì hard-code trong script.

---

## 10) Output và artefacts

Sau khi chạy pipeline, các kết quả được lưu ở:

```text
outputs/
|-- figures/
|   |-- top_products.png
|   |-- rules_support_confidence.png
|   |-- elbow.png
|   |-- cluster_scatter.png
|   |-- revenue_by_cluster.png
|   |-- confusion_matrix.png
|   |-- feature_importance.png
|   |-- forecast_plot.png
|   `-- actual_vs_pred.png
|-- models/
|   |-- kmeans.pkl
|   `-- best_model.pkl
`-- tables/
    |-- top_products.csv
    |-- top_rules.csv
    |-- cluster_stats.csv
    |-- rfm_clustered.csv
    |-- model_metrics.csv
    `-- forecast_metrics.csv
```

---

## 11) Chạy dashboard Streamlit

```bash
streamlit run app/streamlit_app.py
```

Lưu ý:

- Dashboard đọc kết quả từ thư mục `outputs/`.
- Nếu chạy trên máy khác, hãy kiểm tra biến `base` trong `app/streamlit_app.py` để bảo đảm đường dẫn phù hợp với môi trường hiện tại.

---

## 12) Kết quả mong đợi và hướng mở rộng

### 12.1 Kết quả mong đợi

- Xác định tập sản phẩm có xác suất mua kèm cao (rules có lift tốt).
- Hình thành nhóm khách hàng có ý nghĩa kinh doanh (VIP, Loyal, Potential, Lost...).
- Chọn được mô hình phân loại tốt nhất dựa trên metric đã định nghĩa.
- Có baseline dự báo doanh thu để hỗ trợ lập kế hoạch.

### 12.2 Hướng mở rộng

- Bổ sung hyperparameter tuning (GridSearch/Optuna) cho modeling.
- Thêm SHAP/Permutation Importance để giải thích mô hình sâu hơn.
- Nâng cấp forecasting với nhiều mô hình hơn (SARIMA/ETS/XGBoost Time Features).
- Đồng bộ dashboard với đường dẫn tương đối và bộ lọc nâng cao.
- Tích hợp CI để tự động chạy script và kiểm tra output.

---

## 13) Tác giả

Dự án được thực hiện bởi nhóm 11 BTL môn Khai phá dữ liệu.


---

## 14) Tổng hợp công việc theo nhánh

Mục này tổng hợp theo lịch sử commit các nhánh phát triển trong repo, không gắn tên người thực hiện.

### Nhánh `huy`

1. Week 1: Data cleaning + EDA + khởi tạo bộ khung dự án.
2. Week 2: Preprocessing + feature engineering + chuẩn hóa `.gitignore`.

Phần thay đổi tiêu biểu:

- Hệ thống tiền xử lý và tạo feature: `scripts/run_pipeline.py`, `src/features/rfm.py`, `src/features/basket.py`, `src/features/time_features.py`.
- Notebook EDA và preprocess: `notebooks/01_eda.ipynb`, `notebooks/02_preprocess_feature.ipynb`.
- Tài liệu nền tảng: `docs/data_dictionary.md`, `docs/eda_insights.md`.

### Nhánh `tung`

1. Hoàn thiện phần Association Rules (Market Basket).
2. Hoàn thiện phần Clustering (Customer Segmentation).
3. Cập nhật README ở giai đoạn giữa.

Phần thay đổi tiêu biểu:

- Pipeline và thuật toán: `scripts/run_association.py`, `scripts/run_clustering.py`, `src/mining/association.py`, `src/mining/clustering.py`.
- Cấu hình liên quan: `configs/params.yaml`.
- Kết quả sinh ra: `outputs/tables/top_products.csv`, `outputs/tables/top_rules.csv`, `outputs/tables/cluster_stats.csv`, `outputs/tables/rfm_clustered.csv`, `outputs/models/kmeans.pkl`.
- Biểu đồ và notebook liên quan: `notebooks/03_association_rules.ipynb`, `notebooks/04_clustering.ipynb`, các file trong `outputs/figures/`.

### Nhánh `an`

1. Bổ sung phần ML classification và forecasting.
2. Hoàn thiện notebook/report tổng kết cuối kỳ.

Phần thay đổi tiêu biểu:

- Modeling + forecasting: `scripts/run_modeling.py`, `scripts/run_forecasting.py`, `src/models/supervised.py`, `src/models/forecasting.py`.
- Metrics và đánh giá: `src/evaluation/metrics.py`, `src/evaluation/report.py`.
- Notebook giai đoạn cuối: `notebooks/05_classification.ipynb`, `notebooks/06_forecasting.ipynb`, `notebooks/07_evaluation_report.ipynb`.
- Đồng bộ tham số/cách chạy: `configs/params.yaml`, `README.md`.

### Tiến trình tổng quan

1. Khởi tạo và đặt nền móng: cleaning, EDA, cấu trúc dự án.
2. Mở rộng preprocessing và feature engineering.
3. Triển khai Association Rules + Clustering và sinh artefacts.
4. Bổ sung Classification + Forecasting.
5. Chốt báo cáo đánh giá tổng hợp.
