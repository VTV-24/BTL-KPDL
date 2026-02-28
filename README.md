# Superstore Data Mining Project

**Phân tích doanh số siêu thị** – Bài tập lớn môn Khai phá dữ liệu

## Team members
- Member 1: Data + Feature Engineering
- Member 2: Mining + Clustering  
- Member 3: Modeling + Forecasting

---

## Setup

**1. Tạo môi trường:**
```bash
conda create -n BTL_KPDL_env python=3.10 -y
conda activate BTL_KPDL_env
```

**2. Cài dependencies:**
```bash
pip install -r requirements.txt
```

**Thư viện chính:** pandas, numpy, scikit-learn, mlxtend, matplotlib, seaborn, joblib

---

## Dataset

**Nguồn:** Superstore Sales Dataset (Kaggle)

**Đường dẫn:**
- Raw: `data/raw/train.csv` (9800 rows × 18 columns)
- Processed: `data/processed/cleaned.parquet`

**Các cột chính:** Order ID, Customer ID, Product Name, Sub-Category, Sales, Order Date, Region, Segment...

---

## Project Structure

```
BTL-KPDL/
├── configs/
│   └── params.yaml           # Tham số cấu hình (seed, n_clusters, min_support...)
├── data/
│   ├── raw/                  # Dữ liệu gốc
│   └── processed/            # Dữ liệu đã xử lý
├── docs/                     # Tài liệu
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_association_rules.ipynb  ✅
│   ├── 04_clustering.ipynb         ✅
│   ├── 05_classification.ipynb
│   ├── 06_forecasting.ipynb
│   └── 07_evaluation_report.ipynb
├── outputs/
│   ├── figures/              # Biểu đồ
│   ├── models/               # Models saved
│   └── tables/               # Bảng kết quả CSV
├── scripts/
│   ├── run_association.py    ✅
│   ├── run_clustering.py     ✅
│   ├── run_forecasting.py
│   └── run_modeling.py
├── src/
│   ├── data/                 # Data loading, cleaning
│   ├── features/             # Feature engineering (RFM, basket...)
│   ├── mining/               # Association rules, Clustering
│   ├── models/               # Forecasting, Supervised
│   ├── evaluation/           # Metrics, reports
│   ├── utils/                # Config, logger
│   └── visualization/        # Plots
└── app/
    └── streamlit_app.py      # Web app
```

---

## ✅ Completed Tasks

### 1. Association Rules (Market Basket Analysis)

**Mục tiêu:** Tìm itemsets phổ biến và luật kết hợp giữa các Sub-Category

**Chạy pipeline:**
```bash
python scripts/run_association.py
```

**Hoặc chạy notebook:** `notebooks/03_association_rules.ipynb`

**Kết quả:**
- 67 frequent itemsets (min_support = 0.01)
- 21 association rules (min_confidence = 0.1)

**Outputs:**
| File | Mô tả |
|------|-------|
| `outputs/tables/top_products.csv` | Top sản phẩm phổ biến |
| `outputs/tables/top_rules.csv` | Luật kết hợp (antecedent → consequent) |
| `outputs/figures/top_products.png` | Biểu đồ top products |
| `outputs/figures/rules_support_confidence.png` | Support vs Confidence scatter |

**Top Rules Example:**
| Antecedent | Consequent | Support | Confidence | Lift |
|------------|------------|---------|------------|------|
| Labels | Binders | 0.018 | 0.500 | 1.48 |
| Fasteners | Binders | 0.016 | 0.413 | 1.22 |

---

### 2. Clustering (Customer Segmentation – RFM + KMeans)

**Mục tiêu:** Phân khúc 793 khách hàng dựa trên hành vi mua hàng (Recency, Frequency, Monetary)

**Chạy pipeline:**
```bash
python scripts/run_clustering.py
```

**Hoặc chạy notebook:** `notebooks/04_clustering.ipynb`

**Kết quả phân khúc (k=4):**

| Segment | Count | % | Recency | Frequency | Monetary avg | Revenue |
|---------|-------|---|---------|-----------|--------------|---------|
|  VIP | 128 | 16.1% | 108 days | 7.9 orders | $6,407 | $820K |
|  Loyal | 220 | 27.7% | 61 days | 8.6 orders | $2,834 | $623K |
|  Potential | 299 | 37.7% | 75 days | 4.8 orders | $1,620 | $484K |
|  Lost | 146 | 18.4% | 366 days | 4.0 orders | $1,373 | $200K |

**Outputs:**
| File | Mô tả |
|------|-------|
| `outputs/tables/cluster_stats.csv` | Thống kê theo cluster |
| `outputs/tables/rfm_clustered.csv` | RFM data + Segment labels |
| `outputs/models/kmeans.pkl` | Model KMeans trained |
| `outputs/figures/elbow.png` | Elbow + Silhouette chart |
| `outputs/figures/cluster_scatter.png` | Frequency vs Monetary scatter |
| `outputs/figures/revenue_by_cluster.png` | Revenue by segment |

**Marketing Insights:**
- **VIP:** Ưu đãi riêng, loyalty program cao cấp, early access
- **Loyal:** Cross-sell premium, reward points multiplier
- **Potential:** Upsell bundles, email nurture campaign
- **Lost:** Win-back campaign, discount lớn, survey

---

##  Pending Tasks

- [ ] Classification (dự đoán segment của khách hàng mới)
- [ ] Forecasting (dự báo doanh số theo thời gian)
- [ ] Evaluation Report (tổng hợp kết quả)

---

## Configuration

File: `configs/params.yaml`

```yaml
seed: 42

association:
  min_support: 0.01
  min_confidence: 0.1
  top_n: 10

clustering:
  n_clusters: 4
```

---

## Quick Start

```bash
# 1. Activate environment
conda activate BTL_KPDL_env

# 2. Run Association Rules
python scripts/run_association.py

# 3. Run Clustering
python scripts/run_clustering.py

# 4. Check outputs
ls outputs/tables/
ls outputs/figures/
ls outputs/models/
```
