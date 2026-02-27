"""
Customer Segmentation – KMeans Clustering on RFM
=================================================
Phân cụm khách hàng dựa trên Recency, Frequency, Monetary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib


# ------------------------------------------------------------------
# 1. Xử lý outliers (IQR capping)
# ------------------------------------------------------------------
def cap_outliers_iqr(df: pd.DataFrame, cols: list, factor: float = 1.5) -> pd.DataFrame:
    """
    Cap outliers using IQR method.
    Values beyond Q1 - factor*IQR or Q3 + factor*IQR are capped.
    """
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


# ------------------------------------------------------------------
# 2. Scale dữ liệu
# ------------------------------------------------------------------
def scale_rfm(rfm: pd.DataFrame, cols: list = None) -> tuple:
    """
    StandardScaler trên các cột RFM.
    Returns: (rfm_scaled DataFrame, scaler object)
    """
    if cols is None:
        cols = ["Recency", "Frequency", "Monetary"]
    
    scaler = StandardScaler()
    rfm_scaled = rfm.copy()
    rfm_scaled[cols] = scaler.fit_transform(rfm[cols])
    return rfm_scaled, scaler


# ------------------------------------------------------------------
# 3. Elbow method (tìm k tối ưu)
# ------------------------------------------------------------------
def elbow_scores(X: np.ndarray, k_range: range = range(2, 11), random_state: int = 42) -> dict:
    """
    Tính inertia và silhouette cho các giá trị k.
    Returns: dict với keys 'k', 'inertia', 'silhouette'
    """
    inertias = []
    silhouettes = []
    ks = list(k_range)
    
    for k in ks:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, labels) if k > 1 else 0
        silhouettes.append(sil)
    
    return {
        "k": ks,
        "inertia": inertias,
        "silhouette": silhouettes,
    }


# ------------------------------------------------------------------
# 4. Huấn luyện KMeans
# ------------------------------------------------------------------
def train_kmeans(X: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> KMeans:
    """
    Huấn luyện KMeans với số cluster cho trước.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    km.fit(X)
    return km


# ------------------------------------------------------------------
# 5. Gán nhãn cluster vào RFM DataFrame
# ------------------------------------------------------------------
def assign_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """
    Gán cột 'Cluster' vào DataFrame RFM.
    """
    rfm = rfm.copy()
    rfm["Cluster"] = labels
    return rfm


# ------------------------------------------------------------------
# 6. Thống kê theo cluster
# ------------------------------------------------------------------
def cluster_stats(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Tính thống kê trung bình và count theo cluster.
    """
    stats = rfm.groupby("Cluster").agg({
        "Customer ID": "count",
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": ["mean", "sum"],
    }).round(2)
    
    stats.columns = ["Count", "Recency_mean", "Frequency_mean", "Monetary_mean", "Monetary_sum"]
    stats = stats.reset_index()
    stats["Pct"] = (stats["Count"] / stats["Count"].sum() * 100).round(1)
    return stats


# ------------------------------------------------------------------
# 7. Đặt tên nhóm khách hàng
# ------------------------------------------------------------------
def label_clusters(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Đặt tên nhóm dựa trên đặc điểm RFM:
    - Low Recency + High Frequency + High Monetary → VIP
    - Low Recency + High Frequency + Low Monetary → Loyal
    - High Recency + Low Frequency → Lost/At-Risk
    - etc.
    
    Sắp xếp theo Monetary_mean descending để gán nhãn:
    """
    stats = stats.copy()
    
    # Sắp xếp theo Monetary_mean giảm dần
    stats = stats.sort_values("Monetary_mean", ascending=False).reset_index(drop=True)
    
    # Gán tên dựa trên thứ tự
    n_clusters = len(stats)
    if n_clusters == 4:
        labels = ["VIP", "Loyal", "Potential", "Lost"]
    elif n_clusters == 3:
        labels = ["VIP", "Regular", "Lost"]
    elif n_clusters == 5:
        labels = ["VIP", "Loyal", "Potential", "At-Risk", "Lost"]
    else:
        labels = [f"Segment_{i}" for i in range(n_clusters)]
    
    stats["Segment"] = labels
    return stats


# ------------------------------------------------------------------
# 8. Map segment name back to RFM
# ------------------------------------------------------------------
def map_segment_names(rfm: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Map Segment name từ stats vào rfm DataFrame.
    """
    rfm = rfm.copy()
    cluster_to_segment = dict(zip(stats["Cluster"], stats["Segment"]))
    rfm["Segment"] = rfm["Cluster"].map(cluster_to_segment)
    return rfm


# ------------------------------------------------------------------
# 9. Save / Load model
# ------------------------------------------------------------------
def save_model(model, path: str):
    """Lưu model bằng joblib."""
    joblib.dump(model, path)


def load_model(path: str):
    """Load model từ file."""
    return joblib.load(path)
