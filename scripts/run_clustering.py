"""
scripts/run_clustering.py
==========================
Chạy pipeline Customer Segmentation từ CLI.
Output:
  - outputs/tables/cluster_stats.csv
  - outputs/models/kmeans.pkl
  - outputs/figures/elbow.png
  - outputs/figures/cluster_scatter.png
  - outputs/figures/revenue_by_cluster.png
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# ── đảm bảo import src từ project root ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.features.rfm import build_rfm
from src.mining.clustering import (
    cap_outliers_iqr,
    scale_rfm,
    elbow_scores,
    train_kmeans,
    assign_clusters,
    cluster_stats,
    label_clusters,
    map_segment_names,
    save_model,
)

warnings.filterwarnings("ignore")


def main():
    # ── 1. Load config ──────────────────────────────────────────────
    cfg = load_config(os.path.join(ROOT, "configs", "params.yaml"))
    seed = cfg.get("seed", 42)
    n_clusters = cfg.get("clustering", {}).get("n_clusters", 4)

    # ── 2. Load cleaned data & build RFM ────────────────────────────
    processed_dir = os.path.join(ROOT, cfg["paths"]["processed_dir"])
    cleaned_path = os.path.join(processed_dir, "cleaned.parquet")
    df = pd.read_parquet(cleaned_path)
    print(f"[INFO] Loaded cleaned data: {df.shape}")

    rfm = build_rfm(df)
    print(f"[INFO] RFM shape: {rfm.shape}")

    # ── 3. Cap outliers ─────────────────────────────────────────────
    rfm_capped = cap_outliers_iqr(rfm, cols=["Recency", "Frequency", "Monetary"])
    print("[INFO] Outliers capped (IQR method)")

    # ── 4. Scale RFM ────────────────────────────────────────────────
    rfm_scaled, scaler = scale_rfm(rfm_capped, cols=["Recency", "Frequency", "Monetary"])
    X = rfm_scaled[["Recency", "Frequency", "Monetary"]].values
    print("[INFO] RFM scaled with StandardScaler")

    # ── 5. Elbow & Silhouette ───────────────────────────────────────
    scores = elbow_scores(X, k_range=range(2, 11), random_state=seed)
    print(f"[INFO] Elbow scores computed for k=2..10")

    # ── 6. Train KMeans ─────────────────────────────────────────────
    km = train_kmeans(X, n_clusters=n_clusters, random_state=seed)
    labels = km.labels_
    print(f"[INFO] KMeans trained with k={n_clusters}")

    # ── 7. Assign clusters to RFM ───────────────────────────────────
    rfm_clustered = assign_clusters(rfm_capped, labels)

    # ── 8. Cluster stats ────────────────────────────────────────────
    stats = cluster_stats(rfm_clustered)
    stats = label_clusters(stats)
    print(f"[INFO] Cluster stats computed")
    print(stats.to_string(index=False))

    # ── 9. Map segment names back ───────────────────────────────────
    rfm_final = map_segment_names(rfm_clustered, stats)

    # ── 10. Tạo thư mục output ─────────────────────────────────────
    output_dir = os.path.join(ROOT, cfg["paths"].get("output_dir", "outputs"))
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # ── 11. Export CSV ──────────────────────────────────────────────
    stats.to_csv(os.path.join(tables_dir, "cluster_stats.csv"), index=False)
    print(f"[SAVED] {tables_dir}/cluster_stats.csv")

    # Save RFM with clusters for later use
    rfm_final.to_csv(os.path.join(tables_dir, "rfm_clustered.csv"), index=False)
    print(f"[SAVED] {tables_dir}/rfm_clustered.csv")

    # ── 12. Save model ──────────────────────────────────────────────
    save_model(km, os.path.join(models_dir, "kmeans.pkl"))
    print(f"[SAVED] {models_dir}/kmeans.pkl")

    # ── 13. Figures ─────────────────────────────────────────────────
    # 13a. Elbow plot
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Inertia (Elbow)
    axes1[0].plot(scores["k"], scores["inertia"], "bo-", linewidth=2, markersize=8)
    axes1[0].axvline(x=n_clusters, color="r", linestyle="--", label=f"k={n_clusters}")
    axes1[0].set_xlabel("Số cluster (k)")
    axes1[0].set_ylabel("Inertia (SSE)")
    axes1[0].set_title("Elbow Method")
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.3)
    
    # Silhouette
    axes1[1].plot(scores["k"], scores["silhouette"], "go-", linewidth=2, markersize=8)
    axes1[1].axvline(x=n_clusters, color="r", linestyle="--", label=f"k={n_clusters}")
    axes1[1].set_xlabel("Số cluster (k)")
    axes1[1].set_ylabel("Silhouette Score")
    axes1[1].set_title("Silhouette Analysis")
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig1.savefig(os.path.join(figures_dir, "elbow.png"), dpi=150)
    plt.close(fig1)
    print(f"[SAVED] {figures_dir}/elbow.png")

    # 13b. Cluster scatter (Frequency vs Monetary, color by cluster)
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    scatter = ax2.scatter(
        rfm_final["Frequency"],
        rfm_final["Monetary"],
        c=rfm_final["Cluster"],
        cmap="viridis",
        alpha=0.6,
        s=50,
        edgecolors="white",
        linewidths=0.5,
    )
    plt.colorbar(scatter, label="Cluster")
    
    # Add cluster centers (on original scale)
    centers_scaled = km.cluster_centers_
    # Inverse transform to get original scale
    centers_original = scaler.inverse_transform(centers_scaled)
    ax2.scatter(
        centers_original[:, 1],  # Frequency
        centers_original[:, 2],  # Monetary
        c="red",
        marker="X",
        s=200,
        edgecolors="black",
        linewidths=2,
        label="Centroids",
    )
    ax2.set_xlabel("Frequency (số đơn hàng)")
    ax2.set_ylabel("Monetary (tổng chi tiêu $)")
    ax2.set_title("Customer Clusters (Frequency vs Monetary)")
    ax2.legend()
    plt.tight_layout()
    fig2.savefig(os.path.join(figures_dir, "cluster_scatter.png"), dpi=150)
    plt.close(fig2)
    print(f"[SAVED] {figures_dir}/cluster_scatter.png")

    # 13c. Revenue by cluster (stacked bar)
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    
    # Count per segment
    stats_sorted = stats.sort_values("Monetary_sum", ascending=True)
    colors = sns.color_palette("viridis", n_colors=len(stats_sorted))
    
    axes3[0].barh(stats_sorted["Segment"], stats_sorted["Count"], color=colors)
    axes3[0].set_xlabel("Số khách hàng")
    axes3[0].set_title("Số lượng khách hàng theo Segment")
    for i, (cnt, pct) in enumerate(zip(stats_sorted["Count"], stats_sorted["Pct"])):
        axes3[0].text(cnt + 5, i, f"{cnt} ({pct}%)", va="center", fontsize=10)
    
    # Revenue per segment
    axes3[1].barh(stats_sorted["Segment"], stats_sorted["Monetary_sum"], color=colors)
    axes3[1].set_xlabel("Tổng doanh thu ($)")
    axes3[1].set_title("Doanh thu theo Segment")
    for i, rev in enumerate(stats_sorted["Monetary_sum"]):
        axes3[1].text(rev + 1000, i, f"${rev:,.0f}", va="center", fontsize=10)
    
    plt.tight_layout()
    fig3.savefig(os.path.join(figures_dir, "revenue_by_cluster.png"), dpi=150)
    plt.close(fig3)
    print(f"[SAVED] {figures_dir}/revenue_by_cluster.png")

    print("\n[DONE] Clustering pipeline complete.")


if __name__ == "__main__":
    main()
