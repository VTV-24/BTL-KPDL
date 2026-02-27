"""
scripts/run_association.py
==========================
Chạy pipeline Association Rules từ CLI.
Output:
  - outputs/tables/top_products.csv
  - outputs/tables/top_rules.csv
  - outputs/figures/top_products.png
  - outputs/figures/rules_support_confidence.png
"""

import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ── đảm bảo import src từ project root ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.config import load_config
from src.features.basket import build_basket_matrix, build_basket_subcategory
from src.mining.association import (
    basket_summary,
    top_products,
    find_frequent_itemsets,
    generate_rules,
    filter_top_rules,
    rules_to_csv_friendly,
)

warnings.filterwarnings("ignore")


def main():
    # ── 1. Load config ──────────────────────────────────────────────
    cfg = load_config(os.path.join(ROOT, "configs", "params.yaml"))
    assoc_cfg = cfg.get("association", {})
    min_support = assoc_cfg.get("min_support", 0.02)
    min_confidence = assoc_cfg.get("min_confidence", 0.4)
    min_lift = assoc_cfg.get("min_lift", 1.1)

    # ── 2. Load cleaned data ────────────────────────────────────────
    processed_dir = os.path.join(ROOT, cfg["paths"]["processed_dir"])
    cleaned_path = os.path.join(processed_dir, "cleaned.parquet")
    df = pd.read_parquet(cleaned_path)
    print(f"[INFO] Loaded cleaned data: {df.shape}")

    # ── 3. Build basket matrix (Product Name for top products) ─────
    basket_product = build_basket_matrix(df, item_col="Product Name")
    summary_product = basket_summary(basket_product)
    print(f"[INFO] Product-level: Orders={summary_product['n_orders']}, "
          f"Products={summary_product['n_products']}, Sparsity={summary_product['sparsity']}")

    # ── 4. Top sản phẩm bán chạy ───────────────────────────────────
    df_top = top_products(basket_product, top_n=20)
    print(f"[INFO] Top 20 products computed")

    # ── 5. Build basket by Sub-Category (for association rules) ────
    basket = build_basket_subcategory(df)
    summary = basket_summary(basket)
    print(f"[INFO] Sub-Category level: Orders={summary['n_orders']}, "
          f"Sub-Categories={summary['n_products']}, Sparsity={summary['sparsity']}")

    # ── 6. Frequent itemsets (FP-Growth) ────────────────────────────
    freq = find_frequent_itemsets(basket, min_support=min_support, algorithm="fpgrowth")
    print(f"[INFO] Frequent itemsets: {len(freq)}")

    # ── 7. Rules ────────────────────────────────────────────────────
    rules = generate_rules(freq, min_confidence=min_confidence, min_lift=1.0)
    print(f"[INFO] All rules: {len(rules)}")

    top_rules = filter_top_rules(rules, min_lift=min_lift, top_n=30)
    print(f"[INFO] Top rules (lift >= {min_lift}): {len(top_rules)}")

    # ── 8. Tạo thư mục output ──────────────────────────────────────
    output_dir = os.path.join(ROOT, cfg["paths"].get("output_dir", "outputs"))
    tables_dir = os.path.join(output_dir, "tables")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # ── 8. Export CSV ───────────────────────────────────────────────
    df_top.to_csv(os.path.join(tables_dir, "top_products.csv"), index=False)
    print(f"[SAVED] {tables_dir}/top_products.csv")

    top_rules_csv = rules_to_csv_friendly(top_rules)
    top_rules_csv.to_csv(os.path.join(tables_dir, "top_rules.csv"), index=False)
    print(f"[SAVED] {tables_dir}/top_rules.csv")

    # ── 9. Figures ──────────────────────────────────────────────────
    # 9a. Top products bar chart
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_top, x="order_count", y="Product Name", palette="viridis", ax=ax1)
    ax1.set_title("Top 20 sản phẩm bán chạy (theo số đơn hàng)", fontsize=14)
    ax1.set_xlabel("Số đơn hàng")
    ax1.set_ylabel("")
    plt.tight_layout()
    fig1.savefig(os.path.join(figures_dir, "top_products.png"), dpi=150)
    plt.close(fig1)
    print(f"[SAVED] {figures_dir}/top_products.png")

    # 9b. Rules scatter: support vs confidence, size = lift
    if not top_rules.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        scatter = ax2.scatter(
            top_rules["support"],
            top_rules["confidence"],
            s=top_rules["lift"] * 40,
            c=top_rules["lift"],
            cmap="YlOrRd",
            alpha=0.75,
            edgecolors="black",
            linewidths=0.5,
        )
        plt.colorbar(scatter, label="Lift")
        ax2.set_title("Association Rules – Support vs Confidence (size = lift)", fontsize=13)
        ax2.set_xlabel("Support")
        ax2.set_ylabel("Confidence")
        plt.tight_layout()
        fig2.savefig(os.path.join(figures_dir, "rules_support_confidence.png"), dpi=150)
        plt.close(fig2)
        print(f"[SAVED] {figures_dir}/rules_support_confidence.png")
    else:
        print("[WARN] No rules to plot (try lowering min_support / min_confidence)")

    print("\n[DONE] Association Rules pipeline complete.")


if __name__ == "__main__":
    main()
