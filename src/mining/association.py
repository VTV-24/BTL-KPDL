"""
Association Rules – Market Basket Analysis
===========================================
Sử dụng mlxtend (Apriori / FP-Growth) để tìm frequent itemsets & sinh rules.
"""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# ------------------------------------------------------------------
# 1. Thống kê tổng quan basket
# ------------------------------------------------------------------
def basket_summary(basket_matrix: pd.DataFrame) -> dict:
    """
    Nhận basket matrix (Order ID × Product, 0/1)
    Trả về dict: n_orders, n_products, sparsity
    """
    # Bỏ cột Order ID nếu có
    mat = basket_matrix.drop(columns=["Order ID"], errors="ignore")

    n_orders = mat.shape[0]
    n_products = mat.shape[1]
    n_cells = n_orders * n_products
    n_ones = mat.sum().sum()
    sparsity = 1 - (n_ones / n_cells) if n_cells > 0 else 0.0

    return {
        "n_orders": n_orders,
        "n_products": n_products,
        "sparsity": round(sparsity, 4),
    }


# ------------------------------------------------------------------
# 2. Top sản phẩm bán chạy (theo số đơn hàng xuất hiện)
# ------------------------------------------------------------------
def top_products(basket_matrix: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Trả DataFrame: Product Name | order_count  (sorted desc)
    """
    mat = basket_matrix.drop(columns=["Order ID"], errors="ignore")
    counts = mat.sum().sort_values(ascending=False).head(top_n)
    df_top = counts.reset_index()
    df_top.columns = ["Product Name", "order_count"]
    return df_top


# ------------------------------------------------------------------
# 3. Tìm frequent itemsets
# ------------------------------------------------------------------
def find_frequent_itemsets(
    basket_matrix: pd.DataFrame,
    min_support: float = 0.02,
    algorithm: str = "fpgrowth",
) -> pd.DataFrame:
    """
    algorithm: 'apriori' hoặc 'fpgrowth'
    Trả về DataFrame frequent itemsets (itemsets, support)
    """
    mat = basket_matrix.drop(columns=["Order ID"], errors="ignore").astype(bool)

    if algorithm == "apriori":
        freq = apriori(mat, min_support=min_support, use_colnames=True)
    else:
        freq = fpgrowth(mat, min_support=min_support, use_colnames=True)
    freq.sort_values("support", ascending=False, inplace=True)
    freq.reset_index(drop=True, inplace=True)
    return freq



# ------------------------------------------------------------------
# 4. Sinh association rules
# ------------------------------------------------------------------
def generate_rules(
    freq_itemsets: pd.DataFrame,
    min_confidence: float = 0.4,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    """
    Sinh rules từ frequent itemsets, lọc theo confidence & lift.
    """
    if freq_itemsets.empty:
        return pd.DataFrame()

    rules = association_rules(
        freq_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(freq_itemsets)
    )
    rules = rules[rules["lift"] >= min_lift]
    rules.sort_values("lift", ascending=False, inplace=True)
    rules.reset_index(drop=True, inplace=True)
    return rules


# ------------------------------------------------------------------
# 5. Lọc top rules (tiện dùng cho notebook / script)
# ------------------------------------------------------------------
def filter_top_rules(
    rules: pd.DataFrame,
    min_lift: float = 1.1,
    top_n: int = 30,
) -> pd.DataFrame:
    """
    Lọc rules có lift >= min_lift và lấy top_n rules theo lift giảm dần.
    """
    filtered = rules[rules["lift"] >= min_lift].copy()
    filtered.sort_values("lift", ascending=False, inplace=True)
    filtered = filtered.head(top_n).reset_index(drop=True)
    return filtered


# ------------------------------------------------------------------
# 6. Helper – chuyển frozenset → str cho export CSV
# ------------------------------------------------------------------
def rules_to_csv_friendly(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Chuyển cột antecedents / consequents từ frozenset → str
    để export CSV dễ đọc.
    """
    out = rules.copy()
    if "antecedents" in out.columns:
        out["antecedents"] = out["antecedents"].apply(
            lambda s: ", ".join(sorted(s))
        )
    if "consequents" in out.columns:
        out["consequents"] = out["consequents"].apply(
            lambda s: ", ".join(sorted(s))
        )
    return out
