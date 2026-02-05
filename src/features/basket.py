import pandas as pd


# =====================================================
# LONG FORMAT  (FP-Growth / mlxtend.frequent_patterns.fpgrowth)
# =====================================================
def build_basket_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long transaction format

    Output:
    Order ID | Product Name
    """

    basket = (
        df[["Order ID", "Product Name"]]
        .drop_duplicates()
        .sort_values("Order ID")
        .reset_index(drop=True)
    )

    return basket


# =====================================================
# MATRIX FORMAT (Apriori)
# =====================================================
def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basket pivot matrix

    Rows   = Order ID
    Cols   = Product Name
    Values = 0/1

    Output shape: (#orders, #products)
    """

    basket = (
        df
        .groupby(["Order ID", "Product Name"])["Sales"]
        .sum()
        .unstack(fill_value=0)
    )

    # vectorized → faster than applymap
    basket = (basket > 0).astype(int)

    basket.reset_index(inplace=True)

    return basket


# =====================================================
# DEFAULT WRAPPER (project dùng hàm này)
# =====================================================
def build_basket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Default basket builder (matrix format)
    Used by pipeline & notebooks
    """
    return build_basket_matrix(df)
