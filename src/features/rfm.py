import pandas as pd


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build RFM features:
    - Recency
    - Frequency
    - Monetary
    """

    df = df.copy()

    # ===== FIX DATE PARSING =====
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,
        errors="coerce"
    )

    df = df.dropna(subset=["Order Date"])

    snapshot_date = df["Order Date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("Customer ID")
        .agg({
            "Order Date": lambda x: (snapshot_date - x.max()).days,
            "Order ID": "nunique",
            "Sales": "sum"
        })
        .reset_index()
    )

    rfm.columns = ["Customer ID", "Recency", "Frequency", "Monetary"]

    return rfm
