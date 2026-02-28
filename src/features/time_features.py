import pandas as pd


def build_monthly_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly aggregated sales time series

    Output:
        date   | sales
        2014-01-31 | 12345
        2014-02-28 | 15321
        ...

    Used for:
        - ARIMA
        - Holt-Winters
        - Prophet
    """

    df = df.copy()

    # ===== Robust datetime parsing (FIX CRASH) =====
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        dayfirst=True,     # handle dd/mm/yyyy
        errors="coerce"    # avoid crash
    )

    # drop invalid rows
    df = df.dropna(subset=["Order Date"])

    # ===== Monthly aggregation =====
    ts = (
        df
        .set_index("Order Date")
        .resample("M")["Sales"]   # monthly
        .sum()
        .reset_index()
    )

    ts.columns = ["date", "sales"]

    # ===== sort for safety =====
    ts = ts.sort_values("date").reset_index(drop=True)

    return ts
