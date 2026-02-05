import os
import sys
import pandas as pd

sys.path.append(".")

from src.utils.config import load_config
from src.data.loader import load_csv, basic_info
from src.data.cleaner import DataCleaner

# NEW
from src.features.rfm import build_rfm
from src.features.basket import build_basket_long, build_basket_matrix
from src.features.time_features import build_monthly_timeseries


def main():
    cfg = load_config()

    raw_path = cfg["paths"]["raw_data"]
    processed_dir = cfg["paths"]["processed_dir"]

    os.makedirs(processed_dir, exist_ok=True)

    # =================================================
    # 1️⃣ LOAD
    # =================================================
    print("Loading data...")
    df = load_csv(raw_path)
    basic_info(df)

    # =================================================
    # 2️⃣ CLEAN
    # =================================================
    print("Cleaning...")
    cleaner = DataCleaner(df)
    df_clean = (
        cleaner
        .remove_duplicates()
        .fill_missing()
        .get_data()
    )

    cleaned_path = os.path.join(processed_dir, "cleaned.parquet")
    df_clean.to_parquet(cleaned_path, index=False)

    print(f"Saved cleaned data -> {cleaned_path}")

    # =================================================
    # 3️⃣ FEATURE ENGINEERING (TUẦN 2)
    # =================================================
    print("\n========== FEATURE ENGINEERING ==========")

    # ---------- RFM ----------
    print("Building RFM...")
    rfm = build_rfm(df_clean)
    rfm_path = os.path.join(processed_dir, "rfm.parquet")
    rfm.to_parquet(rfm_path, index=False)

    # ---------- Basket ----------
    print("Building Basket...")
    basket_long = build_basket_long(df_clean)
    basket_path = os.path.join(processed_dir, "basket.parquet")
    basket_long.to_parquet(basket_path, index=False)

    # matrix (cho clustering/association)
    basket_matrix = build_basket_matrix(df_clean)
    cluster_input_path = os.path.join(processed_dir, "cluster_input.parquet")
    basket_matrix.to_parquet(cluster_input_path)

    # ---------- Time series ----------
    print("Building Time Series...")
    ts = build_monthly_timeseries(df_clean)
    ts_path = os.path.join(processed_dir, "timeseries_monthly.csv")
    ts.to_csv(ts_path, index=False)

    print("\n✅ All preprocessing + features completed!")
    print("Saved:")
    print("-", cleaned_path)
    print("-", rfm_path)
    print("-", basket_path)
    print("-", cluster_input_path)
    print("-", ts_path)


if __name__ == "__main__":
    main()
