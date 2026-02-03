import os
import sys

sys.path.append(".")

from src.utils.config import load_config
from src.data.loader import load_csv, basic_info
from src.data.cleaner import DataCleaner


def main():
    cfg = load_config()

    raw_path = cfg["paths"]["raw_data"]
    processed_dir = cfg["paths"]["processed_dir"]

    os.makedirs(processed_dir, exist_ok=True)

    print("Loading data...")
    df = load_csv(raw_path)

    basic_info(df)

    print("Cleaning...")
    cleaner = DataCleaner(df)
    df_clean = cleaner.remove_duplicates().fill_missing().get_data()

    save_path = os.path.join(processed_dir, "cleaned.parquet")
    df_clean.to_parquet(save_path, index=False)

    print(f"Saved cleaned data -> {save_path}")


if __name__ == "__main__":
    main()
