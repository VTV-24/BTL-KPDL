import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Load raw csv dataset
    """
    df = pd.read_csv(path)
    return df


def basic_info(df: pd.DataFrame):
    """
    Print basic info for quick EDA
    """
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isna().sum())
    print("\nHead:\n", df.head())
