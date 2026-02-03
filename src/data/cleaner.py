import pandas as pd


class DataCleaner:
    """
    Basic cleaning utilities
    (week 1: placeholder only)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self

    def fill_missing(self):
        # TODO: update later
        self.df = self.df.fillna(0)
        return self

    def get_data(self):
        return self.df
