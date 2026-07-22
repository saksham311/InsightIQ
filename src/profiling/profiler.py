from typing import Any

import pandas as pd


class DataProfiler:
    """
    Generates a comprehensive profile report for review datasets.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    def generate_report(self) -> dict[str, Any]:

        review_lengths = (
            self.df["Text"]
            .fillna("")
            .astype(str)
            .str.split()
            .str.len()
        )

        report = {
            "Dataset Summary": {
                "Rows": len(self.df),
                "Columns": len(self.df.columns),
                "Memory Usage (MB)": round(
                    self.df.memory_usage(deep=True).sum() / (1024**2),
                    2,
                ),
            },

            "Missing Values": (
                self.df.isnull()
                .sum()
                .to_dict()
            ),

            "Duplicate Statistics": {
                "Duplicate Rows": int(self.df.duplicated().sum()),
                "Duplicate Reviews": int(
                    self.df.duplicated(subset=["Text"]).sum()
                ),
            },

            "Review Statistics": {
                "Average Rating": round(
                    self.df["Score"].mean(),
                    2,
                ),
                "Average Review Length (Words)": round(
                    review_lengths.mean(),
                    2,
                ),
                "Shortest Review": int(
                    review_lengths.min()
                ),
                "Longest Review": int(
                    review_lengths.max()
                ),
            },

            "User Statistics": {
                "Unique Users": int(
                    self.df["UserId"].nunique()
                ),
            },

            "Product Statistics": {
                "Unique Products": int(
                    self.df["ProductId"].nunique()
                ),
            },

            "Rating Distribution": (
                self.df["Score"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
        }

        return report