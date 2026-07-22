from pathlib import Path

import pandas as pd
from loguru import logger


class DataLoader:
    """
    Responsible for loading datasets.
    """

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path

    def load_reviews(self) -> pd.DataFrame:

        logger.info(f"Loading dataset from: {self.dataset_path}")

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        df = pd.read_csv(self.dataset_path)

        logger.success(f"Successfully loaded {len(df):,} reviews.")

        return df