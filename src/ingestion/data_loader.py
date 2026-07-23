from pathlib import Path

import pandas as pd
from loguru import logger

from src.config.settings import PROCESSED_REVIEWS_FILE

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

    def load_processed_reviews(self) -> pd.DataFrame:

        logger.info(
            f"Loading processed dataset from: {PROCESSED_REVIEWS_FILE}"
        )

        if not PROCESSED_REVIEWS_FILE.exists():
            raise FileNotFoundError(
                f"Processed dataset not found: {PROCESSED_REVIEWS_FILE}"
            )

        df = pd.read_csv(PROCESSED_REVIEWS_FILE)

        logger.success(
            f"Successfully loaded {len(df):,} processed reviews."
        )

        return df    