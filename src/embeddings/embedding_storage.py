"""
Embedding Storage

Handles saving and loading sentence embeddings.
"""

import numpy as np

from src.config.logging_config import logger
from src.config.settings import EMBEDDINGS_FILE


class EmbeddingStorage:
    """
    Utility class for persisting embedding vectors
    to disk and loading them when required.
    """

    @staticmethod
    def save_embeddings(
        embeddings: np.ndarray,
    ) -> None:
        """
        Save embeddings to disk.

        Args:
            embeddings: NumPy array containing sentence embeddings.
        """

        logger.info(
            f"Saving embeddings to {EMBEDDINGS_FILE}"
        )

        np.save(
            EMBEDDINGS_FILE,
            embeddings,
        )

        logger.success(
            "Embeddings saved successfully."
        )

    @staticmethod
    def load_embeddings() -> np.ndarray:
        """
        Load embeddings from disk.

        Returns:
            NumPy array containing sentence embeddings.
        """

        logger.info(
            f"Loading embeddings from {EMBEDDINGS_FILE}"
        )

        embeddings = np.load(
            EMBEDDINGS_FILE
        ).astype(np.float32)

        logger.success(
            f"Loaded {embeddings.shape[0]:,} embeddings successfully."
        )

        return embeddings