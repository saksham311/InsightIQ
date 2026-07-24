"""
Embedding Storage

Handles saving and loading embeddings.
"""

import numpy as np

from src.config.settings import EMBEDDINGS_FILE
from src.config.logging_config import logger


class EmbeddingStorage:

    @staticmethod
    def save_embeddings(embeddings):

        logger.info(
            f"Saving embeddings to {EMBEDDINGS_FILE}"
        )

        np.save(
            EMBEDDINGS_FILE,
            embeddings,
        )

        logger.info(
            "Embeddings saved successfully."
        )

    @staticmethod
    def load_embeddings():

        logger.info(
            "Loading embeddings..."
        )

        return np.load(
            EMBEDDINGS_FILE
        )