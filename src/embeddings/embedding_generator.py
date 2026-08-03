"""
Embedding Generator

Generates semantic embeddings from preprocessed review text.
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config.settings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)
from src.config.logging_config import logger


class EmbeddingGenerator:
    """
    Generates semantic embeddings for reviews and user queries
    using a SentenceTransformer model.
    """

    def __init__(self):
        """
        Load the SentenceTransformer model and initialize
        embedding generation settings.
        """

        logger.info(
            f"Loading embedding model: {EMBEDDING_MODEL}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        self.batch_size = EMBEDDING_BATCH_SIZE

        logger.info(
            "Embedding model loaded successfully."
        )

    def generate_embeddings(
            self,
            texts: pd.Series,
    ) -> np.ndarray:
        """
        Generate embeddings for a collection of review texts.

        Args:
            texts: Pandas Series containing review text.

        Returns:
            NumPy array of sentence embeddings.
        """

        logger.info(
            "Generating review embeddings..."
        )

        all_embeddings = []

        total = len(texts)

        for start in tqdm(
            range(0, total, self.batch_size),
            desc="Generating Embeddings",
        ):
            end = start + self.batch_size
            batch = texts.iloc[
                start : end
            ].tolist()

            batch_embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            all_embeddings.append(
                batch_embeddings
            )

        embeddings = np.vstack(
            all_embeddings
        ).astype(np.float32)

        logger.info(
            "Embedding generation completed."
        )

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single user query into the same semantic space
        as the indexed review embeddings.

        Args:
            query: Natural language search query.

        Returns:
            L2-normalized embedding vector.
        """

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        norms = np.linalg.norm(
            embedding,
            axis=1,
            keepdims=True,
        )

        norms[norms == 0] = 1.0

        embedding = embedding / norms

        return embedding