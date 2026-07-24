"""
Embedding Generator

Generates semantic embeddings from preprocessed review text.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config.settings import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
)
from src.config.logging_config import logger


class EmbeddingGenerator:

    def __init__(self):

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

    def generate_embeddings(self, texts):

        logger.info(
            "Generating review embeddings..."
        )

        embeddings = []

        total = len(texts)

        for start in tqdm(
            range(0, total, self.batch_size),
            desc="Generating Embeddings",
        ):

            batch = texts.iloc[
                start : start + self.batch_size
            ].tolist()

            batch_embeddings = self.model.encode(
                batch,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            embeddings.append(
                batch_embeddings
            )

        embeddings = np.vstack(
            embeddings
        )

        logger.info(
            "Embedding generation completed."
        )

        return embeddings