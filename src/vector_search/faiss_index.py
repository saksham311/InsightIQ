"""
FAISS Vector Index

Responsible for:

- Building the FAISS index
- Normalizing embeddings
- Saving/loading the index
- Providing access to the search index
"""

# Limit FAISS to a single thread for consistent
# performance across development environments.
import faiss
faiss.omp_set_num_threads(1)

import numpy as np

from src.config.logging_config import logger
from src.config.settings import FAISS_INDEX_FILE


class FAISSIndex:
    """
    Wrapper around a FAISS IndexFlatIP
    used for semantic vector search.
    """

    def __init__(self):
        self.index: faiss.Index | None = None

    def build_index(self, embeddings: np.ndarray):
        """
        Build a FAISS IndexFlatIP from the provided embeddings.

        Args:
            embeddings: Sentence embeddings of shape (n_samples, embedding_dim).

        Returns:
            None
        """

        logger.info("Building FAISS index...")

        embeddings = embeddings.astype(np.float32)

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1.0

        embeddings = embeddings / norms

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32
        )
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        logger.info(f"Adding {embeddings.shape[0]:,} embeddings to the index...")
    
        self.index.add(embeddings)

        logger.success(
            f"Indexed {self.index.ntotal:,} embeddings successfully."
        )  

    def save_index(self):
        """
        Save the FAISS index to disk.

        Raises:
            ValueError: If the index has not been created.
        """
            
        logger.info(f"Saving FAISS index to {FAISS_INDEX_FILE}")

        if self.index is None:
            raise ValueError(
                " No FAISS index found."
                " Please build or load an index before saving."
            )

        faiss.write_index(
            self.index,
            str(FAISS_INDEX_FILE)
        )

        logger.success("FAISS index saved successfully.")

    def load_index(self):
        """
        Load a previously saved FAISS index from disk.
        """

        logger.info(f"Loading FAISS index from {FAISS_INDEX_FILE}")

        self.index = faiss.read_index(
            str(FAISS_INDEX_FILE)
        )

        logger.success("FAISS index loaded successfully.")

    def get_index(self):
        """
        Return the initialized FAISS index.

        Raises:
            ValueError: If the index has not been initialized.
        """

        if self.index is None:
            raise ValueError("FAISS index has not been initialized.")

        return self.index