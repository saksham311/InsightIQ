"""
Semantic Search Engine

Encodes a user query, performs semantic search using FAISS,
and returns the most similar reviews.
"""

import faiss
import pandas as pd

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.config.logging_config import logger
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.config.settings import (
    FAISS_SEARCH_CANDIDATES,
    SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K_RESULTS,
)

class SearchEngine:

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        faiss_index: faiss.Index,
        reranker: CrossEncoderReranker,
        reviews: pd.DataFrame,
    ):
        self.embedding_generator = embedding_generator
        self.faiss_index = faiss_index
        self.reranker = reranker
        self.reviews = reviews

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K_RESULTS,
    ):
        """
        Perform semantic search for a user query.

        Steps:
            1. Encode the query.
            2. Retrieve candidate reviews using FAISS.
            3. Filter duplicate and low-similarity matches.
            4. Rerank candidates using the CrossEncoder.
            5. Return the top-k ranked reviews.
        """

        logger.info(f"Searching for: '{query}'")

        query_embedding = self.embedding_generator.encode_query(query)

        # Retrieve a larger candidate set so the CrossEncoder can rerank them effectively
        #can rerank the most relevant results based on the query and the review text.
        scores, indices = self.faiss_index.search(
            query_embedding,
            FAISS_SEARCH_CANDIDATES,
        )

        results = self.reviews.iloc[indices[0]].copy()

        results["Similarity"] = scores[0]

        # Highest similarity first
        results = results.sort_values(
            by="Similarity",
            ascending=False,
        )

        # Remove duplicate review texts before reranking to avoid biasing the results
        results = results.drop_duplicates(
            subset=["Text"],
            keep="first",
        )

        # Remove weak matches
        results = results[
            results["Similarity"] >= SIMILARITY_THRESHOLD
        ]

        # Return only requested number
        results = results.head(top_k)

        logger.success(
            f"Found {len(results)} matching reviews."
        )

        # Rerank the results
        
        results = self.reranker.rerank(query, results)
        results = results.head(top_k)

        return results