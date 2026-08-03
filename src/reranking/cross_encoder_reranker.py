"""
Cross Encoder Reranker

Re-ranks FAISS search results using a CrossEncoder model.
"""

import pandas as pd
from sentence_transformers import CrossEncoder

from src.config.logging_config import logger
from src.config.settings import CROSS_ENCODER_MODEL


class CrossEncoderReranker:
    """
    Re-ranks semantic search results using a CrossEncoder model
    for improved retrieval quality.
    """

    def __init__(
        self,
        model_name: str = CROSS_ENCODER_MODEL,
    ):
        """
        Load the CrossEncoder reranking model.

        Args:
            model_name: Hugging Face model identifier.
        """

        logger.info(
            f"Loading CrossEncoder model: {model_name}"
        )

        self.model = CrossEncoder(model_name)

        logger.success(
            "CrossEncoder model loaded successfully."
        )

    def rerank(
        self,
        query: str,
        results: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Re-rank candidate reviews using the CrossEncoder.

        Args:
            query: User search query.
            results: Candidate reviews returned by FAISS.

        Returns:
            DataFrame sorted by CrossEncoder relevance score.
        """

        if results.empty:
            return results

        pairs = [
            (query, review)
            for review in results["Text"]
        ]

        scores = self.model.predict(pairs)

        reranked_results = results.copy()

        reranked_results["RerankScore"] = scores

        reranked_results = reranked_results.sort_values(
            by="RerankScore",
            ascending=False,
        )

        reranked_results = reranked_results.reset_index(
            drop=True
        )

        return reranked_results