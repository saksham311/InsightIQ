"""
Context Retriever

Retrieves relevant review and business context
for the RAG AI Analyst.
"""

import pandas as pd

from src.business_insights.topic_analyzer import TopicAnalyzer
from src.vector_search.search_engine import SearchEngine


class ContextRetriever:
    """
    Retrieve relevant semantic and business context
    for an analyst question.
    """

    def __init__(
        self,
        search_engine: SearchEngine,
        topic_analyzer: TopicAnalyzer,
    ):
        self.search_engine = search_engine
        self.topic_analyzer = topic_analyzer

    def retrieve_reviews(
        self,
        query: str,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """
        Retrieve the most relevant reviews for a query.
        """

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        results = self.search_engine.search(
            query
        )

        return results.head(top_k)

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict:
        """
        Retrieve structured context for the RAG pipeline.
        """

        reviews = self.retrieve_reviews(
            query=query,
            top_k=top_k,
        )

        return {
            "query": query,
            "reviews": reviews,
        }