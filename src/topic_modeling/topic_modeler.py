"""
Topic Modeler

Builds and manages a BERTopic model for discovering
semantic topics from customer reviews.
"""

from typing import Optional

import numpy as np
import pandas as pd
from bertopic import BERTopic

from src.config.logging_config import logger


class TopicModeler:
    """
    Wrapper around BERTopic used for topic discovery.
    """

    def __init__(self):
        """
        Initialise the BERTopic model.
        """

        logger.info("Initializing BERTopic model...")

        self.model = BERTopic(
            verbose=True,
        )

        self.topics: Optional[list[int]] = None
        self.probabilities: Optional[np.ndarray] = None

        logger.success("BERTopic initialized successfully.")

    def fit(
        self,
        documents: pd.Series,
        embeddings: np.ndarray,
    ) -> tuple[list[int], np.ndarray]:
        """
        Train the BERTopic model.

        Args:
            documents:
                Preprocessed review text.

            embeddings:
                Sentence embeddings corresponding
                to each review.

        Returns:
            Tuple containing:
                - topic assignments
                - topic probabilities
        """

        logger.info(f"Training BERTopic model on {len(documents):,} documents...")

        topics, probabilities = self.model.fit_transform(
            documents = documents.astype(str).tolist(),
            embeddings = embeddings,
        )

        self.topics = topics
        self.probabilities = probabilities

        logger.success("Topic modeling completed successfully.")

        return topics, probabilities

    def get_topic_info(self) -> pd.DataFrame:
        """
        Return summary information for every topic.
        """

        return self.model.get_topic_info()

    def get_topic(
        self,
        topic_id: int,
    ):
        """
        Return the keywords describing a topic.
        """

        return self.model.get_topic(topic_id)

    def get_document_topics(self) -> list[int]:
        """
        Return the topic assigned to each document.
        """

        if self.topics is None:
            raise ValueError(
                "No topics found. Please fit the model first."
            )
        
        return self.topics

    def get_model(self) -> BERTopic:
        """
        Return the underlying BERTopic model.
        """

        return self.model

    def load_model(
        self,
        model: BERTopic,
    ) -> None:
        """
        Replace the current model with a previously
        trained BERTopic model.
        """

        self.model = model
        self.topics = model.topics_