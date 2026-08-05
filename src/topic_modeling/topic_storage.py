"""
Topic Storage

Responsible for saving and loading the trained BERTopic model.
"""

from pathlib import Path

from bertopic import BERTopic

from src.config.logging_config import logger
from src.config.settings import TOPIC_MODEL_FILE


class TopicStorage:
    """
    Save and load BERTopic models.
    """
    @staticmethod
    def model_exists() -> bool:
        """
        Check whether a trained BERTopic model exists.
        """

        return TOPIC_MODEL_FILE.exists()

    @staticmethod
    def save_model(model: BERTopic):
        """
        Save the trained BERTopic model.
        """

        logger.info(
            f"Saving BERTopic model to {TOPIC_MODEL_FILE}"
        )

        TOPIC_MODEL_FILE.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        model.save(
            str(TOPIC_MODEL_FILE)
        )

        logger.success(
            "BERTopic model saved successfully."
        )

    @staticmethod
    def load_model() -> BERTopic:
        """
        Load a previously trained BERTopic model.
        """

        logger.info(
            f"Loading BERTopic model from {TOPIC_MODEL_FILE}"
        )

        model = BERTopic.load(
            str(TOPIC_MODEL_FILE)
        )

        logger.success(
            "BERTopic model loaded successfully."
        )

        return model

