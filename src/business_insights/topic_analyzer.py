"""
Topic Analyzer

Provides business analytics on top of a trained
BERTopic model.
"""

import pandas as pd
import numpy as np
from collections import Counter
import re

from src.topic_modeling.topic_modeler import TopicModeler


class TopicAnalyzer:
    """
    Analyze discovered topics and expose
    business-friendly statistics.
    """

    def __init__(
        self,
        topic_modeler: TopicModeler,
        reviews: pd.DataFrame,
    ):
        self.topic_modeler = topic_modeler
        self.reviews = reviews

    def get_topic_summary(self) -> pd.DataFrame:
        """
        Return BERTopic topic summary.
        """

        return self.topic_modeler.get_topic_info()

    def get_topic_sizes(self) -> pd.DataFrame:
        """
        Return topic IDs and number of reviews.
        """

        summary = self.get_topic_summary()

        return summary[
            ["Topic", "Count"]
        ]

    def get_topic_keywords(
        self,
        topic_id: int,
    ):
        """
        Return keywords for a topic.
        """

        return self.topic_modeler.get_topic(
            topic_id
        )

    def get_representative_reviews(
        self,
        topic_id: int,
    ):
        """
        Return representative reviews
        for a topic.
        """

        return self.topic_modeler.get_model().get_representative_docs()[
            topic_id
        ]

    def get_topic_rating_distribution(self, topic_id: int) -> dict:
        """
        Return the rating distribution for a given topic.
        """

        topics = self.topic_modeler.get_document_topics()

        if len(topics) != len(self.reviews):
            raise ValueError(
                "Number of topic assignments does not match number of reviews."
            )

        topic_reviews = self.reviews[
            np.array(topics) == topic_id
        ]

        if topic_reviews.empty:
            return {}

        distribution = (
            topic_reviews["Score"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        return distribution

    def get_topic_average_rating(self, topic_id: int) -> float:
        """
        Return the average rating for a given topic.
        """

        topics = self.topic_modeler.get_document_topics()

        topic_reviews = self.reviews[
            np.array(topics) == topic_id
        ]

        if topic_reviews.empty:
            return 0.0

        return float(topic_reviews["Score"].mean())

    def get_topic_rating_profile(self, topic_id: int) -> dict:
        """
        Return a business-friendly rating profile for a topic.
        """

        distribution = self.get_topic_rating_distribution(topic_id)

        if not distribution:
            return {
                "average_rating": 0.0,
                "positive_percentage": 0.0,
                "neutral_percentage": 0.0,
                "negative_percentage": 0.0,
            }

        total_reviews = sum(distribution.values())

        positive = sum(
            count
            for rating, count in distribution.items()
            if rating >= 4
        )

        neutral = distribution.get(3, 0)

        negative = sum(
            count
            for rating, count in distribution.items()
            if rating <= 2
        )

        return {
            "average_rating": self.get_topic_average_rating(topic_id),
            "positive_percentage": (positive / total_reviews) * 100,
            "neutral_percentage": (neutral / total_reviews) * 100,
            "negative_percentage": (negative / total_reviews) * 100,
        }

    def get_negative_topics(
        self,
        min_reviews: int = 250,
        max_average_rating: float = 3.0,
    ) -> pd.DataFrame:
        """
        Identify topics associated with negative customer feedback.
        """

        topics = self.topic_modeler.get_document_topics()

        topic_reviews = self.reviews.copy()
        topic_reviews["Topic"] = topics

        # Ignore BERTopic outlier topic
        topic_reviews = topic_reviews[
            topic_reviews["Topic"] != -1
        ]

        topic_stats = (
            topic_reviews
            .groupby("Topic")
            .agg(
                ReviewCount=("Score", "count"),
                AverageRating=("Score", "mean"),
            )
            .reset_index()
        )

        # Calculate negative review percentage
        negative_reviews = (
            topic_reviews
            .assign(
                IsNegative=topic_reviews["Score"] <= 2
            )
            .groupby("Topic")["IsNegative"]
            .mean()
            .mul(100)
            .reset_index(name="NegativePercentage")
        )

        topic_stats = topic_stats.merge(
            negative_reviews,
            on="Topic",
            how="left",
        )

        negative_topics = topic_stats[
            (topic_stats["ReviewCount"] >= min_reviews)
            & (topic_stats["AverageRating"] <= max_average_rating)
        ]

        return negative_topics.sort_values(
            by="AverageRating"
        )

    def get_topic_priority(
        self,
        topic_id: int,
    ) -> str:
        """
        Determine business priority for a topic
        based on rating and negative feedback.
        """

        profile = self.get_topic_rating_profile(topic_id)

        average_rating = profile["average_rating"]
        negative_percentage = profile["negative_percentage"]

        if average_rating <= 2.0 and negative_percentage >= 70:
            return "CRITICAL"

        if average_rating <= 2.5 and negative_percentage >= 60:
            return "HIGH"

        if average_rating <= 3.0 and negative_percentage >= 40:
            return "MEDIUM"

        return "LOW"


    def get_topic_business_summary(
        self,
        topic_id: int,
    ) -> dict:
        """
        Generate a business-oriented summary for a topic.
        """

        profile = self.get_topic_rating_profile(topic_id)

        topic_info = self.get_topic_summary()

        topic_row = topic_info[
            topic_info["Topic"] == topic_id
        ]

        if topic_row.empty:
            raise ValueError(
                f"Topic {topic_id} not found."
            )

        review_count = int(
            topic_row.iloc[0]["Count"]
        )

        keywords = self.get_topic_keywords(topic_id)

        keyword_list = [
            word
            for word, _ in keywords[:10]
        ]

        return {
            "topic_id": topic_id,
            "review_count": review_count,
            "keywords": keyword_list,
            "average_rating": round(
                profile["average_rating"],
                2,
            ),
            "positive_percentage": round(
                profile["positive_percentage"],
                2,
            ),
            "neutral_percentage": round(
                profile["neutral_percentage"],
                2,
            ),
            "negative_percentage": round(
                profile["negative_percentage"],
                2,
            ),
            "priority": self.get_topic_priority(
                topic_id
            ),
        }

    def get_topic_trends(
        self,
        topic_id: int,
        period: str = "year",
    ) -> pd.DataFrame:
        """
        Analyze how frequently a topic appears over time.

        Parameters
        ----------
        topic_id : int
            BERTopic topic ID.

        period : str
            Time aggregation period.
            Supported values: "year", "month".

        Returns
        -------
        pd.DataFrame
            Topic review volume and average rating over time.
        """

        topics = self.topic_modeler.get_document_topics()

        if len(topics) != len(self.reviews):
            raise ValueError(
                "Number of topic assignments does not match number of reviews."
            )

        topic_reviews = self.reviews.copy()

        topic_reviews["Topic"] = topics

        topic_reviews = topic_reviews[
            topic_reviews["Topic"] == topic_id
        ].copy()

        if topic_reviews.empty:
            return pd.DataFrame()

        topic_reviews["Date"] = pd.to_datetime(
            topic_reviews["Time"],
            unit="s",
            errors="coerce",
        )

        topic_reviews = topic_reviews.dropna(
            subset=["Date"]
        )

        if period == "year":

            topic_reviews["Period"] = (
                topic_reviews["Date"].dt.year
            )

        elif period == "month":

            topic_reviews["Period"] = (
                topic_reviews["Date"].dt.to_period("M")
                .astype(str)
            )

        else:

            raise ValueError(
                "Unsupported period. Use 'year' or 'month'."
            )

        trend = (
            topic_reviews
            .groupby("Period")
            .agg(
                ReviewCount=("Score", "count"),
                AverageRating=("Score", "mean"),
            )
            .reset_index()
            .sort_values("Period")
        )

        return trend    

    def get_topic_pain_points(
        self,
        topic_id: int,
        top_n: int = 10,
        max_rating: int = 2,
    ) -> list:
        """
        Extract recurring pain-point terms from negative reviews
        associated with a topic.

        Parameters
        ----------
        topic_id : int
            BERTopic topic ID.

        top_n : int
            Number of pain-point terms to return.

        max_rating : int
            Maximum rating considered negative.

        Returns
        -------
        list
            Most frequent meaningful terms found in negative reviews.
        """

        topics = self.topic_modeler.get_document_topics()

        if len(topics) != len(self.reviews):
            raise ValueError(
                "Number of topic assignments does not match number of reviews."
            )

        topic_reviews = self.reviews.copy()

        topic_reviews["Topic"] = topics

        negative_reviews = topic_reviews[
            (topic_reviews["Topic"] == topic_id)
            & (topic_reviews["Score"] <= max_rating)
        ].copy()

        if negative_reviews.empty:
            return []

        stopwords = {
            "the", "and", "was", "for", "that", "this",
            "with", "have", "but", "not", "are", "you",
            "they", "very", "from", "had", "were", "has",
            "all", "would", "could", "there", "their",
            "what", "when", "which", "about", "just",
            "really", "some", "more", "than", "too",
            "been", "will", "one", "only", "get",
            "got", "like", "also", "even", "out",
            "our", "my", "me", "we", "it", "i",
        }

        words = []

        for text in negative_reviews["Text"].dropna():

            tokens = re.findall(
                r"\b[a-zA-Z]{3,}\b",
                text.lower(),
            )

            words.extend(
                token
                for token in tokens
                if token not in stopwords
            )

        counts = Counter(words)

        return [
            {
                "term": term,
                "frequency": count,
            }
            for term, count in counts.most_common(top_n)
        ]
    