import re
import unicodedata

import contractions
import pandas as pd
import spacy
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config.settings import (
    PREPROCESS_BATCH_SIZE,
    SPACY_PIPE_BATCH_SIZE,
)

import warnings
from bs4 import MarkupResemblesLocatorWarning

warnings.filterwarnings(
    "ignore",
    category=MarkupResemblesLocatorWarning,
)

class TextPreprocessor:
    """
    Reusable NLP preprocessing pipeline.
    """

    def __init__(self):

        self.nlp = spacy.load(
            "en_core_web_sm",
            disable=["parser", "ner"],
        )

    def _basic_clean(self, text: str) -> str:
        """
        Perform lightweight cleaning before spaCy.
        """

        if pd.isna(text):
            return ""

        text = str(text)

        text = unicodedata.normalize("NFKC", text)

        text = BeautifulSoup(
            text,
            "html.parser",
        ).get_text()

        text = text.lower()

        text = contractions.fix(text)

        text = re.sub(
            r"http\\S+|www\\S+",
            "",
            text,
        )

        text = re.sub(
            r"\\S+@\\S+",
            "",
            text,
        )

        text = re.sub(
            r"\\s+",
            " ",
            text,
        ).strip()

        return text

    def preprocess_batch(
        self,
        texts: list[str],
    ) -> list[str]:
        """
        Clean a batch of reviews using spaCy's nlp.pipe().
        """

        cleaned_inputs = [
            self._basic_clean(text)
            for text in texts
        ]

        docs = self.nlp.pipe(
            cleaned_inputs,
            batch_size=SPACY_PIPE_BATCH_SIZE,
        )

        cleaned_reviews = []

        for doc in docs:

            cleaned_reviews.append(
                " ".join(
                    token.lemma_
                    for token in doc
                    if not token.is_space
                )
            )

        return cleaned_reviews

    def preprocess_series(
        self,
        reviews: pd.Series,
    ) -> pd.Series:
        """
        Batch-process an entire Series.
        """

        reviews = reviews.fillna("")

        cleaned_reviews = []

        total = len(reviews)

        for start in tqdm(
            range(0, total, PREPROCESS_BATCH_SIZE),
            desc="Preprocessing Reviews",
        ):

            batch = reviews.iloc[
                start:start + PREPROCESS_BATCH_SIZE
            ].tolist()

            cleaned_reviews.extend(
                self.preprocess_batch(batch)
            )

        return pd.Series(
            cleaned_reviews,
            index=reviews.index,
        )