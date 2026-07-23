from pprint import pprint

from src.config.logging_config import logger
from src.config.settings import (
    REVIEWS_FILE,
    PROCESSED_REVIEWS_FILE,
)
from src.ingestion.data_loader import DataLoader
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.profiling.profiler import DataProfiler

from pathlib import Path

def main():

    logger.info("Starting InsightIQ...")

    # Load dataset
    loader = DataLoader(REVIEWS_FILE)
    reviews = loader.load_reviews()

    # Profile dataset
    profiler = DataProfiler(reviews)
    report = profiler.generate_report()

    logger.success("Data Profiling Completed Successfully\n")
    pprint(report, sort_dicts=False)

    if Path(PROCESSED_REVIEWS_FILE).exists():

        logger.success("Processed dataset found. Skipping preprocessing.")

        reviews = loader.load_processed_reviews()

    else:

        logger.info("Starting NLP preprocessing...")

        preprocessor = TextPreprocessor()

        reviews["CleanedText"] = preprocessor.preprocess_series(
            reviews["Text"]
        )

        reviews.to_csv(
            PROCESSED_REVIEWS_FILE,
            index=False,
        )

        logger.success(
            f"Processed dataset saved to {PROCESSED_REVIEWS_FILE}"
        )

if __name__ == "__main__":
    main()