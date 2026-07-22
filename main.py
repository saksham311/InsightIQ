from src.config.logging_config import logger
from src.config.settings import REVIEWS_FILE
from src.ingestion.data_loader import DataLoader


def main():

    logger.info("Starting InsightIQ...")

    loader = DataLoader(REVIEWS_FILE)

    reviews = loader.load_reviews()

    logger.info(f"Dataset Shape : {reviews.shape}")

    logger.info("\nColumns:")

    for column in reviews.columns:
        logger.info(f" - {column}")

    logger.success("Data ingestion completed successfully.")


if __name__ == "__main__":
    main()