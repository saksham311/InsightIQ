from pprint import pprint

from src.config.logging_config import logger
from src.config.settings import REVIEWS_FILE
from src.ingestion.data_loader import DataLoader
from src.profiling.profiler import DataProfiler


def main():

    logger.info("Starting InsightIQ...")

    loader = DataLoader(REVIEWS_FILE)

    reviews = loader.load_reviews()

    profiler = DataProfiler(reviews)

    report = profiler.generate_report()

    logger.success("Data Profiling Completed Successfully\n")

    pprint(report, sort_dicts=False)


if __name__ == "__main__":
    main()