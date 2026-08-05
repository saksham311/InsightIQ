

from pathlib import Path
from pprint import pprint

from src.config.logging_config import logger
from src.config.settings import (
    REVIEWS_FILE,
    PROCESSED_REVIEWS_FILE,
    FAISS_INDEX_FILE,
    TOPIC_MODEL_FILE,
)

from src.ingestion.data_loader import DataLoader
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.profiling.profiler import DataProfiler

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.embedding_storage import EmbeddingStorage

from src.vector_search.faiss_index import FAISSIndex
from src.vector_search.search_engine import SearchEngine

from src.reranking.cross_encoder_reranker import CrossEncoderReranker

from src.topic_modeling.topic_modeler import TopicModeler
from src.topic_modeling.topic_storage import TopicStorage

def main():

    logger.info("Starting InsightIQ...")

    # ----------------------------
    # Load Dataset
    # ----------------------------
    loader = DataLoader(REVIEWS_FILE)
    reviews = loader.load_reviews()

    # ----------------------------
    # Profile Dataset
    # ----------------------------
    profiler = DataProfiler(reviews)
    report = profiler.generate_report()

    logger.success("Data Profiling Completed Successfully.\n")
    pprint(report, sort_dicts=False)

    # ----------------------------
    # Preprocess Reviews
    # ----------------------------
    if Path(PROCESSED_REVIEWS_FILE).exists():

        logger.success(
            "Processed dataset found. Skipping preprocessing."
        )

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

    # ----------------------------
    # Load Embeddings
    # ----------------------------
    logger.info("Loading embeddings from disk...")

    embeddings = EmbeddingStorage.load_embeddings()


    # ----------------------------
    # Load / Train Topic Model
    # ----------------------------
    topic_modeler = TopicModeler()

    if TopicStorage.model_exists():

        logger.success(
            "BERTopic model found. Loading existing model."
        )

        topic_model = TopicStorage.load_model()

        topic_modeler.load_model(topic_model)

    else:

        logger.info(
            "Training BERTopic model..."
        )

        topic_modeler.fit(
            documents=reviews["CleanedText"],
            embeddings=embeddings,
        )

        TopicStorage.save_model(
            topic_modeler.get_model()
        )


    # ----------------------------
    # Load / Build FAISS Index
    # ----------------------------
    faiss_index = FAISSIndex()

    if Path(FAISS_INDEX_FILE).exists():

        logger.success(
            "FAISS index found. Loading existing index."
        )

        faiss_index.load_index()

    else:

        logger.info(
            "Building FAISS index from embeddings..."
        )

        faiss_index.build_index(embeddings)
        faiss_index.save_index()

    logger.success("InsightIQ setup completed successfully.")

    # ----------------------------
    # Initialise Search Engine
    # ----------------------------
    embedding_generator = EmbeddingGenerator()

    reranker = CrossEncoderReranker()

    search_engine = SearchEngine(
        embedding_generator=embedding_generator,
        faiss_index=faiss_index.get_index(),
        reranker=reranker,
        reviews=reviews,
    )

    logger.success(
        "InsightIQ is ready for semantic search."
    )

    # ----------------------------
    # Interactive Search
    # ----------------------------
    while True:

        query = input(
            "\nEnter your search query ('exit' to quit): "
        ).strip()

        if query.lower() == "exit":

            logger.info("InsightIQ terminated successfully!")

            break

        if not query:

            logger.warning(
                "Please enter a search query."
            )

            continue

        try:

            results = search_engine.search(query)

            print("\n" + "=" * 100)
            print(
                f"Top {len(results)} Matching Reviews"
            )
            print("=" * 100)

            for i, (_, row) in enumerate(
                results.iterrows(),
                start=1,
            ):

                print(f"\nResult {i}")
                print(
                    f"Rating     : {row['Score']}"
                )
                print(
                    f"Similarity : {row['Similarity']:.4f}"
                )
                print(
                    f"RerankScore: {row['RerankScore']:.4f}"
                )
                print(
                    f"Summary    : {row['Summary']}"
                )
                print(
                    f"Review     : {row['Text']}"
                )

                print("-" * 100)

        except Exception as e:

            logger.error(
                f"Search failed: {e}"
            )


if __name__ == "__main__":
    main()