"""
Application configuration.

Defines project paths, model settings,
preprocessing parameters, and search configuration.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dataset
REVIEWS_FILE = RAW_DATA_DIR / "Reviews.csv"

# Model
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Processed Data Paths
PROCESSED_REVIEWS_FILE = (
    PROCESSED_DATA_DIR / "reviews_processed.csv"
)

# ==========================
# NLP Configuration
# ==========================

PREPROCESS_BATCH_SIZE = 5000
SPACY_PIPE_BATCH_SIZE = 256

# ==========================
# Embedding Configuration
# ==========================

EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_FILE = EMBEDDINGS_DIR / "review_embeddings.npy"
METADATA_FILE = EMBEDDINGS_DIR / "review_metadata.parquet"

EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        512
    )
)

# ===========================
# Vector Search
# ===========================

VECTOR_INDEX_DIR = DATA_DIR / "vector_index"
VECTOR_INDEX_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_FILE = VECTOR_INDEX_DIR / "review_index.faiss"

# Number of candidate reviews retrieved from FAISS
FAISS_SEARCH_CANDIDATES = int(
    os.getenv(
        "FAISS_SEARCH_CANDIDATES",
        50
    )
)

# Minimum cosine similarity to retain a review
SIMILARITY_THRESHOLD = float(
    os.getenv(
        "SIMILARITY_THRESHOLD",
        0.55
    )
)

# Number of results returned to the user
DEFAULT_TOP_K_RESULTS = int(
    os.getenv(
        "DEFAULT_TOP_K_RESULTS",
        5
    )
)

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

# ==========================
# Topic Modeling
# ==========================

TOPIC_MODEL_DIR = DATA_DIR / "topic_models"
TOPIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)

TOPIC_MODEL_FILE = TOPIC_MODEL_DIR / "bertopic_model"