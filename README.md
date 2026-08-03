# 🧠 InsightIQ

> AI-powered Customer Intelligence Platform for Semantic Search, Topic Discovery, and Business Insights.

---

## Overview

InsightIQ is an end-to-end NLP platform that transforms large-scale customer reviews into actionable business intelligence using modern Natural Language Processing and AI.

Rather than relying on keyword matching, InsightIQ understands the semantic meaning of customer feedback, enabling users to retrieve relevant reviews using natural language queries.

This project is being developed incrementally following production-style software engineering practices, with stable versioned releases after each major milestone.

---

## Current Features (v0.2.0)

- Semantic Search using Sentence Transformers
- FAISS Vector Index for efficient similarity search
- CrossEncoder Re-ranking for improved retrieval quality
- NLP preprocessing pipeline
- Dataset profiling and analytics
- Modular, production-oriented project structure
- Configurable application settings
- Interactive command-line search interface

---

## Example

### Query

```text
smooth coffee
```

### Results

```text
1. Smooth, mellow yet flavorful coffee...
2. This coffee is smooth. The best coffee ever...
3. Bold, but smooth...
```

Instead of matching only the keyword "coffee", InsightIQ retrieves reviews that are semantically similar to the user's intent.

---

# Project Architecture

```
Customer Reviews
        │
        ▼
 Data Ingestion
        │
        ▼
 NLP Preprocessing
        │
        ▼
 Sentence Embeddings
        │
        ▼
 FAISS Vector Index
        │
        ▼
 Candidate Retrieval
        │
        ▼
 CrossEncoder Re-ranking
        │
        ▼
 Top Relevant Reviews
```

---

# Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python 3.12 |
| NLP | spaCy |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Re-ranking | CrossEncoder |
| Data Processing | Pandas, NumPy |
| Progress Tracking | tqdm |
| Logging | Loguru |
| Configuration | python-dotenv |

---

# Project Structure

```text
InsightIQ/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   └── vector_index/
│
├── src/
│   ├── config/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── profiling/
│   ├── embeddings/
│   ├── vector_search/
│   └── reranking/
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Current Pipeline

```
Raw Reviews
      │
      ▼
Preprocessing
      │
      ▼
Sentence Embeddings
      │
      ▼
Embedding Storage
      │
      ▼
FAISS Index
      │
      ▼
Semantic Search
      │
      ▼
CrossEncoder Re-ranking
      │
      ▼
Top-k Relevant Reviews
```

---

# Roadmap

## ✅ Phase 1 — Data Foundation

- Data ingestion
- Data profiling
- NLP preprocessing

---

## ✅ Phase 2 — Semantic Search Engine

- Sentence embeddings
- FAISS indexing
- Semantic retrieval
- CrossEncoder re-ranking

---

## 🚧 Phase 3 — Topic Modeling

Planned:

- BERTopic
- UMAP
- HDBSCAN
- Interactive topic explorer

---

## 🚧 Phase 4 — Business Intelligence

Planned:

- KeyBERT keyword extraction
- Topic trends
- Product comparison
- Executive analytics

---

## 🚧 Phase 5 — AI Customer Analyst

Planned:

- Retrieval-Augmented Generation (RAG)
- LLM summarization
- Question answering
- Source attribution

---

## 🚧 Phase 6 — Executive Dashboard

Planned:

- Streamlit dashboard
- KPI cards
- Semantic search interface
- AI insights

---

## 🚧 Phase 7 — Production APIs

Planned:

- FastAPI
- Search API
- AI summarization API
- Health monitoring

---

## 🚧 Phase 8 — Containerization

Planned:

- Docker
- Docker Compose
- PostgreSQL
- Production deployment

---

## Development Philosophy

This project follows a milestone-driven development approach.

Each major feature is:

- Designed
- Implemented
- Tested
- Refactored
- Documented
- Versioned

before moving on to the next phase.

This ensures every release is stable and production-ready.

---

## Current Release

**Version:** `v0.2.0`

### Highlights

- Semantic search
- FAISS indexing
- CrossEncoder re-ranking
- Production-oriented architecture
- Improved code quality and documentation

---

## Future Work

Upcoming releases will introduce:

- Topic modeling
- Business analytics
- RAG pipelines
- LLM-powered insights
- Executive dashboards
- REST APIs
- Docker deployment

---

## License

This project is intended for educational and portfolio purposes.