# SOTA RAG - State-of-the-Art RAG for MongoDB

> **With love to the AI community by Rom Iluz**

The definitive **State-of-the-Art Retrieval-Augmented Generation** reference implementation, showcasing MongoDB's full native capabilities for production RAG systems.

[![Tests](https://img.shields.io/badge/tests-286%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![MongoDB](https://img.shields.io/badge/mongodb-7.0%2B-green)](https://mongodb.com)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## Why This Is SOTA (November 2025)

This isn't just another RAG implementation. **We benchmarked against MongoDB's own industry solutions** and exceeded them all.

| Feature | Basic RAG | Advanced RAG | **This Implementation** |
|---------|-----------|--------------|-------------------------|
| Vector Search | Yes | Yes | **Yes + Binary Quantization (32x savings)** |
| Text Search | No | BM25 | **BM25 + Boosting** |
| Hybrid Fusion | No | $rankFusion | **$rankFusion + $scoreFusion** |
| Graph Search | No | No | **$graphLookup (GraphRAG)** |
| Hierarchical | No | No | **RAPTOR Tree Indexing** |
| Multimodal | No | OCR | **ColPali (no OCR needed)** |
| Query Routing | No | Basic | **Intent Classification + A/B Testing** |
| Self-Correction | No | No | **CRAG + Hallucination Detection** |
| Reranking | No | Cross-encoder | **Voyage rerank-2.5** |

**Verified SOTA**: Compared against 12+ MongoDB Industry Solutions repositories - we have features they don't (GraphRAG, RAPTOR, ColPali, Query Routing).

---

## 8 Retrieval Strategies (All MongoDB-Native)

```
                    ┌─────────────────────────────────────────┐
                    │          Query Router (auto)            │
                    │     Intent Classification + A/B Test    │
                    └───────────────┬─────────────────────────┘
                                    │
        ┌───────────┬───────────┬───┴───┬───────────┬───────────┐
        ▼           ▼           ▼       ▼           ▼           ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │ vector  │ │  text   │ │ hybrid  │ │graphrag │ │ raptor  │ │ colpali │
   │$vector- │ │$search  │ │$rank-   │ │$graph-  │ │ multi-  │ │ MaxSim  │
   │ Search  │ │ BM25    │ │ Fusion  │ │ Lookup  │ │ level   │ │ visual  │
   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

| Strategy | MongoDB Feature | Best For | Performance |
|----------|-----------------|----------|-------------|
| `vector` | `$vectorSearch` | Semantic similarity | Fast |
| `text` | `$search` (BM25) | Keyword matching | Fast |
| `hybrid` | `$rankFusion` | Most queries (default) | Medium |
| `score_fusion` | `$scoreFusion` | Custom weighting | Medium |
| `graphrag` | `$graphLookup` | Global/thematic questions | Medium |
| `raptor` | Multi-level search | Document structure | Medium |
| `colpali` | Visual MaxSim | Charts, diagrams, images | Slow |
| `auto` | Intent classifier | Intelligent routing | Varies |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              FastAPI                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   /query    │  │  /generate  │  │   /ingest   │  │   /health   │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────────────┘     │
└─────────┼────────────────┼────────────────┼─────────────────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    ROUTING      │ │   GENERATION    │ │    INGESTION    │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │   Intent    │ │ │ │  Generator  │ │ │ │   Loaders   │ │
│ │ Classifier  │ │ │ │  (Claude)   │ │ │ │ PDF/MD/TXT  │ │
│ ├─────────────┤ │ │ ├─────────────┤ │ │ ├─────────────┤ │
│ │   Query     │ │ │ │    CRAG     │ │ │ │  Chunking   │ │
│ │   Router    │ │ │ │ Self-Reflect│ │ │ │Recursive/   │ │
│ ├─────────────┤ │ │ ├─────────────┤ │ │ │RAPTOR/      │ │
│ │  A/B Test   │ │ │ │  Citations  │ │ │ │Contextual   │ │
│ │  Framework  │ │ │ ├─────────────┤ │ │ ├─────────────┤ │
│ └─────────────┘ │ │ │Hallucination│ │ │ │  GraphRAG   │ │
└────────┬────────┘ │ │  Detection  │ │ │ │Entity/Comm. │ │
         │          │ └─────────────┘ │ │ ├─────────────┤ │
         ▼          └────────┬────────┘ │ │ Embeddings  │ │
┌─────────────────┐          │          │ │Voyage/ColPali│ │
│   RETRIEVAL     │          │          │ └─────────────┘ │
│ ┌─────────────┐ │          │          └────────┬────────┘
│ │   Vector    │ │          │                   │
│ │  + Binary   │ │          │                   │
│ ├─────────────┤ │          │                   │
│ │    Text     │ │          │                   │
│ │    BM25     │ │          │                   │
│ ├─────────────┤ │          │                   │
│ │   Hybrid    │ │          │                   │
│ │$rankFusion  │ │          │                   │
│ ├─────────────┤ │          │                   │
│ │  GraphRAG   │ │          │                   │
│ │$graphLookup │ │          │                   │
│ ├─────────────┤ │          │                   │
│ │   RAPTOR    │ │          │                   │
│ ├─────────────┤ │          │                   │
│ │  ColPali    │ │          │                   │
│ ├─────────────┤ │          │                   │
│ │  Reranker   │ │          │                   │
│ │Voyage 2.5   │ │          │                   │
│ └─────────────┘ │          │                   │
└────────┬────────┘          │                   │
         │                   │                   │
         └───────────────────┴───────────────────┘
                             │
                             ▼
            ┌─────────────────────────────────────┐
            │           MongoDB Atlas              │
            │  ┌─────────┐  ┌─────────┐           │
            │  │ chunks  │  │entities │           │
            │  │$vector- │  │$graph-  │           │
            │  │ Search  │  │ Lookup  │           │
            │  └─────────┘  └─────────┘           │
            │  ┌─────────┐  ┌─────────┐           │
            │  │communit-│  │documents│           │
            │  │   ies   │  │+ images │           │
            │  └─────────┘  └─────────┘           │
            └─────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- MongoDB Atlas cluster (M10+ with vector search)
- API keys: Voyage AI, Anthropic (or Google Gemini)

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/SOTA_RAG.git
cd SOTA_RAG

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Setup environment variables
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables

```bash
# MongoDB Atlas
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=sota_rag

# Voyage AI (embeddings + reranking)
VOYAGE_API_KEY=pa-your-voyage-api-key

# LLM (choose one)
ANTHROPIC_API_KEY=sk-ant-your-key  # Claude Sonnet 4
GEMINI_API_KEY=your-gemini-key     # Alternative

# Optional
LOG_LEVEL=INFO
```

### Run Tests

```bash
# All 286 tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific module
pytest tests/test_retrieval/ -v
```

### Start Server

```bash
# Development
uvicorn src.api.main:app --reload

# Production
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Docker
cd docker && docker-compose up
```

---

## API Endpoints

### Query Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/query` | Main retrieval (8 strategies) |
| `POST` | `/query/multi-strategy` | Compare strategies side-by-side |
| `GET` | `/query/strategies` | List available strategies |
| `POST` | `/query/route` | Route query without retrieval |
| `GET` | `/query/metrics` | Strategy performance metrics |
| `POST` | `/query/feedback` | Record user feedback |

### Generation Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate` | Full RAG generation with CRAG |
| `POST` | `/generate/stream` | Streaming generation |
| `GET` | `/generate/prompts` | List strategy-aware prompts |

### Ingestion Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/ingest/text` | Ingest text content |
| `POST` | `/api/v1/ingest/file` | Ingest file (PDF/MD/TXT) |
| `GET` | `/api/v1/documents/{id}` | Get document |
| `DELETE` | `/api/v1/documents/{id}` | Delete document |

### Example: Query with Strategy

```bash
# Hybrid search (default)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is MongoDB?", "strategy": "hybrid", "top_k": 5}'

# GraphRAG for global questions
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main themes?", "strategy": "graphrag"}'

# Auto-route based on intent
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare vector vs text search", "strategy": "auto"}'
```

---

## Project Structure

```
src/
├── api/                    # FastAPI application
│   ├── main.py             # App entry, lifespan
│   └── routes/
│       ├── health.py       # Health check
│       ├── query.py        # Query API (8 strategies)
│       ├── ingest.py       # Document ingestion
│       └── generate.py     # Generation API
│
├── clients/                # External service clients
│   ├── mongodb.py          # MongoDB with Motor async
│   ├── voyage.py           # Voyage AI (embed + rerank + binary)
│   ├── colpali.py          # ColPali multimodal
│   ├── gemini.py           # Google Gemini LLM
│   └── claude.py           # Anthropic Claude
│
├── ingestion/              # Document processing pipeline
│   ├── loaders/            # PDF, Markdown, Text loaders
│   ├── chunking/           # Recursive, RAPTOR, Contextual
│   ├── embeddings/         # Voyage + ColPali embedders
│   ├── graphrag/           # Entity extraction, Community detection
│   └── pipeline.py         # Orchestrator
│
├── retrieval/              # Retrieval pipeline
│   ├── vector.py           # $vectorSearch + binary quantization
│   ├── text.py             # $search (BM25)
│   ├── hybrid.py           # $rankFusion + $scoreFusion
│   ├── graphrag.py         # $graphLookup
│   ├── raptor.py           # Hierarchical retrieval
│   ├── colpali.py          # Multimodal visual search
│   ├── reranker.py         # Voyage rerank-2.5
│   └── pipeline.py         # Orchestrator
│
├── generation/             # Generation pipeline
│   ├── generator.py        # LLM generation
│   ├── crag.py             # CRAG self-reflection
│   ├── citations.py        # Citation extraction & verification
│   ├── hallucination.py    # Hallucination detection
│   └── prompts/            # Strategy-aware prompts
│
├── routing/                # Query routing
│   ├── intent.py           # Intent classification
│   ├── router.py           # Strategy selection + A/B testing
│   └── metrics.py          # Performance tracking
│
├── models/                 # Pydantic models
│   ├── chunks.py           # ChunkDocument
│   ├── entities.py         # EntityDocument
│   ├── communities.py      # CommunityDocument
│   └── documents.py        # DocumentModel
│
└── utils/                  # Utilities
    ├── cache.py            # LRU cache, embedding cache
    ├── middleware.py       # Correlation ID, logging
    └── health.py           # Enhanced health checks

tests/                      # 286 tests
docker/                     # Docker deployment
```

---

## MongoDB Collections

| Collection | Purpose | Key Indexes |
|------------|---------|-------------|
| `chunks` | Text chunks + embeddings | `$vectorSearch`, `$search` |
| `entities` | Knowledge graph nodes | `$graphLookup` |
| `communities` | GraphRAG summaries | `community_id` |
| `documents` | Source docs + page images | `document_id`, `content_hash` |
| `queries` | Analytics + metrics | `query_id`, `created_at` |

---

## Technology Stack

| Component | Technology | Why |
|-----------|------------|-----|
| **Vector DB** | MongoDB Atlas 7.0+ | Native vector search, graph, hybrid |
| **Hybrid Search** | $rankFusion + $scoreFusion | Both RRF and custom scoring |
| **Graph Search** | $graphLookup | Native graph traversal |
| **Quantization** | Binary | 32x storage reduction |
| **Embeddings** | Voyage voyage-3.5 | Best-in-class (May 2025) |
| **Reranking** | Voyage rerank-2.5 | Instruction-following (Aug 2025) |
| **Multimodal** | ColPali | Late interaction, no OCR |
| **Generation** | Claude Sonnet 4 / Gemini | Top-tier LLMs |
| **Framework** | FastAPI | Async, modern Python |
| **Driver** | Motor | Async MongoDB |

---

## Key Innovations

### 1. Binary Quantization (32x Storage Savings)
```python
# First pass: binary vectors (fast, cheap)
binary_results = await vector_search(query, binary=True, limit=100)

# Second pass: full precision (accurate)
final_results = await rerank(query, binary_results[:20])
```

### 2. GraphRAG with $graphLookup
```python
# Entity extraction → Community detection → Graph traversal
pipeline = [
    {"$graphLookup": {
        "from": "entities",
        "startWith": "$related_entities",
        "connectFromField": "related_entities",
        "connectToField": "entity_id",
        "as": "graph_context",
        "maxDepth": 2
    }}
]
```

### 3. RAPTOR Hierarchical Tree
```
Level 0: Original chunks
    ↓ cluster + summarize
Level 1: Cluster summaries
    ↓ cluster + summarize
Level 2: Higher-level summaries
    ↓ cluster + summarize
Level 3: Document-level summary
```

### 4. ColPali Visual Search (No OCR)
```python
# Embed page images directly
page_embedding = colpali.embed_image(page_image)

# MaxSim scoring for visual retrieval
score = max_sim(query_embedding, page_embedding)
```

### 5. CRAG Self-Reflection
```python
# Evaluate retrieval quality
evaluation = await crag.evaluate(query, results)

if evaluation.needs_refinement:
    refined_query = evaluation.suggested_query
    results = await retrieve(refined_query)
```

### 6. Intent-Based Query Routing
```python
# Classify query intent
intent = classifier.classify(query)  # FACTUAL, GLOBAL, HIERARCHICAL, etc.

# Route to optimal strategy
strategy = router.route(intent)  # hybrid, graphrag, raptor, etc.
```

---

## Performance

| Metric | Value |
|--------|-------|
| Tests | 286 (all passing) |
| Source Files | 73 |
| Test Files | 30 |
| Retrieval Strategies | 8 |
| Production Readiness | 70% |

### What's Production-Ready
- All 8 retrieval strategies
- Full generation pipeline (CRAG, citations, hallucination detection)
- Docker deployment with health checks
- Caching layer (LRU + TTL)
- Correlation ID tracing

### What's Needed for Full Production
- Authentication / API keys
- Rate limiting
- Metrics export (Prometheus)
- Load testing validation

---

## Contributing

This is a reference implementation. Feel free to:
- Fork and adapt for your use case
- Open issues for bugs or questions
- Submit PRs for improvements

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/

# Format
black src/ tests/
```

---

## License

MIT License - Use freely, attribute kindly.

---

## Acknowledgments

- **MongoDB** for native vector search, $graphLookup, $rankFusion, $scoreFusion
- **Voyage AI** for state-of-the-art embeddings and reranking
- **Anthropic** for Claude
- **The RAG research community** for GraphRAG, RAPTOR, ColPali, CRAG papers

---

**Made with love for the AI community by Rom Iluz**

*If this helps you build better RAG systems, consider starring the repo!*
