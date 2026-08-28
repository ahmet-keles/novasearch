# NovaSearch

A hybrid search engine: PostgreSQL full-text search and pgvector semantic
search behind one FastAPI service, fused with Reciprocal Rank Fusion.

## The problem

Keyword search and semantic search fail in opposite ways. Full-text search
nails exact terms — identifiers, product names, error codes — but misses
paraphrases; vector search catches related wording but can rank a fuzzy
match above an exact one. Production search engines therefore run both and
merge the results. NovaSearch is a small, testable implementation of that
pattern on boring, reliable infrastructure: one PostgreSQL database serves
both retrievers, so there is no separate search cluster to operate or keep
in sync.

## Architecture

```
                 ┌──────────────────────────────┐
  POST /documents│           FastAPI            │ GET /search?q=…&mode=…
  ───────────────►  chunking → embedding        ◄───────────────────────
                 │  ingestion   search   cache  │
                 └───────┬──────────┬───────────┘
                         │          │
              ┌──────────▼──┐   ┌───▼───────┐
              │ PostgreSQL  │   │   Redis   │
              │  + pgvector │   │ (response │
              │             │   │   cache)  │
              │ documents   │   └───────────┘
              │ chunks      │
              │  ├ tsvector ── GIN index (keyword)
              │  └ vector(256) ── HNSW index (semantic)
              └─────────────┘
```

**Ingestion** (`POST /documents`): the document is split by deterministic
word-window chunking (fixed window and overlap over normalized whitespace —
identical input always produces identical chunks), each chunk is embedded,
and document plus chunks are written in a single transaction. A `tsvector`
is a stored generated column, so the keyword index can never drift from the
chunk text.

**Search** (`GET /search`) has three modes:

- `keyword` — PostgreSQL full-text search (`websearch_to_tsquery`, ranked
  by `ts_rank_cd`, GIN-indexed).
- `semantic` — pgvector cosine nearest-neighbour over chunk embeddings
  (HNSW-indexed).
- `hybrid` (default) — both retrievers each contribute a candidate pool,
  merged with **Reciprocal Rank Fusion**: a chunk's score is
  `Σ 1 / (60 + rank)` across the rankings that contain it. RRF operates on
  ranks, not raw scores, so cosine similarity and `ts_rank` never need to
  be calibrated against each other.

**Embeddings** are behind a small `EmbeddingProvider` interface. The
shipped implementation is a deterministic **feature-hashing bag-of-words
baseline** (BLAKE2b bucket + sign per token, L2-normalized): no model
download, no network, hermetic tests. It is *not* a semantic model — texts
score as similar when they share vocabulary. Model-backed providers (local
sentence-transformers, hosted embedding APIs) are the designed next step
and plug in behind the same interface without touching the search code.

**Caching**: search responses are cached in Redis under versioned keys with
a short TTL. Ingestion bumps the namespace version after commit, making all
stale entries unreachable at once. Each search captures the version once
and uses it for both its cache lookup and its cache write, so a search
racing an ingestion can only write into the already-retired namespace —
stale results can never surface under the new one. The cache fails open —
if Redis is down, search still works and only `/health` reports the outage.

## What it can and cannot do

Implemented and tested: document ingestion, deterministic chunking,
embedding storage in pgvector, keyword / semantic / hybrid search, Redis
response caching with write invalidation, health checks, Alembic
migrations, CI.

Not implemented (see roadmap — the API will not pretend otherwise): answer
generation / RAG, model-backed embeddings, reranking, authentication,
document updates and deletion, pagination.

## Local setup

Requirements: Python 3.11+, Docker.

```bash
docker compose up -d --wait          # PostgreSQL (pgvector) on :5442, Redis on :6389

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

alembic upgrade head                 # create the schema
uvicorn app.main:app --reload        # serve on :8000
```

Configuration comes from `NOVA_`-prefixed environment variables (or a
`.env` file — see `.env.example`); the defaults match the compose ports.

## API examples

```bash
# Ingest a document
curl -s -X POST localhost:8000/documents \
  -H 'content-type: application/json' \
  -d '{
        "title": "PostgreSQL indexing",
        "content": "PostgreSQL supports GIN indexes for full-text search. The pgvector extension adds vector similarity with HNSW indexes.",
        "metadata": {"source": "docs"}
      }'
# → {"id": "…", "chunk_count": 1}

# Hybrid search (default mode)
curl -s 'localhost:8000/search?q=vector+indexes&limit=5'

# Explicit modes
curl -s 'localhost:8000/search?q=vector+indexes&mode=semantic'
curl -s 'localhost:8000/search?q=vector+indexes&mode=keyword'

# Read a document back
curl -s localhost:8000/documents/<id>

# Health (503 + "degraded" if PostgreSQL or Redis is down)
curl -s localhost:8000/health
# → {"status": "ok", "database": "up", "redis": "up"}
```

Interactive API docs: `localhost:8000/docs`.

## Testing

```bash
pytest -m "not integration"   # unit tests: chunking, embeddings, rank fusion — no services needed
pytest                        # full suite; integration tests need `docker compose up -d --wait`
ruff check .
```

Integration tests run against real PostgreSQL and Redis: they apply the
Alembic migrations, ingest documents over the HTTP API, and assert
persisted rows, generated tsvectors, embedding dimensions, ranking
behavior per mode, and cache invalidation. CI (GitHub Actions) runs lint
plus the full suite against pgvector and Redis service containers on every
push and pull request.

## Roadmap

1. **Model-backed embeddings** — a real semantic `EmbeddingProvider`
   (local sentence-transformers and/or a hosted API), with embedding
   versioning and re-indexing.
2. **Retrieval quality** — configurable fusion weights, a small relevance
   evaluation harness, optional cross-encoder reranking.
3. **Document lifecycle** — update and delete with index and cache
   consistency, batch ingestion, pagination.
4. **RAG layer** — grounded answer generation over retrieved chunks
   (explicitly not part of this milestone).
5. **Operations** — metrics and tracing, request auth, rate limiting,
   container image for the API itself.
