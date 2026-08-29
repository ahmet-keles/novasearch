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
              │  └ vector(384) ── HNSW index (semantic)
              └─────────────┘
```

**Ingestion** (`POST /documents`): the document is split by deterministic
word-window chunking (fixed window and overlap over normalized whitespace —
identical input always produces identical chunks), each chunk is embedded,
and document plus chunks are written in a single transaction. A `tsvector`
is a stored generated column, so the keyword index can never drift from the
chunk text. Only chunks with indexable tokens (lowercase alphanumerics —
one shared tokenizer defines this for embedding, ingestion, and query
validation) are stored: a punctuation-only window would embed to a zero
vector that cosine distance cannot rank, so it is dropped, and a document
with no indexable tokens at all is rejected with 422. Search queries
without indexable tokens are likewise rejected with 422 in every mode
rather than pretending they are searchable.

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

**Embeddings** are behind a small `EmbeddingProvider` interface with two
implementations, selected by `NOVA_EMBEDDING_PROVIDER`:

- `hashing` (default) — a deterministic **feature-hashing bag-of-words
  baseline** (BLAKE2b bucket + sign per token, L2-normalized): no model
  download, no network, hermetic tests. It is *not* a semantic model —
  texts score as similar when they share vocabulary.
- `model` — a real **sentence-transformer model**
  (`sentence-transformers/all-MiniLM-L6-v2`, 384-dim by default) served on
  CPU via ONNX (`fastembed`), installed with the optional `[model]` extra
  and downloaded once on first use. This is what makes paraphrases match
  without shared vocabulary; the evaluation fixtures in
  `tests/fixtures/semantic_eval.json` measure exactly that, and also pin
  that the hashing baseline *cannot* solve them.

Both embed in batches and both must match the pgvector schema: the
application validates the provider's measured output dimension against the
live `chunks.embedding` column at startup and refuses to start on a
mismatch. Embeddings are provider-specific — one deployment has one
embedding space, and switching it requires a migration plus re-ingestion
(migration `0003`, which moved the schema to 384 dimensions, is exactly
that).

**Caching**: search responses are cached in Redis with a short TTL, under
keys that embed an **invalidation epoch owned by PostgreSQL** (the
single-row `cache_epoch` table). Ingestion increments the epoch *in the
same transaction* as the document write, so invalidation commits
atomically with the data and can never be lost. Each search reads the
epoch once and uses it for both its cache lookup and its cache write, so a
search racing an ingestion can only write under the already-retired epoch —
stale results can never surface under the new one. Redis is purely a
disposable cache, never the source of truth: if it is down, ingestion and
(uncached) search keep working and only `/health` reports the outage — and
because the epoch lives in PostgreSQL, entries left in Redis from before
an outage stay unreachable after it recovers, expiring via their TTL.

## What it can and cannot do

Implemented and tested: document ingestion, deterministic chunking,
embedding storage in pgvector, model-backed semantic embeddings
(all-MiniLM-L6-v2 via ONNX) with configurable provider selection and
startup dimension validation, keyword / semantic / hybrid search,
paraphrase-retrieval evaluation fixtures, Redis response caching with
write invalidation, health checks, Alembic migrations, CI (one job on the
hashing baseline, one running the full suite on the model provider).

Not implemented (see roadmap — the API will not pretend otherwise): answer
generation / RAG, reranking, embedding versioning / online re-indexing,
authentication, document updates and deletion, pagination.

## Local setup

Requirements: Python 3.11+, Docker.

```bash
docker compose up -d --wait          # PostgreSQL (pgvector) on :5442, Redis on :6389

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"              # add ",model" for the sentence-transformer provider

alembic upgrade head                 # create the schema
uvicorn app.main:app --reload        # serve on :8000
```

To search with real semantic embeddings, install the extra and select the
provider (the model, ~90 MB, downloads once on first use):

```bash
pip install -e ".[dev,model]"
NOVA_EMBEDDING_PROVIDER=model uvicorn app.main:app --reload
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

Tests marked `model` exercise the sentence-transformer provider and its
retrieval quality; they skip unless the `[model]` extra is installed and
the model is available (CI's *model embeddings* job makes them mandatory
and additionally runs the entire integration suite with
`NOVA_EMBEDDING_PROVIDER=model`).

Integration tests run against real PostgreSQL and Redis: they apply the
Alembic migrations, ingest documents over the HTTP API, and assert
persisted rows, generated tsvectors, embedding dimensions, ranking
behavior per mode, and cache invalidation. CI (GitHub Actions) runs lint
plus the full suite against pgvector and Redis service containers on every
push and pull request.

## Roadmap

1. **Embedding lifecycle** — embedding versioning and online re-indexing,
   so the embedding space can change without a destructive migration;
   hosted embedding APIs as further providers.
2. **Retrieval quality** — configurable fusion weights, a broader relevance
   evaluation corpus, optional cross-encoder reranking.
3. **Document lifecycle** — update and delete with index and cache
   consistency, batch ingestion, pagination.
4. **RAG layer** — grounded answer generation over retrieved chunks
   (explicitly not part of this milestone).
5. **Operations** — metrics and tracing, request auth, rate limiting,
   container image for the API itself.
