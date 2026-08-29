from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from NOVA_-prefixed environment variables.

    embedding_dim must match the vector dimension fixed by the latest
    database migration (alembic/versions): mismatches are caught at
    startup by the schema validation, and by the database itself at
    insert time — both safe directions.
    """

    model_config = SettingsConfigDict(env_prefix="NOVA_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://nova:nova@localhost:5442/novasearch"
    redis_url: str = "redis://localhost:6389/0"

    embedding_dim: int = 384

    # "hashing": deterministic lexical baseline, no dependencies beyond the
    # standard library — the default, and what the hermetic tests use.
    # "model": a real sentence-transformer model served via ONNX (fastembed);
    # requires `pip install -e ".[model]"` and a one-time model download.
    embedding_provider: Literal["hashing", "model"] = "hashing"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 32
    # Where the model provider stores downloaded model files (None: fastembed
    # default). CI points this at a persisted cache directory.
    embedding_cache_dir: str | None = None

    chunk_max_words: int = 200
    chunk_overlap_words: int = 40

    # How many candidates each retriever contributes before hybrid fusion.
    search_candidate_pool: int = 50

    search_cache_ttl_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
