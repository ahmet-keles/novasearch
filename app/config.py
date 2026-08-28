from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from NOVA_-prefixed environment variables.

    embedding_dim must match the vector dimension fixed in the database
    migrations (alembic/versions): changing one without the other makes
    inserts fail at the database, which is the safe direction.
    """

    model_config = SettingsConfigDict(env_prefix="NOVA_", env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://nova:nova@localhost:5442/novasearch"
    redis_url: str = "redis://localhost:6389/0"

    embedding_dim: int = 256

    chunk_max_words: int = 200
    chunk_overlap_words: int = 40

    # How many candidates each retriever contributes before hybrid fusion.
    search_candidate_pool: int = 50

    search_cache_ttl_seconds: int = 30


@lru_cache
def get_settings() -> Settings:
    return Settings()
