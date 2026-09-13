from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # PostgreSQL (used by PostgresSaver checkpointer + admin access)
    database_url: str

    # Read-only agent credentials (role is created by resources/sql/init_db.sql)
    postgres_user: str
    postgres_password: str
    postgres_host: str
    postgres_port: int
    postgres_db: str

    # ChromaDB vector store
    chroma_host: str
    chroma_port: int
    chroma_collection: str = "enterprise_knowledge_base"

    # LLM
    llm_api_key: str
    llm_base_url: str
    llm_model: str

    @property
    def readonly_db_uri(self) -> str:
        # NOTE: "+psycopg" is required so SQLAlchemy uses the psycopg3 driver
        # (a bare "postgresql://" URL selects the psycopg2 dialect, which is
        # not installed).
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def checkpoint_pool_uri(self) -> str:
        return self.database_url


settings = Settings()
