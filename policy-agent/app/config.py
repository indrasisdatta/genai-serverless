"""Configuration settings for the policy agent."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # LLM Configuration
    groq_api_key: str = ""
    llm_provider: str = "groq"  # groq | ollama
    fast_model: str = "openai/gpt-oss-20b"
    strong_model: str = "openai/gpt-oss-120b"
    ollama_model: str = "llama3.1:8b"

    # Embedding Configuration
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    rerank_model: str = "BAAI/bge-reranker-base"

    # Database Configuration
    database_url: str = "postgresql://app:app@localhost:5432/policy"

    # Optional Langfuse Configuration
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: Optional[str] = None

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()