"""Embedding factory for HuggingFace embeddings."""

from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache()
def get_embedder() -> HuggingFaceEmbeddings:
    """
    Get HuggingFace embeddings instance.

    Returns:
        Configured HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )