"""LLM factory for creating ChatGroq or ChatOllama instances."""

from functools import lru_cache
from typing import Literal

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from app.config import settings

# Tune to your actual Groq limits (requests per second)
_limiter = InMemoryRateLimiter(requests_per_second=0.5, max_bucket_size=5)


@lru_cache()
def get_llm(tier: Literal["fast", "strong"] = "fast"):
    """
    Get an LLM instance based on the tier.

    Args:
        tier: "fast" for simple tasks, "strong" for complex reasoning

    Returns:
        ChatGroq or ChatOllama instance
    """
    if settings.llm_provider == "ollama":
        return ChatOllama(model=settings.ollama_model, temperature=0)

    model = settings.fast_model if tier == "fast" else settings.strong_model
    return ChatGroq(
        model=model,
        temperature=0,
        max_retries=3,
        rate_limiter=_limiter,
    )