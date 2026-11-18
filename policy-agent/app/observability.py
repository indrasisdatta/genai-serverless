"""Observability - Langfuse integration for tracing."""

from typing import Optional


def get_langfuse_handler() -> Optional[object]:
    """
    Get a Langfuse callback handler for tracing.

    Returns:
        Langfuse CallbackHandler or None if not configured
    """
    from app.config import settings

    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None

    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except ImportError:
        return None