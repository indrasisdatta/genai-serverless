"""Per-request context propagated via contextvars — safe under asyncio."""
import contextvars
from typing import Optional 

_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", 
    default=None
)

def set_request_id(rid: str) -> None: 
    _request_id.set(rid)

def get_request_id() -> Optional[str]:
    return _request_id.get()
