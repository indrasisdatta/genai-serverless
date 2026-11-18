"""API routes - /chat (SSE), /threads/{id}, /resume."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/chat")
async def chat():
    """Streaming chat endpoint with SSE."""
    return {"status": "not implemented"}


@router.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get conversation history for a thread."""
    return {"status": "not implemented"}


@router.post("/resume")
async def resume():
    """Resume after an interrupt."""
    return {"status": "not implemented"}