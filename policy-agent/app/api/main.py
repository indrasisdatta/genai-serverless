"""FastAPI app - lifespan, SSE streaming, and routes."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initializes checkpointer and compiled graph."""
    # In production, initialize AsyncPostgresSaver here:
    # from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    # async with AsyncPostgresSaver.from_conn_string(settings.database_url) as cp:
    #     await cp.setup()
    #     app.state.graph = build_graph(checkpointer=cp)
    #     yield
    yield


app = FastAPI(lifespan=lifespan, title="Policy Agent API", version="0.1.0")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Policy Agent API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}