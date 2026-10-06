
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    async_sessionmaker, 
    AsyncSession
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

ASYNC_DATABASE_URL = DATABASE_URL.replace(
    "postgresql+psycopg2://",
    "postgresql+psycopg://",
    1
)

print('Async DB URL: ', ASYNC_DATABASE_URL)

engine = create_engine(
    os.getenv('DATABASE_URL'),
    pool_size=5,
    max_overflow=10
)

async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=10,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
