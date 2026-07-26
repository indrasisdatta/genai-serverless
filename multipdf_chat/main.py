from fastapi import File, FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
from requests import Session
# from fastapi.exceptions import RequestValidationError
from multipdf_chat.api.upload import upload_handler
from multipdf_chat.api.query import query_answer
from typing import List

from multipdf_chat.db import SessionLocal, get_db
from multipdf_chat.helper import setup_logging, stream_user_input
from multipdf_chat.models.userQuery import UserQuery

import logging 
import sys 
import time 
import uuid 
from pythonjsonlogger import jsonlogger

app = FastAPI()

setup_logging()

logger = logging.getLogger("api")

origins = [
    "http://localhost:3000",  "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path
        }
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.error(
            "request_failed",
            exc_info=True,
            extra={"request_id": request_id}
        )
        raise

    duration = time.time() - start_time 

    logger.info(
        "request_completed",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "duration": round(duration, 3)
        }
    )

    response.headers["X-Request-ID"] = request_id
    return response
    

@app.on_event("startup")
def load_models():
    app.state.db = SessionLocal
    app.state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@app.get('/')
def home():
    return { 'data': 'Welcome to Fast API home' }

@app.post('/upload')
def uploadFile(request: Request, files: List[UploadFile] = File(...)):
    logger.info(f'Uploaded files: {files}')
    return upload_handler(files, request)

@app.post('/user_query')
def userQuery(userQuery: UserQuery, request: Request):
    # logger.info(f'User query: {userQuery}')
    return query_answer(userQuery.user_question, userQuery.session_id, request)

@app.post('/chat/stream')
def streamUserQuery(userQuery: UserQuery, request: Request):
    logger.info(f"User query: {userQuery}")
    return StreamingResponse(
        stream_user_input(
            request, 
            userQuery.user_question, 
            userQuery.session_id
        ),
        media_type="text/plain"
    )

# import uvicorn
# uvicorn.run(app)