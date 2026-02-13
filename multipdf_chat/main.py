from fastapi import File, FastAPI, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_huggingface import HuggingFaceEmbeddings
# from fastapi.exceptions import RequestValidationError
from multipdf_chat.api.upload import upload_handler
from multipdf_chat.api.query import query_answer
from typing import List

from multipdf_chat.helper import stream_user_input
from multipdf_chat.models.userQuery import UserQuery

app = FastAPI()

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

@app.on_event("startup")
def load_models():
    app.state.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

@app.get('/')
def home():
    return { 'data': 'Welcome to Fast API home' }

@app.post('/upload')
def uploadFile(files: List[UploadFile] = File(...)):
    print('Uploaded files: ', files)
    return upload_handler(files)

@app.post('/user_query')
def userQuery(userQuery: UserQuery):
    print(userQuery)
    return query_answer(userQuery.user_question, userQuery.session_id)

@app.post('/chat/stream')
def userQuery(userQuery: UserQuery, request: Request):
    print(userQuery)
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