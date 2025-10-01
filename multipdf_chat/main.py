from fastapi import File, FastAPI, UploadFile
# from fastapi.exceptions import RequestValidationError
from multipdf_chat.api.upload import upload_handler
from multipdf_chat.api.query import query_answer
from typing import List

from multipdf_chat.models.userQuery import UserQuery

app = FastAPI()

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

# import uvicorn
# uvicorn.run(app)