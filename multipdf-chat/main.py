from fastapi import File, FastAPI, UploadFile
# from fastapi.exceptions import RequestValidationError
from api.main import upload_handler

app = FastAPI()

@app.get('/')
def home():
    return { 'data': 'Welcome to Fast API home' }

@app.post('/upload')
def uploadFile(files: list[UploadFile]):
    print('Uploaded files: ', files)
    return {"filename": files[0].filename}

# import uvicorn
# uvicorn.run(app)