# GenAI Serverless

This project demonstrates a **serverless architecture** using **AWS Lambda**, **ECS**, **FastAPI**, and **Docker** for deployment.

---

### Lambda Layer Dependencies

To set up the Lambda layer dependencies:

```bash
cd multipdf-chat/lambda_layer
pip install -r requirements.txt -t python/
```

### FastAPI Setup


1. Create a virtual environment     
```bash python -m venv myvenv```

2. Activate environment (choose one based on your OS)
```bash myvenv/Scripts/activate.bat      # Windows CMD
.\myvenv\Scripts\Activate.ps1    # Windows PowerShell
```

3. Install dependencies
```bash
pip install uvicorn
pip install -r requirements.txt
```

4. Run FastAPI server 
```bash uvicorn multipdf_chat.main:app --reload```

Your FastAPI app will be running at: http://127.0.0.1:8000

### Docker Setup
⚠️ Make sure you are logged in with docker login before running these commands.

1. Build the Docker image

`docker build --compress -t multipdf-genai-fastapi .`

2. Tag the image for DockerHub

`docker tag multipdf-genai-fastapi:latest indrasisdatta/multipdf-genai-fastapi:latest`

3. Push the image to DockerHub

`docker push indrasisdatta/multipdf-genai-fastapi:latest`