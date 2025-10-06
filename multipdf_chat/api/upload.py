from email import message
from fastapi import HTTPException
from requests import session
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# import google.generativeai as genai
from ..helper import generate_embedding, get_pdf_texts, get_text_chunks, user_input

from dotenv import load_dotenv
import os
import uuid
import json

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")

def upload_handler(pdf_files):
    try:
        if not pdf_files:
            raise HTTPException(status_code=400, detail="No PDF file is provided")
        
        session_id = str(uuid.uuid4())   
        # Extract texts from PDF
        raw_text = get_pdf_texts(pdf_files)
        chunks = get_text_chunks(raw_text)
        generate_embedding(chunks, session_id)

        return {
            "session_id": session_id, 
            "message": "Embeddings generated"
        }
    except HTTPException as e:
        raise e
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

def lambda_handler(event, context):
    try:
        session_id = str(uuid.uuid4())
        pdf_files = event.get("pdf_files", [])
        if not pdf_files:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No PDF file provided"})
            }
        raw_text = get_pdf_texts(pdf_files)
        chunks = get_text_chunks(raw_text)
        generate_embedding(chunks, session_id)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "session_id": session_id, 
                "message": "Embeddings generated"
            })
        }
    except Exception as e: 
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }