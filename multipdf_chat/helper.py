import asyncio
from fastapi import Request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
import boto3 
from botocore.exceptions import ClientError
import os
import mimetypes
import tempfile
import logging
import sys
from pythonjsonlogger import jsonlogger
import watchtower 
import os 
import io
from dotenv import load_dotenv
from multipdf_chat.api.StreamingHandler import StreamingHandler

load_dotenv()

S3_STORAGE_ENABLED = os.getenv('S3_STORAGE_ENABLED')

def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s"
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # pdf_reader = PdfReader(pdf)
        # Read file bytes 
        pdf_bytes = pdf.file.read()
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in pdf_reader.pages:
            text += page.extract_text()
        # Reset pointer - important if file is read again 
        pdf.file.seek(0)
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks 

def generate_embedding(request: Request, chunks, session_id):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    embeddings = request.app.state.embeddings
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(f"faiss_index/{session_id}")

    if S3_STORAGE_ENABLED:
        upload_faiss_to_s3(f"faiss_index/{session_id}")

    # put_metric('Embeddings generated', 1)

def get_conversational_chain(streaming=False, callbacks=None):

    prompt_template = """
    Answer  the question as detailed as possible from the provided context. If answer is not within the provided content, just mention "Content not available".
    Context:\n{context}\n 
    Question:\n{question}\n

    Answer:\n
    """

    model = ChatGroq(
        model="llama-3.1-8b-instant", 
        groq_api_key=os.getenv('GROQ_API_KEY'),
        streaming=streaming,
        callbacks=callbacks
    )

    prompt = PromptTemplate(
        template=prompt_template, 
        input_variables=["context", "question"]
    )

    # Build a ready-made question answering chain on top of your documents and LLM
    qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return qa_chain

def user_input(user_question, session_id, request):
    # put_metric('User queries entered', 1)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    embeddings = request.app.state.embeddings
    
    faiss_folder = f"faiss_index/{session_id}"
    if S3_STORAGE_ENABLED:
        faiss_folder = download_faiss_from_s3(f"faiss_index/{session_id}")
    
    vector_store = FAISS.load_local(faiss_folder, embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain(streaming=False, callbacks=None)

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    logger.info(response)

    return response['output_text']

    # put_metric('User queries answered', 1)

async def stream_user_input(request: Request, user_question, session_id):
    # put_metric('User queries entered', 1)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    embeddings = request.app.state.embeddings
    faiss_folder = f"faiss_index/{session_id}"
    if S3_STORAGE_ENABLED:
        faiss_folder = download_faiss_from_s3(f"faiss_index/{session_id}")
    
    vector_store = FAISS.load_local(
        faiss_folder, embeddings, allow_dangerous_deserialization=True
    )
    docs = vector_store.similarity_search(user_question)

    handler = StreamingHandler()

    chain = get_conversational_chain(streaming=True, callbacks=[handler])

    # Run chain in the background
    asyncio.create_task(
        chain.ainvoke({
            "input_documents": docs, 
            "question": user_question
        })
    )

    buffer = ""

    while (True):
        token = await handler.queue.get() 
        if token is None: 
            break
        if not token or token.strip() == "":
            continue

        buffer += token 
        if len(buffer) > 20:
            yield buffer
            buffer = "" 

    # put_metric('User queries answered', 1)

    if buffer:
        yield buffer

def upload_faiss_to_s3(folder):
    bucket_name = os.getenv('AWS_S3_UPLOAD_BUCKET')
    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE'))
    s3 = session.client('s3')

    try:
        for filename in os.listdir(folder):
            local_path = os.path.join(folder, filename)
            s3_key = f"{folder}/{filename}"
            s3.upload_file(local_path, bucket_name, s3_key)
    except ClientError as e:
        logger.info(e)
        return False    
    return True

def download_faiss_from_s3(folder):
    bucket_name = os.getenv('AWS_S3_UPLOAD_BUCKET')
    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE'))
    s3 = session.client('s3')
    tmp_dir = tempfile.mkdtemp()

    for filename in ["index.faiss", "index.pkl"]:
        s3_key = f"{folder}/{filename}"
        local_path = os.path.join(tmp_dir, filename)
        logger.info("S3 download:", bucket_name, s3_key, "->", local_path)
        s3.download_file(bucket_name, s3_key, local_path)

    return os.path.join(folder, tmp_dir)