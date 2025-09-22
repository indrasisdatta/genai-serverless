from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
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
import watchtower 
import os 

def get_pdf_texts(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        # print(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks 

def generate_embedding(chunks, session_id):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(f"faiss_index/{session_id}")
    upload_faiss_to_s3(f"faiss_index/{session_id}")

    # put_metric('Embeddings generated', 1)

def get_conversational_chain():

    prompt_template = """
    Answer  the question as detailed as possible from the provided context. If answer is not within the provided content, just mention "Content not available".
    Context:\n{context}\n 
    Question:\n{question}\n

    Answer:\n
    """

    model = ChatGroq(model="Gemma2-9b-It", groq_api_key=os.getenv('GROQ_API_KEY'))

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Build a ready-made question answering chain on top of your documents and LLM
    qa_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return qa_chain

def user_input(user_question, session_id):
    # put_metric('User queries entered', 1)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    faiss_folder = download_faiss_from_s3(f"faiss_index/{session_id}")
    vector_store = FAISS.load_local(faiss_folder, embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)

    return response['output_text']

    # put_metric('User queries answered', 1)


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
        print(e)
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
        print("S3 download:", bucket_name, s3_key, "->", local_path)
        s3.download_file(bucket_name, s3_key, local_path)

    return os.path.join(folder, tmp_dir)