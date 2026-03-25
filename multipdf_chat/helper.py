import asyncio
import json
import uuid
from fastapi import HTTPException, Request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_experimental.text_splitter import SemanticChunker
from langchain_groq import ChatGroq
import boto3 
from botocore.exceptions import ClientError
import os
import mimetypes
import tempfile
import logging
import sys
from pythonjsonlogger import jsonlogger
from sqlalchemy import text
import watchtower 
import os 
import io
from dotenv import load_dotenv
from multipdf_chat.api.StreamingHandler import StreamingHandler
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.docstore.document import Document

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

def get_parent_child_splitters(request: Request):
    embeddings = request.app.state.embeddings

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=200
    )
    child_splitter = SemanticChunker(
        embeddings, 
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=90
    )
    return parent_splitter, child_splitter

def get_text_chunks(text, type, request: Request):
    embeddings = request.app.state.embeddings
    if type == "recursive_char":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
    elif type == "semantic":
        splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90
        )
    chunks = splitter.split_text(text)
    return chunks 

# def generate_embedding(request: Request, chunks, session_id):
#     # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
#     embeddings = request.app.state.embeddings
#     vector_store = FAISS.from_texts(chunks, embedding=embeddings)
#     vector_store.save_local(f"faiss_index/{session_id}")

#     if S3_STORAGE_ENABLED:
#         upload_faiss_to_s3(f"faiss_index/{session_id}")

#     # put_metric('Embeddings generated', 1)

def generate_embedding(request: Request, raw_text, session_id):
    embeddings = request.app.state.embeddings
    db = request.app.state.db()

    try:

        # Create splitters
        parent_splitter, child_splitter = get_parent_child_splitters(request)

        parent_chunks = parent_splitter.split_text(raw_text)

        parent_rows = []
        child_rows = []

        # Prepare all parent child data
        for parent_chunk in parent_chunks:
            parent_id = str(uuid.uuid4())

            parent_rows.append({
                "id": parent_id,
                "content": parent_chunk,
                "metadata": json.dumps({ "session_id" : session_id })
            })

            child_chunks = child_splitter.split_text(parent_chunk)

            for child_chunk in child_chunks:
                child_rows.append({
                    "id": str(uuid.uuid4()),
                    "parent_id": parent_id,
                    "content": child_chunk,
                    "metadata": json.dumps({ "session_id" : session_id })
                })

        # Generate embeddings in batch 
        contents = [row['content'] for row in child_rows]
        embeddings_list = embeddings.embed_documents(contents)

        # Attach embeddings 
        for i, row in enumerate(child_rows):
            row['embedding'] = str(embeddings_list[i])

        # Bulk Insert parents 
        db.execute(
            text("""
                INSERT INTO parent_documents (id, content, metadata) 
                VALUES (:id, :content, :metadata) 
                """),
            parent_rows
        )

        # Bulk insert children 
        db.execute(
            text(
                """
                INSERT INTO child_chunks 
                (id, parent_id, content, embedding, metadata) 
                VALUES (:id, :parent_id, :content, :embedding, :metadata)
                """
            ),
            child_rows
        )

        # for parent_chunk in parent_chunks:
        #     parent_id = str(uuid.uuid4())

        #     # Insert parent
        #     db.execute(
        #         text("""
        #         INSERT INTO parent_documents (id, content, metadata) 
        #         VALUES (:id, :content, :metadata)
        #         """),
        #         {
        #             "id": parent_id,
        #             "content": parent_chunk,
        #             "metadata": json.dumps({"session_id": session_id})
        #         }
        #     )

        #     # Child chunks 
        #     child_chunks = child_splitter.split_text(parent_chunk)

        #     for child_chunk in child_chunks:
        #         embedding = embeddings.embed_query(child_chunk)
        #         db.execute(
        #             text("""
        #             INSERT INTO child_chunks 
        #             (id, parent_id, content, embedding, metadata) 
        #             VALUES 
        #             (:id, :parent_id, :content, :embedding, :metadata)
        #             """),
        #             {
        #                 "id": str(uuid.uuid4()), 
        #                 "parent_id": parent_id, 
        #                 "content": child_chunk, 
        #                 "embedding": embedding, 
        #                 "metadata": json.dumps({ "session_id": session_id })
        #             }
        #         )

        db.commit()

    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()

def generate_embedding_FAISS(request: Request, raw_text, session_id):
    embeddings = request.app.state.embeddings

    # Create splitters
    parent_splitter, child_splitter = get_parent_child_splitters(request)
    # Create document
    docs = [
        Document(
            page_content=raw_text, 
            metadata={"doc_id": session_id}
        )
    ]
    # Vector store - child chunks
    vector_store = FAISS.from_documents([], embedding=embeddings)
    # Parent document store 
    docstore = InMemoryStore()

    # Create retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    retriever.add_documents(docs)

    # Save vector store
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
    embeddings = request.app.state.embeddings
    db = request.app.state.db() 

    """
        Steps:
        1) Convert user query → embedding
        2) Run SQL → get top chunks in child_chunks table
        3) Extract parent_id
        4) Fetch full documents from parent_documents table
        5) Send to LLM
    """

    try:
        query_embeddings = embeddings.embed_query(user_question)

        # Step 1: Vector search 
        result = db.execute(
            text("""
                SELECT parent_id FROM child_chunks 
                WHERE metadata ->> 'session_id' = :session_id
                ORDER BY embedding <=> :embedding::vector 
                LIMIT 5
            """),
            {
                "session_id": session_id,
                "embedding": query_embeddings
            }
        )

        rows = result.fetchall()

        parent_ids = list(set([r[0] for r in rows]))

        if not parent_ids:
            return "No relevant data"
        
        # Step 2: Fetch parents 
        result = db.execute(
            text("""
                SELECT content FROM parent_documents 
                 WHERE id = ANY(:ids)
            """),
            {
                "ids": parent_ids
            }
        )
        parent_docs = [row[0] for row in result.fetchall()]

        # Step 3: Get conversational chain 
        chain = get_conversational_chain(streaming=False, callbacks=None)

        response = chain(
            { "input_documents": parent_docs, "question": user_question },
            return_only_outputs=True
        )
        return response['output_text']

    except Exception as e:
        raise e
    finally:
        db.close()

def user_input_FAISS(user_question, session_id, request):
    # put_metric('User queries entered', 1)
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") 
    embeddings = request.app.state.embeddings
    
    faiss_folder = f"faiss_index/{session_id}"
    if S3_STORAGE_ENABLED:
        faiss_folder = download_faiss_from_s3(f"faiss_index/{session_id}")
    
    # Load vector stores (child chunks)
    vector_store = FAISS.load_local(
        faiss_folder, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    docstore = InMemoryStore()

    parent_splitter, child_splitter = get_parent_child_splitters(request)
    # Create retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
    # Retrieve
    docs = retriever.get_relevant_documents(user_question)
    # docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain(streaming=False, callbacks=None)

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    # logger.info(response)

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
        logger.info(f"S3 download: {bucket_name} {s3_key} -> {local_path}")
        s3.download_file(bucket_name, s3_key, local_path)

    return os.path.join(folder, tmp_dir)