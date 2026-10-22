import os
import logging 
from fastapi import HTTPException, Request
from dotenv import load_dotenv
from sqlalchemy import text, bindparam
from ..helper import get_pdf_text, generate_embedding
import uuid

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")

logger = logging.getLogger("api")

FIND_DOCUMENT = text("""
    SELECT doc_id, slug, title, product_line, storage_path 
    FROM documents 
    WHERE slug IN :slugs
""").bindparams(
    bindparam("slugs", expanding=True)
)

def create_file_embeddings_handler(doc_slugs, request: Request):
    """POST /create_embeddings - create embeddings of the mentioned documents."""

    db = request.app.state.db()
    results = []

    try:
        logger.info(f'Doc slugs: {doc_slugs}')
        result = db.execute(
            FIND_DOCUMENT,
            {"slugs": doc_slugs}
        )
        documents = result.mappings().all()

        for document in documents:

            slug = document['slug']

            try:

                # Read PDF from file path
                raw_text = get_pdf_text(document['storage_path'])

                # Unique ID for this ingestion operation.
                ingest_session_id = str(uuid.uuid4())

                generate_embedding(request, raw_text, document, ingest_session_id)

                results.append({
                    "slug": slug,
                    'status': 'success',
                    'message': 'Embeddings generated'
                })

            except HTTPException as e:
                raise e
            except Exception as e:
                logger.exception(f"Embeddings generation failed for {slug}")
                results.append({
                    "slug": slug,
                    'status': 'failed',
                    'message': e
                })
                # raise HTTPException(status_code=400, detail=str(e)) from e            
        
        success = True
        for result in results:
            if result['status'] == 'failed': 
                success = False 
                break

        return {
            "status": success, 
            "data": results
        }
    
    finally: 
        db.close()