import os
import logging 
from fastapi import HTTPException, Request
from dotenv import load_dotenv
from sqlalchemy import text, bindparam

load_dotenv()

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_TOKEN")

logger = logging.getLogger("api")

FIND_DOCUMENT = text("""
    SELECT doc_id, slug, product_line 
    FROM documents 
    WHERE slug IN :slugs
""").bindparams(
    bindparam("slugs", expanding=True)
)

def create_file_embeddings_handler(doc_slugs, request: Request):
    """POST /create_embeddings - create embeddings of the mentioned documents."""
    try:
        logger.info(f'Doc slugs: {doc_slugs}')

        result = request.state.db.execute(
            FIND_DOCUMENT,
            {"slugs", doc_slugs}
        )
        documents = result.mappings().all()
        
        return {
            "doc_slugs": doc_slugs, 
            "documents": documents,
            "message": "Embeddings generated"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))