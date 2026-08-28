
import logging 
from fastapi import HTTPException

logger = logging.getLogger("api")

def create_file_embeddings_handler(doc_slugs, request):
    try:
        logger.info(f'Doc slugs: {doc_slugs}')
        return {
            "doc_slugs": doc_slugs, 
            "message": "Embeddings generated"
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))