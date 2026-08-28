from pydantic import BaseModel

class CreateEmbeddingPayload(BaseModel):
    doc_slugs: list[str]