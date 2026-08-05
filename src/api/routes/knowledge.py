"""RAG knowledge base endpoint."""

from fastapi import APIRouter

from api.dependencies import RagDep
from api.models.knowledge import KnowledgeAddedRequest, KnowledgeAddedResponse

router = APIRouter()


@router.post('/knowledge', response_model=KnowledgeAddedResponse)
async def add_knowledge(rag: RagDep, body: KnowledgeAddedRequest):
    """Add to knowledge base"""

    chunks = rag.add_texts(texts=body.texts, source=body.source)

    return KnowledgeAddedResponse(chunks=chunks)
