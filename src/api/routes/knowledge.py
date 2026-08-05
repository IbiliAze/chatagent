"""RAG knowledge base endpoint."""

from fastapi import Request

from api.main import app, rag
from api.models.knowledge import KnowledgeAddedRequest, KnowledgeAddedResponse


@app.post('/knowledge', response_model=KnowledgeAddedResponse)
async def add_knowledge(request: Request, body: KnowledgeAddedRequest):  # pylint: disable=unused-argument
    """Add to knowledge base"""

    chunks = rag.add_texts(texts=body.texts, source=body.source)

    return KnowledgeAddedResponse(chunks=chunks)
