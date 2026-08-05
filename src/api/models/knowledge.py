"""Response model for the metrics endpoint."""

from pydantic import BaseModel


class KnowledgeAddedRequest(BaseModel):
    """Metrics endpoint request"""

    texts: list[str]
    source: str


class KnowledgeAddedResponse(BaseModel):
    """Metrics endpoint response"""

    chunks: int
