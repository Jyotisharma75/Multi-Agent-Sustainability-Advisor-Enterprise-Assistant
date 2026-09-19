"""Document retrieval tools: hybrid document search and pure vector search."""

from __future__ import annotations

from pydantic import BaseModel, Field

from sustainability_advisor.retrieval.models import RetrievalResult, SearchFilters
from sustainability_advisor.retrieval.retriever import DocumentRetriever
from sustainability_advisor.tools.base import Tool, ToolContext


class DocumentSearchInput(BaseModel):
    query: str = Field(min_length=2, max_length=2000)
    filters: SearchFilters | None = None
    top_k: int | None = Field(default=None, gt=0)


class DocumentSearchTool(Tool[DocumentSearchInput, RetrievalResult]):
    """Hybrid (keyword plus vector) retrieval with metadata filters and citations."""

    name = "document_search"
    input_model = DocumentSearchInput
    output_model = RetrievalResult

    def __init__(self, retriever: DocumentRetriever) -> None:
        self._retriever = retriever

    async def run(self, payload: DocumentSearchInput, ctx: ToolContext) -> RetrievalResult:
        return await self._retriever.search(
            payload.query, payload.filters, top_k=payload.top_k, mode="hybrid"
        )


class VectorSearchTool(Tool[DocumentSearchInput, RetrievalResult]):
    """Pure embedding similarity retrieval, for paraphrased or conceptual queries."""

    name = "vector_search"
    input_model = DocumentSearchInput
    output_model = RetrievalResult

    def __init__(self, retriever: DocumentRetriever) -> None:
        self._retriever = retriever

    async def run(self, payload: DocumentSearchInput, ctx: ToolContext) -> RetrievalResult:
        return await self._retriever.search(
            payload.query, payload.filters, top_k=payload.top_k, mode="vector"
        )
