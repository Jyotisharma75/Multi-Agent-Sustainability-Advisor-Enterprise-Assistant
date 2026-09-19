"""Document retrieval service.

Embeds the query, runs hybrid or vector search against the configured index,
drops results under the minimum relevance score, drops passages that look like
indirect prompt injection, and turns what remains into citations whose title
and source come from the document catalogue in Azure SQL.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Literal

from sustainability_advisor.config.models import RetrievalConfig
from sustainability_advisor.domain.models import Citation
from sustainability_advisor.observability.telemetry import record_retrieval
from sustainability_advisor.repositories.documents import DocumentRepository
from sustainability_advisor.retrieval.embeddings import EmbeddingProvider
from sustainability_advisor.retrieval.filters import validate_filters
from sustainability_advisor.retrieval.index import SearchIndex
from sustainability_advisor.retrieval.models import RetrievalResult, RetrievedChunk, SearchFilters

_EXCERPT_CHARS = 600


class DocumentRetriever:
    def __init__(
        self,
        config: RetrievalConfig,
        index: SearchIndex,
        embeddings: EmbeddingProvider,
        documents: DocumentRepository,
        is_suspicious: Callable[[str], bool],
    ) -> None:
        self._config = config
        self._index = index
        self._embeddings = embeddings
        self._documents = documents
        self._is_suspicious = is_suspicious

    @property
    def index(self) -> SearchIndex:
        return self._index

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        *,
        top_k: int | None = None,
        mode: Literal["hybrid", "vector"] = "hybrid",
        min_score: float | None = None,
    ) -> RetrievalResult:
        filters = filters or SearchFilters()
        validate_filters(filters, self._config.allowed_filter_fields)
        k = min(top_k or self._config.top_k, self._config.top_k * self._config.candidate_multiplier)
        vector = await self._embeddings.embed_query(query)
        if mode == "hybrid":
            hits = await self._index.hybrid_search(query, vector, filters, k)
        else:
            hits = await self._index.vector_search(vector, filters, k)

        threshold = self._config.min_retrieval_score if min_score is None else min_score
        relevant = [h for h in hits if h.score >= threshold]
        safe = [h for h in relevant if not self._is_suspicious(h.chunk.content)]
        record_retrieval([h.score for h in safe], mode, len(safe))
        citations = await self._citations(safe)
        return RetrievalResult(
            query=query,
            mode=mode,
            chunks=safe,
            citations=citations,
            discarded_low_score=len(hits) - len(relevant),
            discarded_suspicious=len(relevant) - len(safe),
        )

    async def _citations(self, hits: list[RetrievedChunk]) -> list[Citation]:
        ids = sorted({h.chunk.document_id for h in hits})
        catalogue = await asyncio.to_thread(self._documents.get_many, ids)
        citations = []
        for hit in hits:
            chunk = hit.chunk
            record = catalogue.get(chunk.document_id)
            excerpt = chunk.content[:_EXCERPT_CHARS]
            citations.append(
                Citation(
                    citation_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=record.title if record else chunk.title,
                    source_uri=record.source_uri if record else chunk.source_uri,
                    doc_type=record.doc_type if record else chunk.doc_type,
                    jurisdiction=record.jurisdiction if record else chunk.jurisdiction,
                    effective_date=record.effective_date if record else chunk.effective_date,
                    section=chunk.section,
                    page=chunk.page,
                    chunk_id=chunk.chunk_id,
                    retrieval_score=round(hit.score, 4),
                    excerpt=excerpt,
                )
            )
        return citations
