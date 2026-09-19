"""Search index contract and reciprocal rank fusion."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sustainability_advisor.retrieval.models import DocumentChunk, RetrievedChunk, SearchFilters


def reciprocal_rank_fusion(
    keyword_ranking: list[str],
    vector_ranking: list[str],
    *,
    k: int,
    keyword_weight: float,
    vector_weight: float,
) -> dict[str, float]:
    """Fuse two rankings. Scores are normalised so a result ranked first by
    both retrievers scores 1.0."""
    scores: dict[str, float] = {}
    for weight, ranking in ((keyword_weight, keyword_ranking), (vector_weight, vector_ranking)):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    best_possible = (keyword_weight + vector_weight) / (k + 1)
    if best_possible <= 0:
        return dict.fromkeys(scores, 0.0)
    return {item: min(score / best_possible, 1.0) for item, score in scores.items()}


class SearchIndex(ABC):
    @abstractmethod
    async def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        """Insert or replace chunks with their embeddings."""

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Remove every chunk of a document."""

    @abstractmethod
    async def hybrid_search(
        self, query: str, vector: list[float], filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        """Keyword and vector retrieval fused into one ranking."""

    @abstractmethod
    async def vector_search(
        self, vector: list[float], filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        """Pure vector similarity retrieval."""

    @abstractmethod
    async def count(self) -> int:
        """Number of chunks in the index."""
