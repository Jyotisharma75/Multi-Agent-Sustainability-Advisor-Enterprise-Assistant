"""In process hybrid index (BM25 plus cosine similarity).

Used in development and tests in place of Azure AI Search. It implements the
same contract, metadata filtering and fusion, so agents and tools behave the
same way against either backend.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections import Counter

from sustainability_advisor.config.models import RetrievalConfig
from sustainability_advisor.retrieval.filters import matches
from sustainability_advisor.retrieval.index import SearchIndex, reciprocal_rank_fusion
from sustainability_advisor.retrieval.models import DocumentChunk, RetrievedChunk, SearchFilters

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


class InMemoryHybridIndex(SearchIndex):
    def __init__(self, config: RetrievalConfig) -> None:
        self._config = config
        self._chunks: dict[str, DocumentChunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._terms: dict[str, Counter[str]] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        async with self._lock:
            for chunk, vector in zip(chunks, vectors, strict=True):
                self._chunks[chunk.chunk_id] = chunk
                self._vectors[chunk.chunk_id] = vector
                self._terms[chunk.chunk_id] = Counter(
                    tokenize(f"{chunk.title} {chunk.section or ''} {chunk.content}")
                )

    async def delete_document(self, document_id: str) -> None:
        async with self._lock:
            for chunk_id in [c for c, ch in self._chunks.items() if ch.document_id == document_id]:
                self._chunks.pop(chunk_id, None)
                self._vectors.pop(chunk_id, None)
                self._terms.pop(chunk_id, None)

    async def count(self) -> int:
        return len(self._chunks)

    def _candidates(self, filters: SearchFilters) -> list[str]:
        return [cid for cid, chunk in self._chunks.items() if matches(chunk, filters)]

    def _bm25(self, query: str, candidates: list[str]) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms or not candidates:
            return []
        n = len(self._chunks)
        avg_len = sum(sum(c.values()) for c in self._terms.values()) / max(n, 1)
        k1, b = self._config.bm25_k1, self._config.bm25_b
        scored = []
        for cid in candidates:
            counts = self._terms[cid]
            length = sum(counts.values())
            score = 0.0
            for term in set(terms):
                tf = counts.get(term, 0)
                if tf == 0:
                    continue
                df = sum(1 for c in self._terms.values() if term in c)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg_len))
            if score > 0:
                scored.append((cid, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _vector_ranking(
        self, vector: list[float], candidates: list[str]
    ) -> list[tuple[str, float]]:
        scored = [(cid, _cosine(vector, self._vectors[cid])) for cid in candidates]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    async def hybrid_search(
        self, query: str, vector: list[float], filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        candidates = self._candidates(filters)
        depth = top_k * self._config.candidate_multiplier
        keyword = [cid for cid, _ in self._bm25(query, candidates)[:depth]]
        semantic = [cid for cid, _ in self._vector_ranking(vector, candidates)[:depth]]
        fused = reciprocal_rank_fusion(
            keyword,
            semantic,
            k=self._config.rrf_k,
            keyword_weight=self._config.keyword_weight,
            vector_weight=self._config.vector_weight,
        )
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievedChunk(
                chunk=self._chunks[cid],
                score=score,
                keyword_rank=keyword.index(cid) + 1 if cid in keyword else None,
                vector_rank=semantic.index(cid) + 1 if cid in semantic else None,
            )
            for cid, score in ranked
        ]

    async def vector_search(
        self, vector: list[float], filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        ranked = self._vector_ranking(vector, self._candidates(filters))[:top_k]
        return [
            RetrievedChunk(
                chunk=self._chunks[cid], score=max(0.0, min(score, 1.0)), vector_rank=rank
            )
            for rank, (cid, score) in enumerate(ranked, start=1)
        ]
