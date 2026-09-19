"""Azure AI Search index adapter.

Hybrid queries send the text and a ``VectorizedQuery`` in one request so the
service performs its own reciprocal rank fusion, optionally followed by the
semantic ranker. Metadata filters are translated to OData with escaping. The
index itself is provisioned by infrastructure code, not created here; its
field names come from configuration.

Expected index fields: key (``chunk_id``), ``document_id``, ``title``,
``source_uri``, ``doc_type``, ``jurisdiction``, ``facility_id``,
``effective_date`` (Edm.DateTimeOffset), ``section``, ``page``, ``ordinal``,
the configured content field and the configured vector field.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sustainability_advisor.config.loader import read_secret
from sustainability_advisor.config.models import AzureSearchConfig
from sustainability_advisor.domain.errors import RetrievalError
from sustainability_advisor.retrieval.filters import to_odata
from sustainability_advisor.retrieval.index import SearchIndex
from sustainability_advisor.retrieval.models import DocumentChunk, RetrievedChunk, SearchFilters

_SELECT = [
    "chunk_id",
    "document_id",
    "title",
    "source_uri",
    "doc_type",
    "jurisdiction",
    "facility_id",
    "effective_date",
    "section",
    "page",
    "ordinal",
]


class AzureAISearchIndex(SearchIndex):
    def __init__(self, config: AzureSearchConfig) -> None:
        self._config = config
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents.aio import SearchClient

            endpoint = read_secret(self._config.endpoint_env)
            if not endpoint:
                raise RetrievalError(f"{self._config.endpoint_env} is not set.")
            credential: Any
            if self._config.auth_mode == "entra":
                from azure.identity.aio import DefaultAzureCredential

                credential = DefaultAzureCredential()
            else:
                key = read_secret(self._config.api_key_env)
                if not key:
                    raise RetrievalError(f"{self._config.api_key_env} is not set.")
                credential = AzureKeyCredential(key)
            self._client = SearchClient(endpoint, self._config.index_name, credential)
        return self._client

    def _to_document(self, chunk: DocumentChunk, vector: list[float]) -> dict[str, Any]:
        data = chunk.model_dump()
        content = data.pop("content")
        if chunk.effective_date:
            data["effective_date"] = f"{chunk.effective_date.isoformat()}T00:00:00Z"
        data[self._config.content_field] = content
        data[self._config.vector_field] = vector
        return data

    def _to_chunk(self, doc: dict[str, Any]) -> DocumentChunk:
        effective = doc.get("effective_date")
        if isinstance(effective, datetime):
            effective = effective.date()
        elif isinstance(effective, str):
            effective = date.fromisoformat(effective[:10])
        return DocumentChunk(
            chunk_id=str(doc["chunk_id"]),
            document_id=str(doc["document_id"]),
            title=str(doc["title"]),
            source_uri=str(doc["source_uri"]),
            doc_type=str(doc["doc_type"]),
            jurisdiction=doc.get("jurisdiction"),
            facility_id=doc.get("facility_id"),
            effective_date=effective,
            section=doc.get("section"),
            page=doc.get("page"),
            ordinal=int(doc.get("ordinal") or 0),
            content=str(doc.get(self._config.content_field, "")),
        )

    async def upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        client = self._get_client()
        documents = [self._to_document(c, v) for c, v in zip(chunks, vectors, strict=True)]
        try:
            results = await client.merge_or_upload_documents(documents=documents)
        except Exception as exc:
            raise RetrievalError(f"Index upload failed: {type(exc).__name__}") from exc
        failed = [r.key for r in results if not r.succeeded]
        if failed:
            raise RetrievalError("Some chunks failed to index.", details={"keys": failed[:20]})

    async def delete_document(self, document_id: str) -> None:
        client = self._get_client()
        escaped = document_id.replace("'", "''")
        keys = []
        results = await client.search(
            search_text="*", filter=f"document_id eq '{escaped}'", select=["chunk_id"]
        )
        async for item in results:
            keys.append({"chunk_id": item["chunk_id"]})
        if keys:
            await client.delete_documents(documents=keys)

    async def _search(
        self, text: str | None, vector: list[float], filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        from azure.search.documents.models import VectorizedQuery

        client = self._get_client()
        kwargs: dict[str, Any] = {
            "search_text": text,
            "vector_queries": [
                VectorizedQuery(
                    vector=vector, k_nearest_neighbors=top_k, fields=self._config.vector_field
                )
            ],
            "filter": to_odata(filters),
            "select": [*_SELECT, self._config.content_field],
            "top": top_k,
        }
        use_semantic = bool(
            text and self._config.use_semantic_ranker and self._config.semantic_configuration
        )
        if use_semantic:
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = self._config.semantic_configuration
        try:
            results = await client.search(**kwargs)
            hits: list[RetrievedChunk] = []
            rank = 0
            async for doc in results:
                rank += 1
                reranker = doc.get("@search.reranker_score")
                if use_semantic and reranker is not None:
                    score = float(reranker) / self._config.reranker_score_normalizer
                else:
                    score = float(doc.get("@search.score", 0.0)) / self._config.score_normalizer
                hits.append(
                    RetrievedChunk(
                        chunk=self._to_chunk(doc),
                        score=max(0.0, min(score, 1.0)),
                        vector_rank=rank if text is None else None,
                    )
                )
            return hits
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Azure AI Search query failed: {type(exc).__name__}") from exc

    async def hybrid_search(
        self, query: str, vector: list[float], filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        return await self._search(query, vector, filters, top_k)

    async def vector_search(
        self, vector: list[float], filters: SearchFilters, top_k: int
    ) -> list[RetrievedChunk]:
        return await self._search(None, vector, filters, top_k)

    async def count(self) -> int:
        client = self._get_client()
        try:
            return int(await client.get_document_count())
        except Exception as exc:
            raise RetrievalError(f"Azure AI Search is unreachable: {type(exc).__name__}") from exc
