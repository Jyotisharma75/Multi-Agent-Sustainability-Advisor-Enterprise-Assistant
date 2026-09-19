"""Document ingestion: parse, chunk, embed, index and catalogue.

Unchanged documents (same checksum) are skipped unless ``force`` is set, so
ingestion is idempotent. Run as ``sa-ingest`` or ``python -m
sustainability_advisor.retrieval.ingest``.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from pydantic import BaseModel

from sustainability_advisor.config.models import ChunkingConfig
from sustainability_advisor.observability.logging import get_logger
from sustainability_advisor.repositories.base import utcnow
from sustainability_advisor.repositories.documents import DocumentRepository
from sustainability_advisor.retrieval.chunking import parse_document
from sustainability_advisor.retrieval.embeddings import EmbeddingProvider
from sustainability_advisor.retrieval.index import SearchIndex

logger = get_logger(__name__)


class IngestReport(BaseModel):
    documents_indexed: int
    documents_skipped: int
    chunks_indexed: int


async def ingest_folder(
    folder: Path,
    *,
    chunking: ChunkingConfig,
    embeddings: EmbeddingProvider,
    index: SearchIndex,
    documents: DocumentRepository,
    force: bool = False,
    in_memory_index: bool = False,
) -> IngestReport:
    indexed = skipped = chunk_total = 0
    paths = await asyncio.to_thread(lambda: sorted(folder.glob("*.md")))
    for path in paths:
        parsed = parse_document(path, chunking)
        existing = await asyncio.to_thread(documents.checksum, parsed.record.document_id)
        # An in process index starts empty, so it is always rebuilt.
        if existing == parsed.record.checksum and not force and not in_memory_index:
            skipped += 1
            continue
        vectors = await embeddings.embed([c.content for c in parsed.chunks])
        await index.delete_document(parsed.record.document_id)
        await index.upsert(parsed.chunks, vectors)
        record = parsed.record.model_copy(update={"indexed_at": utcnow()})
        await asyncio.to_thread(documents.upsert, record)
        indexed += 1
        chunk_total += len(parsed.chunks)
        logger.info("ingest.document", document_id=record.document_id, chunks=len(parsed.chunks))
    return IngestReport(
        documents_indexed=indexed, documents_skipped=skipped, chunks_indexed=chunk_total
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index sustainability documents.")
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from sustainability_advisor.container import build_container

    async def run() -> None:
        container = build_container()
        folder = args.folder or Path(container.settings.retrieval.documents_path)
        report = await ingest_folder(
            folder,
            chunking=container.settings.retrieval.chunking,
            embeddings=container.embeddings,
            index=container.search_index,
            documents=container.repositories.documents,
            force=args.force,
            in_memory_index=container.settings.retrieval.backend == "memory",
        )
        print(report.model_dump_json(indent=2))
        container.close()

    asyncio.run(run())


if __name__ == "__main__":
    main()
