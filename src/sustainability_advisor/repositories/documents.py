"""Document catalogue repository.

Holds the metadata of every indexed document. The search index holds chunks;
this table is the system of record for titles, types and source locations.
"""

from __future__ import annotations

from sqlalchemy import delete, select

from sustainability_advisor.domain.models import DocumentRecord
from sustainability_advisor.repositories.base import Repository


class DocumentRepository(Repository):
    def upsert(self, record: DocumentRecord) -> None:
        t = self._tables.documents
        with self._engine.begin() as conn:
            conn.execute(delete(t).where(t.c.document_id == record.document_id))
            conn.execute(t.insert().values(**record.model_dump()))

    def get_many(self, document_ids: list[str]) -> dict[str, DocumentRecord]:
        if not document_ids:
            return {}
        t = self._tables.documents
        rows = self._fetch(select(t).where(t.c.document_id.in_(document_ids)))
        records = [DocumentRecord.model_validate(dict(r._mapping)) for r in rows]
        return {r.document_id: r for r in records}

    def list_all(self) -> list[DocumentRecord]:
        t = self._tables.documents
        rows = self._fetch(select(t).order_by(t.c.document_id))
        return [DocumentRecord.model_validate(dict(r._mapping)) for r in rows]

    def checksum(self, document_id: str) -> str | None:
        t = self._tables.documents
        rows = self._fetch(select(t.c.checksum).where(t.c.document_id == document_id))
        return str(rows[0][0]) if rows else None
