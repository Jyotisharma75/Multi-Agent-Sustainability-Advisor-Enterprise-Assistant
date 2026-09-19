"""Document parsing and chunking.

Source documents are Markdown with a YAML front matter block holding their
metadata. Headings define sections and ``<!-- page: N -->`` markers carry the
page number of the original publication, so each chunk can be cited by
section and page. Chunks are built from whole paragraphs up to a size limit
with a character overlap taken from the end of the previous chunk.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

from sustainability_advisor.config.models import ChunkingConfig
from sustainability_advisor.domain.errors import InputRejectedError
from sustainability_advisor.domain.models import DocumentRecord
from sustainability_advisor.retrieval.models import DocumentChunk

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PAGE = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->\s*$", re.IGNORECASE)
_REQUIRED = ("document_id", "title", "doc_type", "source_uri")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    record: DocumentRecord
    chunks: list[DocumentChunk]


def _metadata(raw: str, path: Path) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise InputRejectedError(f"{path.name} has no front matter block")
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        raise InputRejectedError(f"{path.name} front matter must be a mapping")
    missing = [key for key in _REQUIRED if not meta.get(key)]
    if missing:
        raise InputRejectedError(f"{path.name} front matter is missing {missing}")
    return meta, raw[match.end() :]


def parse_document(path: Path, config: ChunkingConfig) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    meta, body = _metadata(raw, path)
    effective = meta.get("effective_date")
    effective_date = (
        (effective if isinstance(effective, date) else date.fromisoformat(str(effective)))
        if effective
        else None
    )
    record = DocumentRecord(
        document_id=str(meta["document_id"]),
        title=str(meta["title"]),
        doc_type=str(meta["doc_type"]),
        source_uri=str(meta["source_uri"]),
        jurisdiction=meta.get("jurisdiction"),
        facility_id=meta.get("facility_id"),
        effective_date=effective_date,
        version=str(meta["version"]) if meta.get("version") is not None else None,
        checksum=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
    )
    return ParsedDocument(record=record, chunks=chunk_body(body, record, config))


def _blocks(body: str) -> list[tuple[str | None, int | None, str]]:
    """Split the body into (section, page, paragraph) blocks."""
    section: str | None = None
    page: int | None = None
    blocks: list[tuple[str | None, int | None, str]] = []
    buffer: list[str] = []

    def flush() -> None:
        text = " ".join(line.strip() for line in buffer).strip()
        if text:
            blocks.append((section, page, text))
        buffer.clear()

    for line in body.splitlines():
        page_match = _PAGE.match(line.strip())
        heading = _HEADING.match(line)
        if page_match:
            flush()
            page = int(page_match.group(1))
        elif heading:
            flush()
            section = heading.group(2).strip()
        elif not line.strip():
            flush()
        else:
            buffer.append(line)
    flush()
    return blocks


def chunk_body(body: str, record: DocumentRecord, config: ChunkingConfig) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    current: list[str] = []
    current_section: str | None = None
    current_page: int | None = None

    def emit() -> None:
        if not current:
            return
        ordinal = len(chunks)
        chunks.append(
            DocumentChunk(
                chunk_id=f"{record.document_id}-{ordinal:04d}",
                document_id=record.document_id,
                title=record.title,
                source_uri=record.source_uri,
                doc_type=record.doc_type,
                jurisdiction=record.jurisdiction,
                facility_id=record.facility_id,
                effective_date=record.effective_date,
                section=current_section,
                page=current_page,
                ordinal=ordinal,
                content="\n".join(current),
            )
        )

    for section, page, text in _blocks(body):
        size = sum(len(p) for p in current)
        boundary = section != current_section and current
        if boundary or (current and size + len(text) > config.max_chars):
            emit()
            tail = current[-1][-config.overlap_chars :] if (current and not boundary) else ""
            current = [tail] if tail and config.overlap_chars else []
        if not current:
            current_section, current_page = section, page
        current.append(text)
    emit()
    return chunks
