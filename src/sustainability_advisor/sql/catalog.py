"""Queryable schema catalogue.

Only tables and columns listed under ``sql.exposed_tables`` in configuration
exist as far as the SQL generator and validator are concerned. Everything
else in the database, including audit and conversation tables, is invisible
to generated SQL.
"""

from __future__ import annotations

from dataclasses import dataclass

from sustainability_advisor.config.loader import ConfigurationError
from sustainability_advisor.config.models import SqlSafetyConfig
from sustainability_advisor.db.tables import Tables


@dataclass(frozen=True, slots=True)
class CatalogColumn:
    name: str
    type_name: str
    description: str


@dataclass(frozen=True, slots=True)
class CatalogTable:
    name: str
    description: str
    synonyms: tuple[str, ...]
    columns: tuple[CatalogColumn, ...]

    @property
    def column_names(self) -> frozenset[str]:
        return frozenset(c.name.lower() for c in self.columns)


class SchemaCatalog:
    def __init__(self, tables: dict[str, CatalogTable], schema_name: str | None) -> None:
        self._tables = {name.lower(): table for name, table in tables.items()}
        self.schema_name = schema_name

    @classmethod
    def build(
        cls, config: SqlSafetyConfig, tables: Tables, schema_name: str | None
    ) -> SchemaCatalog:
        result: dict[str, CatalogTable] = {}
        for name, exposed in config.exposed_tables.items():
            table = tables.metadata.tables.get(f"{schema_name}.{name}" if schema_name else name)
            if table is None:
                raise ConfigurationError(f"sql.exposed_tables lists unknown table {name!r}")
            columns = []
            for column_name in exposed.columns:
                if column_name not in table.c:
                    raise ConfigurationError(
                        f"sql.exposed_tables.{name} lists unknown column {column_name!r}"
                    )
                column = table.c[column_name]
                columns.append(
                    CatalogColumn(
                        name=column_name,
                        type_name=str(column.type),
                        description=column.comment or "",
                    )
                )
            result[name] = CatalogTable(
                name=name,
                description=exposed.description,
                synonyms=tuple(s.lower() for s in exposed.synonyms),
                columns=tuple(columns),
            )
        return cls(result, schema_name)

    def get(self, name: str) -> CatalogTable | None:
        return self._tables.get(name.lower())

    @property
    def tables(self) -> list[CatalogTable]:
        return list(self._tables.values())

    def mentioned_tables(self, question: str) -> list[CatalogTable]:
        """Tables whose name or a synonym appears in the question."""
        text = question.lower()
        found = []
        for table in self._tables.values():
            terms = (table.name.lower(), table.name.lower().replace("_", " "), *table.synonyms)
            if any(term and term in text for term in terms):
                found.append(table)
        return found

    def describe(self) -> str:
        """Render the catalogue for a generation prompt."""
        lines = []
        for table in self._tables.values():
            qualified = f"{self.schema_name}.{table.name}" if self.schema_name else table.name
            lines.append(f"TABLE {qualified}: {table.description}")
            for column in table.columns:
                suffix = f" -- {column.description}" if column.description else ""
                lines.append(f"  {column.name} {column.type_name}{suffix}")
        return "\n".join(lines)
