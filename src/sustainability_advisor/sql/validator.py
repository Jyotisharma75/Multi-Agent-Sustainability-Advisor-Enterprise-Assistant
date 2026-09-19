"""Semantic SQL validation and rewriting.

Runs after the guard has accepted the statement kind. It checks the query
against the exposed catalogue and the complexity ceilings, then produces the
statement that will actually run:

* every table must be an exposed table (CTE names and derived tables aside)
* every column must belong to a referenced exposed table, a CTE, a derived
  table or a projection alias
* ``SELECT *`` is refused unless configured, ``COUNT(*)`` is always fine
* join count and subquery depth are capped
* schema qualifiers are normalised onto the configured schema
* the row count is clamped to ``max_rows + 1`` so truncation is detectable
* the query is transpiled from the generation dialect to the execution one
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlglot import exp

from sustainability_advisor.config.models import SqlSafetyConfig
from sustainability_advisor.sql.catalog import SchemaCatalog


@dataclass(slots=True)
class ValidationResult:
    issues: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    display_sql: str = ""
    execution_sql: str = ""

    @property
    def ok(self) -> bool:
        return not self.issues


class SQLValidator:
    def __init__(self, config: SqlSafetyConfig, catalog: SchemaCatalog) -> None:
        self._config = config
        self._catalog = catalog
        self._accepted_qualifiers = {q.lower() for q in config.accepted_schema_qualifiers}
        if catalog.schema_name:
            self._accepted_qualifiers.add(catalog.schema_name.lower())

    def validate(
        self, expression: exp.Expr, *, execution_dialect: str, max_rows: int
    ) -> ValidationResult:
        result = ValidationResult()
        tree = expression.copy()
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        derived = {
            sub.alias_or_name.lower() for sub in tree.find_all(exp.Subquery) if sub.alias_or_name
        }

        alias_to_table: dict[str, str] = {}
        referenced: list[str] = []
        for table in tree.find_all(exp.Table):
            name = table.name.lower()
            if name in cte_names:
                continue
            catalog_table = self._catalog.get(name)
            if catalog_table is None:
                result.issues.append(f"The table {table.name} is not available for querying.")
                continue
            if table.db and table.db.lower() not in self._accepted_qualifiers:
                result.issues.append(f"The schema {table.db} is not permitted.")
                continue
            self._normalise_qualifier(table)
            referenced.append(catalog_table.name)
            alias_to_table[(table.alias_or_name or name).lower()] = catalog_table.name
            alias_to_table[name] = catalog_table.name
        if result.issues:
            return result
        result.tables = sorted(set(referenced))

        self._check_columns(tree, alias_to_table, cte_names | derived, result)
        self._check_star(tree, result)
        self._check_complexity(tree, result)
        if result.issues:
            return result

        capped = self._apply_row_cap(tree, max_rows + 1)
        result.display_sql = capped.sql(
            dialect=self._config.generation_dialect, pretty=True, comments=False
        )
        result.execution_sql = capped.sql(dialect=execution_dialect, comments=False)
        return result

    def _normalise_qualifier(self, table: exp.Table) -> None:
        schema = self._catalog.schema_name
        if schema:
            table.set("db", exp.to_identifier(schema))
        else:
            table.set("db", None)

    def _check_columns(
        self,
        tree: exp.Expr,
        alias_to_table: dict[str, str],
        virtual_sources: set[str],
        result: ValidationResult,
    ) -> None:
        projection_aliases = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
        referenced_columns: set[str] = set()
        for table_name in set(alias_to_table.values()):
            catalog_table = self._catalog.get(table_name)
            if catalog_table is not None:
                referenced_columns |= catalog_table.column_names
        for column in tree.find_all(exp.Column):
            if isinstance(column.this, exp.Star):
                continue
            name = column.name.lower()
            qualifier = column.table.lower() if column.table else ""
            if qualifier:
                if qualifier in virtual_sources:
                    continue
                resolved = alias_to_table.get(qualifier)
                if resolved is None:
                    result.issues.append(f"The column {column.sql()} has an unknown qualifier.")
                    return
                catalog_table = self._catalog.get(resolved)
                if catalog_table is None or name not in catalog_table.column_names:
                    result.issues.append(f"The column {column.sql()} is not available.")
                    return
            elif name not in referenced_columns and name not in projection_aliases:
                if virtual_sources:
                    continue
                result.issues.append(f"The column {column.name} is not available.")
                return

    def _check_star(self, tree: exp.Expr, result: ValidationResult) -> None:
        if self._config.allow_select_star:
            return
        for select in tree.find_all(exp.Select):
            for projection in select.expressions:
                if isinstance(projection, exp.Star) or (
                    isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
                ):
                    result.issues.append("SELECT * is not permitted; name the columns required.")
                    return

    def _check_complexity(self, tree: exp.Expr, result: ValidationResult) -> None:
        joins = len(list(tree.find_all(exp.Join)))
        if joins > self._config.max_joins:
            result.issues.append(
                f"The query uses {joins} joins; the limit is {self._config.max_joins}."
            )
        depth = _subquery_depth(tree)
        if depth > self._config.max_subquery_depth:
            result.issues.append(
                f"The query nests subqueries {depth} deep; the limit is "
                f"{self._config.max_subquery_depth}."
            )

    @staticmethod
    def _apply_row_cap(tree: exp.Expr, cap: int) -> exp.Expr:
        if isinstance(tree, exp.Select):
            existing = _limit_value(tree)
            if existing is None or existing > cap:
                return tree.limit(cap)
            return tree
        if isinstance(tree, exp.Query):
            return exp.select("*").from_(tree.subquery("capped")).limit(cap)
        return tree


def _limit_value(select: exp.Select) -> int | None:
    limit = select.args.get("limit")
    node = limit.expression if isinstance(limit, exp.Limit) else None
    if node is None:
        fetch = select.args.get("fetch")
        node = fetch.args.get("count") if isinstance(fetch, exp.Fetch) else None
    if isinstance(node, exp.Literal) and node.is_int:
        return int(node.this)
    return None


def _subquery_depth(node: exp.Expr) -> int:
    deepest = 0
    for sub in node.find_all(exp.Subquery):
        depth = 0
        parent = sub.parent
        while parent is not None:
            if isinstance(parent, exp.Subquery):
                depth += 1
            parent = parent.parent
        deepest = max(deepest, depth + 1)
    return deepest
