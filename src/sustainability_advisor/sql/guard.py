"""Statement level SQL safety.

Works on the parsed syntax tree rather than text, because text matching is
defeated by comments, casing and whitespace, while a parser sees what the
database will see. Enforced:

* the statement parses and there is exactly one of them
* the statement kind is on the configured allowlist (SELECT by default)
* no nested write statement hides inside a CTE or subquery
* no ``SELECT ... INTO``
* no blocked function or routine prefix (OPENROWSET, xp_, sp_, ...)
* no variables, session parameters, placeholders, temp tables, table valued
  functions, system catalogue schemas or cross database names
* a raw token scan as defence in depth for constructs the parser absorbs

Every list consulted comes from configuration.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import sqlglot
from sqlglot import TokenType, exp
from sqlglot.errors import SqlglotError

from sustainability_advisor.config.models import SqlSafetyConfig

_LITERAL_TOKENS = frozenset(
    {
        TokenType.STRING,
        TokenType.NATIONAL_STRING,
        TokenType.RAW_STRING,
        TokenType.HEREDOC_STRING,
        TokenType.UNICODE_STRING,
        TokenType.BYTE_STRING,
        TokenType.HEX_STRING,
        TokenType.NUMBER,
        TokenType.IDENTIFIER,
    }
)

STATEMENT_NODES: dict[str, tuple[type[exp.Expr], ...]] = {
    "SELECT": (exp.Select, exp.SetOperation, exp.Subquery),
    "INSERT": (exp.Insert,),
    "UPDATE": (exp.Update,),
    "DELETE": (exp.Delete,),
    "MERGE": (exp.Merge,),
    "CREATE": (exp.Create,),
    "DROP": (exp.Drop,),
    "ALTER": (exp.Alter,),
    "TRUNCATE": (exp.TruncateTable,),
    "EXEC": (exp.Execute,),
    "DECLARE": (exp.Declare,),
}

_WRITE_NODES: tuple[tuple[str, type[exp.Expr]], ...] = tuple(
    (kind, node) for kind, nodes in STATEMENT_NODES.items() if kind != "SELECT" for node in nodes
)


@dataclass(frozen=True, slots=True)
class GuardIssue:
    code: str
    message: str


@dataclass(slots=True)
class GuardResult:
    sql: str
    statement_kind: str | None = None
    expression: exp.Expr | None = None
    issues: list[GuardIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues and self.expression is not None


class SQLGuard:
    def __init__(self, config: SqlSafetyConfig) -> None:
        self._config = config
        self._allowed = frozenset(s.upper() for s in config.allowed_statements)
        self._blocked_functions = frozenset(f.upper() for f in config.blocked_functions)
        self._blocked_prefixes = tuple(p.upper() for p in config.blocked_identifier_prefixes)
        self._system_schemas = frozenset(s.lower() for s in config.system_schemas)
        permitted = set(self._allowed)
        if "EXEC" in permitted:
            permitted.add("EXECUTE")
        self._blocked_keywords = frozenset(
            k.upper() for k in config.blocked_keywords if k.upper() not in permitted
        )

    def check(self, sql: str, *, dialect: str) -> GuardResult:
        text = (sql or "").strip()
        while text.endswith(";"):
            text = text[:-1].rstrip()
        result = GuardResult(sql=text)
        if not text:
            result.issues.append(GuardIssue("empty_statement", "No SQL statement was produced."))
            return result
        if len(text) > self._config.max_sql_chars:
            result.issues.append(
                GuardIssue("statement_too_long", "The statement exceeds the configured length.")
            )
            return result
        try:
            parsed = [s for s in sqlglot.parse(text, read=dialect) if s is not None]
        except SqlglotError as exc:
            result.issues.append(GuardIssue("syntax_error", f"The statement did not parse: {exc}"))
            return result
        if len(parsed) != 1 or isinstance(parsed[0], exp.Block):
            result.issues.append(
                GuardIssue("multiple_statements", "Only a single statement may be executed.")
            )
            return result
        root = parsed[0]
        kind = self.statement_kind(root)
        result.statement_kind = kind
        if kind not in self._allowed:
            result.issues.append(
                GuardIssue("statement_not_allowed", f"{kind} statements are not permitted.")
            )
            return result
        for check in (
            self._check_nested_statements,
            self._check_select_into,
            self._check_functions,
            self._check_tables,
            self._check_variables,
        ):
            issue = check(root)
            if issue is not None:
                result.issues.append(issue)
                return result
        token_issue = self._check_tokens(text, dialect)
        if token_issue is not None:
            result.issues.append(token_issue)
            return result
        result.expression = root
        return result

    @staticmethod
    def statement_kind(node: exp.Expr) -> str:
        for kind, node_types in STATEMENT_NODES.items():
            if isinstance(node, node_types):
                return kind
        if isinstance(node, exp.Command):
            command = str(node.this or "COMMAND").upper().split()
            return command[0] if command else "COMMAND"
        return type(node).__name__.upper()

    def _check_nested_statements(self, root: exp.Expr) -> GuardIssue | None:
        for kind, node_type in _WRITE_NODES:
            if kind in self._allowed:
                continue
            for node in root.find_all(node_type):
                if node is not root:
                    return GuardIssue("nested_statement", f"A nested {kind} is not permitted.")
        for node in root.find_all(exp.Command):
            name = (str(node.this or "").upper().split() or ["COMMAND"])[0]
            if name not in self._allowed:
                return GuardIssue("command_not_allowed", f"The command {name} is not permitted.")
        return None

    @staticmethod
    def _check_select_into(root: exp.Expr) -> GuardIssue | None:
        for select in root.find_all(exp.Select):
            if select.args.get("into") is not None:
                return GuardIssue("select_into", "SELECT INTO writes a table and is not permitted.")
        return None

    def _check_functions(self, root: exp.Expr) -> GuardIssue | None:
        for node in root.find_all(exp.Func):
            names: set[str] = {type(node).__name__.upper()}
            if isinstance(node, exp.Anonymous) and node.this:
                names.add(str(node.this).upper())
            with contextlib.suppress(Exception):
                names.add(str(node.sql_name()).upper())
            if names & self._blocked_functions or any(
                n.startswith(self._blocked_prefixes) for n in names
            ):
                return GuardIssue("function_not_allowed", "A referenced function is not permitted.")
        return None

    def _check_tables(self, root: exp.Expr) -> GuardIssue | None:
        for table in root.find_all(exp.Table):
            if isinstance(table.this, exp.Func | exp.Anonymous):
                return GuardIssue("table_function", "Table valued functions are not permitted.")
            if isinstance(table.this, exp.Dot) or (
                table.args.get("catalog") and not self._config.allow_cross_database
            ):
                return GuardIssue("cross_database", "Cross database references are not permitted.")
            name = table.name or ""
            is_temp = isinstance(table.this, exp.Identifier) and bool(
                table.this.args.get("temporary")
            )
            if not self._config.allow_temp_tables and (is_temp or name.startswith(("#", "@"))):
                return GuardIssue("temp_table", "Temporary tables are not permitted.")
            if table.db and table.db.lower() in self._system_schemas:
                return GuardIssue("system_schema", "System catalogue objects are not readable.")
            if name.lower() in self._system_schemas:
                return GuardIssue("system_schema", "System catalogue objects are not readable.")
        return None

    def _check_variables(self, root: exp.Expr) -> GuardIssue | None:
        if not self._config.allow_variables:
            for node_type in (exp.Parameter, exp.SessionParameter):
                if next(root.find_all(node_type), None) is not None:
                    return GuardIssue("variables", "Variables and session parameters are refused.")
        if next(root.find_all(exp.Placeholder), None) is not None:
            return GuardIssue("placeholder", "Parameter placeholders are not permitted.")
        return None

    def _check_tokens(self, text: str, dialect: str) -> GuardIssue | None:
        try:
            tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(text)
        except SqlglotError:
            return None
        for token in tokens:
            if token.token_type in _LITERAL_TOKENS:
                continue
            word = token.text.upper()
            if word in self._blocked_keywords:
                return GuardIssue("blocked_keyword", f"The keyword {word} is not permitted.")
            if word.startswith(self._blocked_prefixes):
                return GuardIssue("blocked_identifier", "A referenced routine is not permitted.")
        return None
