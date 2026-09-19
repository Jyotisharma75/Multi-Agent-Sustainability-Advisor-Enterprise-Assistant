"""SQL safety: the guard must refuse every write, escape or exfiltration attempt,
and the validator must confine queries to the exposed catalogue."""

from __future__ import annotations

import pytest

from sustainability_advisor.config.models import Settings
from sustainability_advisor.db.tables import build_tables
from sustainability_advisor.sql.catalog import SchemaCatalog
from sustainability_advisor.sql.guard import SQLGuard
from sustainability_advisor.sql.validator import SQLValidator, ValidationResult

pytestmark = pytest.mark.security

ATTACKS = [
    "DROP TABLE emissions",
    "DELETE FROM emissions",
    "UPDATE emissions SET co2e_tonnes = 0",
    "INSERT INTO facilities (facility_id, name) VALUES ('x', 'y')",
    "SELECT facility_id FROM facilities; DROP TABLE facilities",
    "SELECT facility_id INTO backup_table FROM facilities",
    "EXEC xp_cmdshell 'dir'",
    "SELECT * FROM OPENROWSET('SQLNCLI', 'server=x', 'select 1')",
    "WITH x AS (SELECT facility_id FROM facilities) DELETE FROM facilities",
    "SELECT name FROM sys.tables",
    "SELECT table_name FROM information_schema.tables",
    "SELECT facility_id FROM otherdb.dbo.facilities",
    "SELECT facility_id FROM #temp",
    "SELECT @@VERSION",
    "DECLARE @x INT",
    "SELECT SUSER_SNAME()",
    "TRUNCATE TABLE emissions",
    "ALTER TABLE emissions ADD c INT",
    "CREATE TABLE t (a INT)",
    "GRANT SELECT ON emissions TO public",
    "SELECT name FROM sqlite_master",
    "",
    "   ;  ",
]


@pytest.fixture
def guard(settings: Settings) -> SQLGuard:
    return SQLGuard(settings.sql)


@pytest.fixture
def validator(settings: Settings) -> SQLValidator:
    catalog = SchemaCatalog.build(settings.sql, build_tables(None), None)
    return SQLValidator(settings.sql, catalog)


@pytest.mark.parametrize("sql", ATTACKS)
def test_guard_rejects_attacks(guard: SQLGuard, sql: str) -> None:
    result = guard.check(sql, dialect="tsql")
    assert not result.ok, sql
    assert result.issues


def test_guard_rejects_overlong_statement(guard: SQLGuard, settings: Settings) -> None:
    sql = (
        "SELECT facility_id FROM facilities WHERE name = '" + "a" * settings.sql.max_sql_chars + "'"
    )
    assert not guard.check(sql, dialect="tsql").ok


def test_guard_accepts_safe_select_and_literal_keywords(guard: SQLGuard) -> None:
    sql = (
        "SELECT TOP 5 f.name, SUM(e.co2e_tonnes) AS total FROM emissions e "
        "JOIN facilities f ON f.facility_id = e.facility_id "
        "WHERE f.name <> 'DROP TABLE' GROUP BY f.name ORDER BY total DESC;"
    )
    result = guard.check(sql, dialect="tsql")
    assert result.ok, result.issues
    assert result.statement_kind == "SELECT"


def _validate(
    guard: SQLGuard, validator: SQLValidator, sql: str, rows: int = 50
) -> ValidationResult:
    guarded = guard.check(sql, dialect="tsql")
    assert guarded.ok and guarded.expression is not None, guarded.issues
    return validator.validate(guarded.expression, execution_dialect="sqlite", max_rows=rows)


def test_validator_rejects_unexposed_table(guard: SQLGuard, validator: SQLValidator) -> None:
    result = _validate(guard, validator, "SELECT event_id FROM audit_events")
    assert not result.ok


def test_validator_rejects_unknown_column(guard: SQLGuard, validator: SQLValidator) -> None:
    result = _validate(guard, validator, "SELECT e.secret_column FROM emissions e")
    assert not result.ok


def test_validator_rejects_select_star(guard: SQLGuard, validator: SQLValidator) -> None:
    result = _validate(guard, validator, "SELECT * FROM emissions")
    assert not result.ok


def test_validator_allows_count_star(guard: SQLGuard, validator: SQLValidator) -> None:
    result = _validate(guard, validator, "SELECT COUNT(*) AS n FROM emissions")
    assert result.ok, result.issues


def test_validator_enforces_join_limit(
    guard: SQLGuard, validator: SQLValidator, settings: Settings
) -> None:
    joins = " ".join(
        f"JOIN facilities f{i} ON f{i}.facility_id = e.facility_id"
        for i in range(settings.sql.max_joins + 1)
    )
    result = _validate(guard, validator, f"SELECT e.scope FROM emissions e {joins}")
    assert not result.ok


def test_validator_caps_rows_and_transpiles(guard: SQLGuard, validator: SQLValidator) -> None:
    result = _validate(guard, validator, "SELECT TOP 1000 facility_id FROM dbo.facilities", rows=10)
    assert result.ok
    execution = result.execution_sql
    assert "LIMIT 11" in execution
    assert "dbo." not in execution
    assert "TOP" in result.display_sql


def test_validator_keeps_smaller_limit(guard: SQLGuard, validator: SQLValidator) -> None:
    result = _validate(guard, validator, "SELECT TOP 3 facility_id FROM facilities", rows=10)
    assert "LIMIT 3" in result.execution_sql


def test_validator_rejects_foreign_schema(guard: SQLGuard, validator: SQLValidator) -> None:
    result = _validate(guard, validator, "SELECT facility_id FROM staging.facilities")
    assert not result.ok


def test_validator_accepts_cte_and_aliases(guard: SQLGuard, validator: SQLValidator) -> None:
    sql = (
        "WITH m AS (SELECT facility_id, SUM(co2e_tonnes) AS total FROM emissions "
        "GROUP BY facility_id) SELECT m.facility_id, m.total FROM m ORDER BY m.total DESC"
    )
    result = _validate(guard, validator, sql)
    assert result.ok, result.issues
