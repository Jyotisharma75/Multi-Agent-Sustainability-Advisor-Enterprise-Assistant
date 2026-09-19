"""Engine construction for Azure SQL and SQLite.

Two engines are built from one configuration:

* the *application* engine used by repositories, which may write audit,
  conversation, forecast and recommendation records
* the *query* engine used only for model generated SQL. On Azure SQL it
  connects with ``ApplicationIntent=ReadOnly`` (routed to a readable
  secondary when one exists) and the deployment should grant its identity
  ``db_datareader`` only. On SQLite it opens the file in read only mode.

Azure SQL authenticates with a Microsoft Entra access token by default,
passed through the ODBC pre connect attribute, so no password is stored.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import Engine, create_engine, event, text

from sustainability_advisor.config.loader import read_secret
from sustainability_advisor.config.models import AzureSqlConfig, DatabaseConfig
from sustainability_advisor.db.tables import Tables, build_tables
from sustainability_advisor.domain.errors import DependencyUnavailableError

#: ODBC connection attribute that carries an Entra access token.
SQL_COPT_SS_ACCESS_TOKEN = 1256


@dataclass(frozen=True, slots=True)
class Database:
    app_engine: Engine
    query_engine: Engine
    tables: Tables
    dialect: str
    config: DatabaseConfig

    def ping(self) -> None:
        with self.app_engine.connect() as conn:
            conn.execute(text(self.config.readiness_query))

    def dispose(self) -> None:
        self.app_engine.dispose()
        self.query_engine.dispose()


def pack_access_token(token: str) -> bytes:
    """Encode an access token the way the SQL Server ODBC driver expects."""
    encoded = token.encode("utf-16-le")
    return struct.pack("<i", len(encoded)) + encoded


def _odbc_connection_string(cfg: AzureSqlConfig, *, read_only: bool) -> str:
    server = read_secret(cfg.server_env)
    database = read_secret(cfg.database_env)
    if not server or not database:
        raise DependencyUnavailableError(
            f"Azure SQL requires {cfg.server_env} and {cfg.database_env} to be set."
        )
    parts = [
        f"Driver={{{cfg.driver}}}",
        f"Server=tcp:{server},1433",
        f"Database={database}",
        f"Encrypt={'yes' if cfg.encrypt else 'no'}",
        f"TrustServerCertificate={'yes' if cfg.trust_server_certificate else 'no'}",
        f"Connection Timeout={cfg.connect_timeout_seconds}",
    ]
    if read_only:
        parts.append("ApplicationIntent=ReadOnly")
    if cfg.auth_mode == "sql_password":
        user = read_secret(cfg.username_env)
        password = read_secret(cfg.password_env)
        if not user or not password:
            raise DependencyUnavailableError(
                f"sql_password auth requires {cfg.username_env} and {cfg.password_env}."
            )
        parts.extend([f"Uid={user}", f"Pwd={password}"])
    return ";".join(parts) + ";"


def _azure_engine(config: DatabaseConfig, *, read_only: bool) -> Engine:
    cfg = config.azure_sql
    odbc = _odbc_connection_string(cfg, read_only=read_only)
    engine = create_engine(
        f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}",
        pool_size=cfg.pool_size,
        max_overflow=cfg.max_overflow,
        pool_recycle=cfg.pool_recycle_seconds,
        pool_pre_ping=True,
        fast_executemany=True,
    )
    timeout = int(config.query_timeout_seconds)

    if cfg.auth_mode == "entra":
        from azure.identity import DefaultAzureCredential, ManagedIdentityCredential

        query_client_id = read_secret(cfg.query_identity_client_id_env) if read_only else None
        # The read only engine may run as a separate identity that holds only
        # db_datareader style grants on the exposed tables (migrations/002).
        credential: Any = (
            ManagedIdentityCredential(client_id=query_client_id)
            if query_client_id
            else DefaultAzureCredential()
        )

        @event.listens_for(engine, "do_connect")
        def _inject_token(dialect: Any, conn_rec: Any, cargs: Any, cparams: dict[str, Any]) -> None:
            token = credential.get_token(cfg.token_scope).token
            cparams["attrs_before"] = {SQL_COPT_SS_ACCESS_TOKEN: pack_access_token(token)}

    @event.listens_for(engine, "connect")
    def _set_timeout(dbapi_connection: Any, connection_record: Any) -> None:
        dbapi_connection.timeout = timeout

    return engine


def _sqlite_engine(config: DatabaseConfig, *, read_only: bool) -> Engine:
    path = Path(config.sqlite.path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if read_only:
        url = f"sqlite+pysqlite:///file:{path.as_posix()}?mode=ro&uri=true"
    else:
        url = f"sqlite+pysqlite:///{path.as_posix()}"
    engine = create_engine(url, connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_connection: Any, connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        if read_only:
            cursor.execute("PRAGMA query_only=ON")
        cursor.close()

    return engine


def build_database(config: DatabaseConfig) -> Database:
    """Create engines and table definitions for the configured backend."""
    if config.backend == "azure_sql":
        tables = build_tables(config.schema_name)
        app_engine = _azure_engine(config, read_only=False)
        query_engine = _azure_engine(config, read_only=True)
        dialect = "tsql"
    else:
        tables = build_tables(None)
        app_engine = _sqlite_engine(config, read_only=False)
        if config.sqlite.create_schema:
            tables.metadata.create_all(app_engine)
        query_engine = _sqlite_engine(config, read_only=True)
        dialect = "sqlite"
    return Database(
        app_engine=app_engine,
        query_engine=query_engine,
        tables=tables,
        dialect=dialect,
        config=config,
    )
