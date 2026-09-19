"""Relational schema.

The same SQLAlchemy Core definitions serve Azure SQL in production and SQLite
in development and tests. Column comments document meaning for the SQL
generation prompt. ``migrations/001_schema.sql`` is the authoritative Azure
SQL DDL and mirrors these definitions.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    UnicodeText,
)


@dataclass(frozen=True, slots=True)
class Tables:
    metadata: MetaData
    facilities: Table
    emissions: Table
    energy: Table
    production: Table
    sensors: Table
    sensor_readings: Table
    sustainability_metrics: Table
    anomalies: Table
    recommendations: Table
    forecast_results: Table
    documents: Table
    audit_events: Table
    conversations: Table
    conversation_messages: Table


def build_tables(schema: str | None) -> Tables:
    """Build the table definitions for an optional schema name."""
    md = MetaData(schema=schema)
    fk_prefix = f"{schema}." if schema else ""

    def fk(target: str) -> ForeignKey:
        return ForeignKey(f"{fk_prefix}{target}")

    facilities = Table(
        "facilities",
        md,
        Column("facility_id", String(64), primary_key=True, comment="Facility identifier"),
        Column("name", String(200), nullable=False, comment="Facility display name"),
        Column("region", String(100), comment="Operating region"),
        Column("country", String(100), comment="Country"),
        Column("industry", String(100), comment="Industry segment"),
        Column("capacity", Float, comment="Nameplate capacity"),
        Column("capacity_unit", String(50), comment="Unit of capacity"),
        Column("is_active", Boolean, nullable=False, default=True, comment="Active flag"),
    )
    emissions = Table(
        "emissions",
        md,
        Column("emission_id", Integer, primary_key=True, autoincrement=True),
        Column("facility_id", String(64), fk("facilities.facility_id"), nullable=False),
        Column("period_start", Date, nullable=False, comment="First day of reporting period"),
        Column("period_end", Date, nullable=False, comment="Last day of reporting period"),
        Column("scope", Integer, nullable=False, comment="GHG Protocol scope 1, 2 or 3"),
        Column("category", String(100), comment="Emission category"),
        Column("co2e_tonnes", Float, nullable=False, comment="Emissions in tonnes CO2e"),
        Column("source", String(100), comment="Data source system"),
        Column("methodology", String(100), comment="Calculation methodology"),
        Index("ix_emissions_facility_period", "facility_id", "period_start"),
    )
    energy = Table(
        "energy",
        md,
        Column("energy_id", Integer, primary_key=True, autoincrement=True),
        Column("facility_id", String(64), fk("facilities.facility_id"), nullable=False),
        Column("period_start", Date, nullable=False),
        Column("period_end", Date, nullable=False),
        Column("energy_type", String(50), nullable=False, comment="electricity, natural_gas, ..."),
        Column("consumption_mwh", Float, nullable=False, comment="Energy consumed in MWh"),
        Column("renewable_mwh", Float, nullable=False, comment="Renewable portion in MWh"),
        Column("cost", Float, comment="Invoiced cost, null when not reported"),
        Column("currency", String(3), comment="ISO currency of cost"),
        Index("ix_energy_facility_period", "facility_id", "period_start"),
    )
    production = Table(
        "production",
        md,
        Column("production_id", Integer, primary_key=True, autoincrement=True),
        Column("facility_id", String(64), fk("facilities.facility_id"), nullable=False),
        Column("period_start", Date, nullable=False),
        Column("period_end", Date, nullable=False),
        Column("product", String(100), comment="Product line"),
        Column("output_quantity", Float, nullable=False, comment="Units produced"),
        Column("output_unit", String(50), comment="Unit of output"),
        Index("ix_production_facility_period", "facility_id", "period_start"),
    )
    sensors = Table(
        "sensors",
        md,
        Column("sensor_id", String(64), primary_key=True),
        Column("facility_id", String(64), fk("facilities.facility_id"), nullable=False),
        Column("sensor_type", String(100), nullable=False, comment="Measured quantity"),
        Column("unit", String(50)),
        Column("location", String(200)),
    )
    sensor_readings = Table(
        "sensor_readings",
        md,
        Column("reading_id", Integer, primary_key=True, autoincrement=True),
        Column("sensor_id", String(64), fk("sensors.sensor_id"), nullable=False),
        Column("recorded_at", DateTime, nullable=False),
        Column("value", Float, nullable=False),
        Index("ix_sensor_readings_sensor_time", "sensor_id", "recorded_at"),
    )
    sustainability_metrics = Table(
        "sustainability_metrics",
        md,
        Column("metric_id", Integer, primary_key=True, autoincrement=True),
        Column("facility_id", String(64), fk("facilities.facility_id"), nullable=False),
        Column("metric_name", String(100), nullable=False, comment="KPI or reference metric"),
        Column("period_start", Date, nullable=False),
        Column("period_end", Date, nullable=False),
        Column("value", Float, comment="Measured value, null when not measured"),
        Column("unit", String(50)),
        Column("target_value", Float, comment="Approved target, null when none"),
        Column("source", String(100)),
        Index("ix_metrics_facility_name", "facility_id", "metric_name", "period_start"),
    )
    anomalies = Table(
        "anomalies",
        md,
        Column("anomaly_id", String(64), primary_key=True),
        Column("facility_id", String(64), fk("facilities.facility_id"), nullable=False),
        Column("metric", String(100), nullable=False),
        Column("period_start", Date, nullable=False),
        Column("observed_value", Float, nullable=False),
        Column("expected_value", Float, nullable=False),
        Column("score", Float, nullable=False),
        Column("severity", String(20), nullable=False),
        Column("method", String(50), nullable=False),
        Column("detected_at", DateTime, nullable=False),
        Column("status", String(20), nullable=False),
    )
    recommendations = Table(
        "recommendations",
        md,
        Column("recommendation_id", String(64), primary_key=True),
        Column("facility_id", String(64)),
        Column("conversation_id", String(64)),
        Column("issue", UnicodeText, nullable=False),
        Column("severity", String(20), nullable=False),
        Column("priority", String(20), nullable=False),
        Column("confidence", Float, nullable=False),
        Column("payload", UnicodeText, nullable=False),
        Column("created_at", DateTime, nullable=False),
    )
    forecast_results = Table(
        "forecast_results",
        md,
        Column("forecast_id", String(64), primary_key=True),
        Column("facility_id", String(64), nullable=False),
        Column("metric", String(100), nullable=False),
        Column("method", String(50), nullable=False),
        Column("horizon", Integer, nullable=False),
        Column("backtest_mape", Float),
        Column("payload", UnicodeText, nullable=False),
        Column("generated_at", DateTime, nullable=False),
    )
    documents = Table(
        "documents",
        md,
        Column("document_id", String(128), primary_key=True),
        Column("title", String(300), nullable=False),
        Column("doc_type", String(100), nullable=False),
        Column("source_uri", String(1000), nullable=False),
        Column("jurisdiction", String(100)),
        Column("facility_id", String(64)),
        Column("effective_date", Date),
        Column("version", String(50)),
        Column("checksum", String(128), nullable=False),
        Column("indexed_at", DateTime),
    )
    audit_events = Table(
        "audit_events",
        md,
        Column("event_id", String(64), primary_key=True),
        Column("conversation_id", String(64)),
        Column("request_id", String(64)),
        Column("event_type", String(100), nullable=False),
        Column("actor", String(100), nullable=False),
        Column("payload", UnicodeText, nullable=False),
        Column("created_at", DateTime, nullable=False),
        Index("ix_audit_conversation", "conversation_id", "created_at"),
    )
    conversations = Table(
        "conversations",
        md,
        Column("conversation_id", String(64), primary_key=True),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
    )
    conversation_messages = Table(
        "conversation_messages",
        md,
        Column("message_id", String(64), primary_key=True),
        Column(
            "conversation_id",
            String(64),
            fk("conversations.conversation_id"),
            nullable=False,
        ),
        Column("role", String(20), nullable=False),
        Column("content", UnicodeText, nullable=False),
        Column("payload", UnicodeText, nullable=False),
        Column("created_at", DateTime, nullable=False),
        Index("ix_messages_conversation", "conversation_id", "created_at"),
    )
    return Tables(
        metadata=md,
        facilities=facilities,
        emissions=emissions,
        energy=energy,
        production=production,
        sensors=sensors,
        sensor_readings=sensor_readings,
        sustainability_metrics=sustainability_metrics,
        anomalies=anomalies,
        recommendations=recommendations,
        forecast_results=forecast_results,
        documents=documents,
        audit_events=audit_events,
        conversations=conversations,
        conversation_messages=conversation_messages,
    )
