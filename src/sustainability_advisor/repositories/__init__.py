"""Repositories: the only path from services to Azure SQL."""

from __future__ import annotations

from dataclasses import dataclass

from sustainability_advisor.db.engine import Database
from sustainability_advisor.repositories.anomalies import AnomalyRepository
from sustainability_advisor.repositories.audit_events import AuditEventRepository
from sustainability_advisor.repositories.conversations import ConversationRepository
from sustainability_advisor.repositories.documents import DocumentRepository
from sustainability_advisor.repositories.emissions import EmissionRepository
from sustainability_advisor.repositories.energy import EnergyRepository
from sustainability_advisor.repositories.facilities import FacilityRepository
from sustainability_advisor.repositories.forecast_results import ForecastResultRepository
from sustainability_advisor.repositories.production import ProductionRepository
from sustainability_advisor.repositories.recommendations import RecommendationRepository
from sustainability_advisor.repositories.sensors import SensorRepository
from sustainability_advisor.repositories.sustainability_metrics import (
    SustainabilityMetricRepository,
)


@dataclass(frozen=True, slots=True)
class Repositories:
    facilities: FacilityRepository
    emissions: EmissionRepository
    energy: EnergyRepository
    production: ProductionRepository
    sensors: SensorRepository
    metrics: SustainabilityMetricRepository
    anomalies: AnomalyRepository
    recommendations: RecommendationRepository
    forecasts: ForecastResultRepository
    documents: DocumentRepository
    audit: AuditEventRepository
    conversations: ConversationRepository


def build_repositories(db: Database) -> Repositories:
    engine, tables = db.app_engine, db.tables
    return Repositories(
        facilities=FacilityRepository(engine, tables),
        emissions=EmissionRepository(engine, tables),
        energy=EnergyRepository(engine, tables),
        production=ProductionRepository(engine, tables),
        sensors=SensorRepository(engine, tables),
        metrics=SustainabilityMetricRepository(engine, tables),
        anomalies=AnomalyRepository(engine, tables),
        recommendations=RecommendationRepository(engine, tables),
        forecasts=ForecastResultRepository(engine, tables),
        documents=DocumentRepository(engine, tables),
        audit=AuditEventRepository(engine, tables),
        conversations=ConversationRepository(engine, tables),
    )
