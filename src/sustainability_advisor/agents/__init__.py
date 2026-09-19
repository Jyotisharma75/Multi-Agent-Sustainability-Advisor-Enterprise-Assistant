"""Specialised agents and their registry."""

from __future__ import annotations

from sustainability_advisor.agents.anomaly_detection import AnomalyDetectionAgent
from sustainability_advisor.agents.base import AgentDependencies, BaseAgent
from sustainability_advisor.agents.compliance_knowledge import ComplianceKnowledgeAgent
from sustainability_advisor.agents.data_analyst import DataAnalystAgent
from sustainability_advisor.agents.document_intelligence import DocumentIntelligenceAgent
from sustainability_advisor.agents.forecasting import ForecastingAgent
from sustainability_advisor.agents.orchestrator import OrchestratorAgent
from sustainability_advisor.agents.recommendation import RecommendationAgent
from sustainability_advisor.agents.response_synthesizer import ResponseSynthesizer
from sustainability_advisor.agents.sql_agent import SQLAgent
from sustainability_advisor.agents.sustainability_analyst import SustainabilityAnalystAgent
from sustainability_advisor.config.loader import ConfigurationError

AGENT_CLASSES: dict[str, type[BaseAgent]] = {
    cls.name: cls
    for cls in (
        OrchestratorAgent,
        SustainabilityAnalystAgent,
        DataAnalystAgent,
        ForecastingAgent,
        AnomalyDetectionAgent,
        RecommendationAgent,
        DocumentIntelligenceAgent,
        SQLAgent,
        ComplianceKnowledgeAgent,
        ResponseSynthesizer,
    )
}


def build_agents(deps: AgentDependencies) -> dict[str, BaseAgent]:
    missing = set(AGENT_CLASSES) - set(deps.settings.agents)
    if missing:
        raise ConfigurationError(f"Agents without configuration: {sorted(missing)}")
    unknown = set(deps.settings.agents) - set(AGENT_CLASSES)
    if unknown:
        raise ConfigurationError(f"Configured agents without implementation: {sorted(unknown)}")
    return {name: cls(deps.settings.agents[name], deps) for name, cls in AGENT_CLASSES.items()}
