"""Composition root.

Builds every component from configuration exactly once. Tests pass their own
provider registry, embedding provider or index to replace external services
without touching anything else.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sustainability_advisor.agents import build_agents
from sustainability_advisor.agents.base import AgentDependencies, BaseAgent
from sustainability_advisor.config.loader import get_settings
from sustainability_advisor.config.models import Settings
from sustainability_advisor.db.engine import Database, build_database
from sustainability_advisor.graph.builder import GraphNodes, build_graph
from sustainability_advisor.guardrails.input_validator import InputValidator
from sustainability_advisor.guardrails.output_validator import OutputValidator
from sustainability_advisor.guardrails.prompt_injection import InjectionDetector
from sustainability_advisor.guardrails.tool_permissions import ToolPermissionPolicy
from sustainability_advisor.llm.factory import ProviderChain, ProviderRegistry, build_registry
from sustainability_advisor.llm.huggingface_local import HuggingFaceLocalProvider
from sustainability_advisor.observability.logging import get_logger
from sustainability_advisor.prompts.loader import PromptLibrary
from sustainability_advisor.repositories import Repositories, build_repositories
from sustainability_advisor.retrieval.azure_search import AzureAISearchIndex
from sustainability_advisor.retrieval.embeddings import EmbeddingProvider, build_embedding_provider
from sustainability_advisor.retrieval.index import SearchIndex
from sustainability_advisor.retrieval.ingest import IngestReport, ingest_folder
from sustainability_advisor.retrieval.memory_index import InMemoryHybridIndex
from sustainability_advisor.retrieval.retriever import DocumentRetriever
from sustainability_advisor.services.anomaly_service import AnomalyService
from sustainability_advisor.services.assistant_service import AssistantService
from sustainability_advisor.services.audit_service import AuditService, ConversationService
from sustainability_advisor.services.forecast_service import ForecastService
from sustainability_advisor.services.kpi_service import KpiService
from sustainability_advisor.services.recommendation_service import RecommendationService
from sustainability_advisor.services.scenario_service import ScenarioService
from sustainability_advisor.sql.catalog import SchemaCatalog
from sustainability_advisor.sql.executor import SQLExecutor
from sustainability_advisor.sql.generator import COMMON_PROMPT, SQLGenerator
from sustainability_advisor.sql.guard import SQLGuard
from sustainability_advisor.sql.router import ModelRouter
from sustainability_advisor.sql.service import NLToSQLService
from sustainability_advisor.sql.validator import SQLValidator
from sustainability_advisor.tools.analysis_tools import (
    AnomalyAnalysisTool,
    EmissionForecastTool,
    RecommendationRankingTool,
    ScenarioAnalysisTool,
)
from sustainability_advisor.tools.base import ToolRegistry
from sustainability_advisor.tools.data_tools import (
    FacilityDirectoryTool,
    KpiLookupTool,
    SQLQueryTool,
)
from sustainability_advisor.tools.retrieval_tools import DocumentSearchTool, VectorSearchTool

logger = get_logger(__name__)

REQUIRED_PROMPTS = [
    "common",
    "intent_classification",
    "sql_generation",
    "sql_verification",
    "compliance_extraction",
    "response_synthesis",
]


@dataclass
class Container:
    settings: Settings
    database: Database
    repositories: Repositories
    providers: ProviderRegistry
    prompts: PromptLibrary
    embeddings: EmbeddingProvider
    search_index: SearchIndex
    retriever: DocumentRetriever
    sql_service: NLToSQLService
    tools: ToolRegistry
    agents: dict[str, BaseAgent]
    graph: Any
    assistant: AssistantService
    injection: InjectionDetector

    async def startup(self) -> IngestReport | None:
        self.prompts.validate(REQUIRED_PROMPTS)
        local = self.providers.all().get("local")
        if (
            isinstance(local, HuggingFaceLocalProvider)
            and self.settings.llm.local.preload
            and local.is_available()
        ):
            await local.preload()
        retrieval = self.settings.retrieval
        if retrieval.auto_ingest_on_start:
            folder = Path(retrieval.documents_path)
            if await asyncio.to_thread(folder.is_dir):
                report = await ingest_folder(
                    folder,
                    chunking=retrieval.chunking,
                    embeddings=self.embeddings,
                    index=self.search_index,
                    documents=self.repositories.documents,
                    in_memory_index=retrieval.backend == "memory",
                )
                logger.info("startup.ingested", **report.model_dump())
                return report
            logger.warning("startup.documents_missing", path=str(folder))
        return None

    def close(self) -> None:
        self.database.dispose()


def build_container(
    settings: Settings | None = None,
    *,
    providers: ProviderRegistry | None = None,
    embeddings: EmbeddingProvider | None = None,
    search_index: SearchIndex | None = None,
) -> Container:
    cfg = settings or get_settings()
    database = build_database(cfg.database)
    repos = build_repositories(database)
    registry = providers or build_registry(cfg.llm)
    prompts = PromptLibrary(cfg.prompts)
    injection = InjectionDetector(cfg.guardrails)

    embedding_provider = embeddings or build_embedding_provider(
        cfg.retrieval.embeddings, cfg.llm.azure_openai
    )
    index: SearchIndex
    if search_index is not None:
        index = search_index
    elif cfg.retrieval.backend == "azure_search":
        index = AzureAISearchIndex(cfg.retrieval.azure_search)
    else:
        index = InMemoryHybridIndex(cfg.retrieval)
    retriever = DocumentRetriever(
        cfg.retrieval, index, embedding_provider, repos.documents, injection.is_suspicious
    )

    catalog = SchemaCatalog.build(cfg.sql, database.tables, cfg.database.schema_name)
    router = ModelRouter(cfg.routing, registry, catalog)
    sql_service = NLToSQLService(
        sql_config=cfg.sql,
        routing_config=cfg.routing,
        registry=registry,
        router=router,
        generator=SQLGenerator(
            prompts,
            catalog,
            dialect=cfg.sql.generation_dialect,
            max_rows=cfg.sql.max_rows,
            repair_attempts=cfg.llm.json_repair_attempts,
        ),
        guard=SQLGuard(cfg.sql),
        validator=SQLValidator(cfg.sql, catalog),
        executor=SQLExecutor(
            database.query_engine, timeout_seconds=cfg.database.query_timeout_seconds
        ),
        execution_dialect=database.dialect,
    )

    policy = ToolPermissionPolicy(cfg.agents, cfg.tools)
    tools = ToolRegistry(cfg.tools, policy)
    for tool in (
        FacilityDirectoryTool(repos),
        SQLQueryTool(sql_service),
        KpiLookupTool(KpiService(cfg.kpis, repos), repos, cfg.kpis.default_kpis),
        DocumentSearchTool(retriever),
        VectorSearchTool(retriever),
        AnomalyAnalysisTool(AnomalyService(cfg.anomaly, repos)),
        EmissionForecastTool(ForecastService(cfg.forecasting, repos)),
        ScenarioAnalysisTool(ScenarioService(cfg.scenario, repos)),
        RecommendationRankingTool(RecommendationService(cfg.recommendations, repos)),
    ):
        if cfg.tools[tool.name].enabled:
            tools.register(tool)

    chain = ProviderChain(
        registry,
        cfg.llm.agent_provider_order,
        repair_template=prompts.text(COMMON_PROMPT, "json_repair", error="$error"),
        repair_attempts=cfg.llm.json_repair_attempts,
        task_orders=cfg.llm.task_provider_order,
    )
    agents = build_agents(AgentDependencies(settings=cfg, tools=tools, llm=chain, prompts=prompts))
    graph = build_graph(
        GraphNodes(
            cfg,
            agents,
            prompts,
            InputValidator(cfg.guardrails),
            injection,
            OutputValidator(cfg.guardrails),
        )
    )
    assistant = AssistantService(
        cfg,
        graph,
        repos,
        AuditService(repos),
        ConversationService(cfg.conversation, repos),
    )
    return Container(
        settings=cfg,
        database=database,
        repositories=repos,
        providers=registry,
        prompts=prompts,
        embeddings=embedding_provider,
        search_index=index,
        retriever=retriever,
        sql_service=sql_service,
        tools=tools,
        agents=agents,
        graph=graph,
        assistant=assistant,
        injection=injection,
    )
