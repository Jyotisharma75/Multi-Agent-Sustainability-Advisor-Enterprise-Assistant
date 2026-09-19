"""Typed configuration sections.

Every tunable value in the service lives here as a field with no business
default that matters: the effective values come from ``configs/base.yaml``,
the environment overlay and ``SA_`` environment variables. Sections forbid
unknown keys so a misspelt setting fails at start up instead of being ignored.

Fields ending in ``_env`` hold the *name* of an environment variable, never a
secret value, so configuration files can be committed safely.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Section(BaseModel):
    """Base class for configuration sections."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RetryConfig(Section):
    max_attempts: int = Field(ge=1, le=10)
    initial_backoff_seconds: float = Field(ge=0)
    max_backoff_seconds: float = Field(ge=0)


class AppConfig(Section):
    name: str
    environment: str
    version: str


class ApiAuthConfig(Section):
    enabled: bool
    header_name: str
    api_keys_env: str


class ApiConfig(Section):
    prefix: str
    cors_origins: list[str]
    max_request_bytes: int = Field(gt=0)
    request_timeout_seconds: float = Field(gt=0)
    analyze_intent: str
    scenario_intent: str
    auth: ApiAuthConfig


class LoggingConfig(Section):
    level: str
    json_output: bool
    include_llm_payloads: bool


class AzureSqlConfig(Section):
    server_env: str
    database_env: str
    driver: str
    auth_mode: Literal["entra", "sql_password"]
    username_env: str
    password_env: str
    token_scope: str
    query_identity_client_id_env: str
    encrypt: bool
    trust_server_certificate: bool
    connect_timeout_seconds: int = Field(gt=0)
    pool_size: int = Field(gt=0)
    max_overflow: int = Field(ge=0)
    pool_recycle_seconds: int = Field(gt=0)


class SqliteConfig(Section):
    path: str
    create_schema: bool


class DatabaseConfig(Section):
    backend: Literal["azure_sql", "sqlite"]
    schema_name: str | None
    query_timeout_seconds: float = Field(gt=0)
    readiness_query: str
    azure_sql: AzureSqlConfig
    sqlite: SqliteConfig


class AzureOpenAIConfig(Section):
    enabled: bool
    endpoint_env: str
    api_key_env: str
    auth_mode: Literal["api_key", "entra"]
    token_scope: str
    api_version: str
    chat_deployment: str
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    retry: RetryConfig


class LocalModelConfig(Section):
    enabled: bool
    model_id: str
    revision: str
    device: str
    dtype: str
    cache_dir: str | None
    max_new_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0, le=2)
    timeout_seconds: float = Field(gt=0)
    max_concurrency: int = Field(ge=1)
    num_threads: int | None
    trust_remote_code: bool
    preload: bool


class MockModelConfig(Section):
    enabled: bool


class LLMConfig(Section):
    agent_provider_order: list[str] = Field(min_length=1)
    task_provider_order: dict[str, list[str]]
    json_repair_attempts: int = Field(ge=0, le=3)
    azure_openai: AzureOpenAIConfig
    local: LocalModelConfig
    mock: MockModelConfig


class RoutingWeights(Section):
    table_count: float = Field(ge=0)
    question_length: float = Field(ge=0)
    aggregation: float = Field(ge=0)
    time_expressions: float = Field(ge=0)
    comparison: float = Field(ge=0)
    intent: float = Field(ge=0)


class RoutingConfig(Section):
    enabled: bool
    primary_when_simple: str
    primary_when_complex: str
    local_max_complexity: float = Field(ge=0, le=1)
    always_verify_above_complexity: float = Field(ge=0, le=1)
    verify_below_confidence: float = Field(ge=0, le=1)
    disagreement_confidence_penalty: float = Field(ge=0, le=1)
    max_repair_attempts: int = Field(ge=0, le=3)
    table_count_saturation: int = Field(gt=0)
    question_length_saturation: int = Field(gt=0)
    time_expression_saturation: int = Field(gt=0)
    aggregation_keywords: list[str]
    comparison_keywords: list[str]
    time_keywords: list[str]
    intent_complexity: dict[str, float]
    weights: RoutingWeights


class ExposedTable(Section):
    description: str
    columns: list[str] = Field(min_length=1)
    synonyms: list[str]


class SqlSafetyConfig(Section):
    generation_dialect: str
    allowed_statements: list[str]
    max_sql_chars: int = Field(gt=0)
    max_rows: int = Field(gt=0)
    max_joins: int = Field(ge=0)
    max_subquery_depth: int = Field(ge=0)
    allow_select_star: bool
    allow_cross_database: bool
    allow_temp_tables: bool
    allow_variables: bool
    blocked_functions: list[str]
    blocked_keywords: list[str]
    blocked_identifier_prefixes: list[str]
    system_schemas: list[str]
    accepted_schema_qualifiers: list[str]
    dry_run_rows: int = Field(gt=0)
    exposed_tables: dict[str, ExposedTable]


class AzureSearchConfig(Section):
    endpoint_env: str
    api_key_env: str
    auth_mode: Literal["api_key", "entra"]
    index_name: str
    content_field: str
    vector_field: str
    key_field: str
    use_semantic_ranker: bool
    semantic_configuration: str | None
    score_normalizer: float = Field(gt=0)
    reranker_score_normalizer: float = Field(gt=0)


class EmbeddingConfig(Section):
    provider: Literal["azure_openai", "sentence_transformers", "hash"]
    model: str
    dimensions: int = Field(gt=0)
    batch_size: int = Field(gt=0)


class ChunkingConfig(Section):
    max_chars: int = Field(gt=100)
    overlap_chars: int = Field(ge=0)


class RetrievalConfig(Section):
    backend: Literal["azure_search", "memory"]
    top_k: int = Field(gt=0)
    candidate_multiplier: int = Field(ge=1)
    rrf_k: int = Field(gt=0)
    keyword_weight: float = Field(ge=0)
    vector_weight: float = Field(ge=0)
    min_retrieval_score: float = Field(ge=0, le=1)
    bm25_k1: float = Field(gt=0)
    bm25_b: float = Field(ge=0, le=1)
    allowed_filter_fields: list[str]
    documents_path: str
    auto_ingest_on_start: bool
    azure_search: AzureSearchConfig
    embeddings: EmbeddingConfig
    chunking: ChunkingConfig


class InjectionPattern(Section):
    pattern: str
    weight: float = Field(gt=0)


class GuardrailConfig(Section):
    min_input_chars: int = Field(ge=1)
    max_input_chars: int = Field(gt=0)
    identifier_pattern: str
    max_facility_ids: int = Field(gt=0)
    injection_enabled: bool
    injection_block_threshold: float = Field(gt=0)
    injection_patterns: list[InjectionPattern]
    min_final_confidence: float = Field(ge=0, le=1)
    validation_failure_penalty: float = Field(ge=0, le=1)
    require_citations_for_documents: bool
    grounding_enabled: bool
    grounding_relative_tolerance: float = Field(ge=0)
    grounding_ignore_below: float = Field(ge=0)
    max_answer_chars: int = Field(gt=0)
    hidden_reasoning_tags: list[str]


class IntentRoute(Section):
    description: str
    keywords: list[str]
    agents: list[str] = Field(min_length=1)


class OrchestrationConfig(Section):
    max_graph_iterations: int = Field(ge=1)
    max_replans: int = Field(ge=0)
    recursion_limit: int = Field(gt=0)
    max_parallel_agents: int = Field(ge=1)
    intent_min_confidence: float = Field(ge=0, le=1)
    default_intent: str
    default_lookback_months: int = Field(gt=0)
    intents: dict[str, IntentRoute]
    agent_dependencies: dict[str, list[str]]

    @model_validator(mode="after")
    def _default_intent_exists(self) -> OrchestrationConfig:
        if self.default_intent not in self.intents:
            raise ValueError(f"default_intent {self.default_intent!r} is not a configured intent")
        return self


class AgentSpec(Section):
    enabled: bool
    responsibility: str
    allowed_tools: list[str]
    timeout_seconds: float = Field(gt=0)
    retry: RetryConfig
    max_iterations: int = Field(ge=1)
    prompt: str | None
    min_confidence: float = Field(ge=0, le=1)
    required: bool


class ToolSpec(Section):
    enabled: bool
    description: str
    timeout_seconds: float = Field(gt=0)
    retry: RetryConfig


class KpiDefinition(Section):
    description: str
    kind: Literal["ratio", "sum", "metric"]
    numerator: str
    denominator: str | None
    unit: str
    scale: float
    higher_is_better: bool
    target_metric: str | None
    target_basis: Literal["absolute", "per_month"]


class KpiConfig(Section):
    definitions: dict[str, KpiDefinition]
    on_track_tolerance: float = Field(ge=0)
    default_kpis: list[str]


class ForecastConfig(Section):
    default_horizon: int = Field(gt=0)
    max_horizon: int = Field(gt=0)
    seasonal_period: int = Field(gt=0)
    min_history_points: int = Field(gt=1)
    holdout_points: int = Field(gt=0)
    interval_coverage: float = Field(gt=0, lt=1)
    methods: list[Literal["holt_winters", "seasonal_naive", "drift"]] = Field(min_length=1)
    history_months: int = Field(gt=0)
    target_metric: str
    target_basis: Literal["absolute", "per_month"]


class SeverityBand(Section):
    name: str
    min_score: float


class AnomalyConfig(Section):
    robust_z_threshold: float = Field(gt=0)
    rolling_window: int = Field(gt=1)
    min_points: int = Field(gt=2)
    isolation_forest_enabled: bool
    isolation_forest_contamination: float = Field(gt=0, lt=0.5)
    isolation_forest_min_points: int = Field(gt=2)
    random_state: int
    severity_bands: list[SeverityBand] = Field(min_length=1)
    history_months: int = Field(gt=0)
    default_metrics: list[str] = Field(min_length=1)
    metric_units: dict[str, str]


class ScenarioLever(Section):
    kind: Literal["energy_reduction", "renewable_share", "production_change"]
    description: str
    unit: str
    min_value: float
    max_value: float
    scopes: list[int]


class ScenarioConfig(Section):
    levers: dict[str, ScenarioLever]
    baseline_months: int = Field(gt=0)


class Playbook(Section):
    issue_template: str
    business_impact_template: str
    estimated_complexity: Literal["low", "medium", "high"]
    actions: list[str] = Field(min_length=1)
    cost_reference_metric: str | None


class RecommendationConfig(Section):
    max_recommendations: int = Field(gt=0)
    min_evidence_items: int = Field(ge=1)
    severity_thresholds: dict[str, float]
    severity_scores: dict[str, float]
    complexity_scores: dict[str, float]
    priority_weights: dict[str, float]
    priority_bands: list[SeverityBand] = Field(min_length=1)
    playbooks: dict[str, Playbook]


class ConversationConfig(Section):
    max_history_turns: int = Field(ge=0)
    max_message_chars: int = Field(gt=0)


class PromptConfig(Section):
    directory: str
    version: str


class ComplianceConfig(Section):
    document_types: list[str] = Field(min_length=1)


class Settings(Section):
    """The complete, validated service configuration."""

    app: AppConfig
    api: ApiConfig
    logging: LoggingConfig
    database: DatabaseConfig
    llm: LLMConfig
    routing: RoutingConfig
    sql: SqlSafetyConfig
    retrieval: RetrievalConfig
    guardrails: GuardrailConfig
    orchestration: OrchestrationConfig
    agents: dict[str, AgentSpec]
    tools: dict[str, ToolSpec]
    kpis: KpiConfig
    forecasting: ForecastConfig
    anomaly: AnomalyConfig
    scenario: ScenarioConfig
    recommendations: RecommendationConfig
    conversation: ConversationConfig
    prompts: PromptConfig
    compliance: ComplianceConfig

    @model_validator(mode="after")
    def _cross_references(self) -> Settings:
        for intent, route in self.orchestration.intents.items():
            unknown = [a for a in route.agents if a not in self.agents]
            if unknown:
                raise ValueError(f"intent {intent!r} references unknown agents {unknown}")
        for agent, spec in self.agents.items():
            unknown_tools = [t for t in spec.allowed_tools if t not in self.tools]
            if unknown_tools:
                raise ValueError(f"agent {agent!r} references unknown tools {unknown_tools}")
        for forced in (self.api.analyze_intent, self.api.scenario_intent):
            if forced not in self.orchestration.intents:
                raise ValueError(f"api intent {forced!r} is not a configured intent")
        for agent, deps in self.orchestration.agent_dependencies.items():
            unknown = [a for a in [agent, *deps] if a not in self.agents]
            if unknown:
                raise ValueError(f"agent_dependencies references unknown agents {unknown}")
        return self
