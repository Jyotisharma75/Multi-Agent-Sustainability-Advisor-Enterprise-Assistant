"""LangGraph orchestration graph.

    input_guard -> orchestrate -> execute -> validate -> synthesize -> finalize
         |              |                      |
         +--(blocked)---+----> finalize        +--(replan, bounded)--> execute

* ``input_guard`` validates and normalises the message and blocks prompt
  injection
* ``orchestrate`` runs the Orchestrator Agent: intent, entities, task plan
* ``execute`` runs pending agents level by level; agents in one level run
  concurrently up to ``max_parallel_agents``; each receives the results of the
  agents it depends on
* ``validate`` checks required agents, citations, evidence and confidence. If
  a required agent failed and the replan budget allows, only the failed agents
  and their dependants are scheduled again
* ``synthesize`` runs the Response Synthesizer on all results
* ``finalize`` sanitises the answer and assembles the response payload

Loops are bounded twice: by ``max_replans`` and ``max_graph_iterations`` in
the state, and by LangGraph's ``recursion_limit`` from configuration.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from sustainability_advisor.agents.base import AgentInput, BaseAgent, RequestEntities
from sustainability_advisor.agents.orchestrator import OrchestratorOutput, TaskPlan
from sustainability_advisor.agents.response_synthesizer import ResponseSynthesizer
from sustainability_advisor.config.models import Settings
from sustainability_advisor.domain.errors import AdvisorError
from sustainability_advisor.domain.models import AgentResult, AgentStatus
from sustainability_advisor.graph.state import AssistantState
from sustainability_advisor.guardrails.confidence import validate_results
from sustainability_advisor.guardrails.input_validator import InputValidator
from sustainability_advisor.guardrails.output_validator import OutputValidator
from sustainability_advisor.guardrails.prompt_injection import InjectionDetector
from sustainability_advisor.observability.logging import get_logger
from sustainability_advisor.prompts.loader import PromptLibrary

logger = get_logger(__name__)

ORCHESTRATOR = "orchestrator"
SYNTHESIZER = "response_synthesizer"
COMMON_PROMPT = "common"


class GraphNodes:
    def __init__(
        self,
        settings: Settings,
        agents: dict[str, BaseAgent],
        prompts: PromptLibrary,
        input_validator: InputValidator,
        injection: InjectionDetector,
        output_validator: OutputValidator,
    ) -> None:
        self._settings = settings
        self._agents = agents
        self._prompts = prompts
        self._input = input_validator
        self._injection = injection
        self._output = output_validator

    def _agent_input(
        self, state: AssistantState, *, intent: str, upstream: list[AgentResult]
    ) -> AgentInput:
        return AgentInput(
            request_id=state["request_id"],
            conversation_id=state["conversation_id"],
            query=state["query"],
            intent=intent,
            entities=state.get("entities") or state["hints"],
            upstream=upstream,
            history=state.get("history", []),
            forced_intent=bool(state.get("forced_intent")),
        )

    # -- nodes -------------------------------------------------------------
    async def input_guard(self, state: AssistantState) -> dict[str, Any]:
        try:
            query = self._input.message(state["query"])
            self._injection.check(query)
        except AdvisorError as exc:
            logger.warning("guard.input_blocked", code=exc.code)
            return {"blocked": True, "block_code": exc.code, "block_message": exc.message}
        return {"query": query, "blocked": False}

    async def orchestrate(self, state: AssistantState) -> dict[str, Any]:
        agent = self._agents[ORCHESTRATOR]
        result = await agent.execute(
            self._agent_input(state, intent=state.get("forced_intent") or "", upstream=[])
        )
        if result.status == AgentStatus.FAILED:
            return {
                "blocked": True,
                "block_code": result.errors[0] if result.errors else "orchestration_failed",
                "block_message": self._prompts.text(COMMON_PROMPT, "orchestration_failed"),
                "orchestrator_result": result,
            }
        output = OrchestratorOutput.model_validate(result.data)
        first_pass = [agent for level in output.plan.levels for agent in level]
        return {
            "intent": output.intent,
            "intent_confidence": output.intent_confidence,
            "classification_method": output.classification_method,
            "entities": output.entities,
            "plan": output.plan,
            "pending": first_pass,
            "results": {},
            "iteration": 0,
            "replans": 0,
            "notes": output.notes,
            "orchestrator_result": result,
        }

    async def execute(self, state: AssistantState) -> dict[str, Any]:
        plan: TaskPlan = state["plan"]
        results = dict(state.get("results", {}))
        pending = set(state.get("pending", []))
        dependencies = {t.agent: t.depends_on for t in plan.tasks}
        semaphore = asyncio.Semaphore(self._settings.orchestration.max_parallel_agents)

        async def run(agent_name: str) -> AgentResult:
            upstream = [results[d] for d in dependencies.get(agent_name, []) if d in results]
            async with semaphore:
                return await self._agents[agent_name].execute(
                    self._agent_input(state, intent=state["intent"], upstream=upstream)
                )

        for level in plan.levels:
            batch = [a for a in level if a in pending]
            if not batch:
                continue
            outcomes = await asyncio.gather(*(run(a) for a in batch))
            for name, outcome in zip(batch, outcomes, strict=True):
                results[name] = outcome
        return {"results": results, "pending": [], "iteration": state.get("iteration", 0) + 1}

    async def validate(self, state: AssistantState) -> dict[str, Any]:
        results = list(state.get("results", {}).values())
        required = {
            name
            for name in (t.agent for t in state["plan"].tasks)
            if self._settings.agents[name].required
        }
        report = validate_results(results, required, self._settings.guardrails)
        update: dict[str, Any] = {"validation": report}
        orchestration = self._settings.orchestration
        can_replan = (
            report.failed_required_agents
            and state.get("replans", 0) < orchestration.max_replans
            and state.get("iteration", 0) < orchestration.max_graph_iterations
        )
        if can_replan:
            rerun = set(report.failed_required_agents)
            for task in state["plan"].tasks:
                if rerun & set(task.depends_on):
                    rerun.add(task.agent)
            update["pending"] = sorted(rerun)
            update["replans"] = state.get("replans", 0) + 1
            logger.info("graph.replan", agents=sorted(rerun), replans=update["replans"])
        return update

    async def synthesize(self, state: AssistantState) -> dict[str, Any]:
        upstream = list(state.get("results", {}).values())
        synthesizer = self._agents[SYNTHESIZER]
        inp = self._agent_input(state, intent=state["intent"], upstream=upstream)
        result = await synthesizer.execute(inp)
        if result.status == AgentStatus.FAILED and isinstance(synthesizer, ResponseSynthesizer):
            logger.warning("graph.synthesis_fallback", errors=result.errors)
            fallback = synthesizer.fallback(inp)
            result = fallback.model_copy(
                update={"errors": result.errors, "latency_ms": result.latency_ms}
            )
        return {"synthesis": result}

    async def finalize(self, state: AssistantState) -> dict[str, Any]:
        if state.get("blocked"):
            return {
                "final": {
                    "blocked": True,
                    "code": state.get("block_code"),
                    "answer": self._output.sanitise(
                        state.get("block_message")
                        or self._prompts.text(COMMON_PROMPT, "blocked_message")
                    ),
                }
            }
        synthesis = state["synthesis"]
        data = synthesis.data
        report = state["validation"]
        grounded = bool(data.get("grounding", {}).get("grounded"))
        confidence = report.confidence if grounded else 0.0
        if not report.passed:
            confidence *= self._settings.guardrails.validation_failure_penalty
        low = confidence < self._settings.guardrails.min_final_confidence
        warnings = list(report.issues) + list(state.get("notes", []))
        if synthesis.status == AgentStatus.FAILED:
            warnings.append("The answer could not be composed.")
        if low:
            warnings.append(self._prompts.text(COMMON_PROMPT, "low_confidence_notice"))
        return {
            "final": {
                "blocked": False,
                "answer": self._output.sanitise(str(data.get("answer", ""))),
                "key_findings": [self._output.sanitise(f) for f in data.get("key_findings", [])],
                "decision_explanation": self._output.sanitise(
                    str(data.get("decision_explanation", ""))
                ),
                "method": data.get("method"),
                "confidence": round(confidence, 4),
                "low_confidence": low,
                "grounded": grounded,
                "citation_markers": data.get("citation_markers", {}),
                "evidence_ids": data.get("evidence_ids", []),
                "warnings": warnings,
            }
        }

    # -- routing -----------------------------------------------------------
    @staticmethod
    def after_guard(state: AssistantState) -> Literal["orchestrate", "finalize"]:
        return "finalize" if state.get("blocked") else "orchestrate"

    @staticmethod
    def after_orchestrate(state: AssistantState) -> Literal["execute", "finalize"]:
        return "finalize" if state.get("blocked") else "execute"

    @staticmethod
    def after_validate(state: AssistantState) -> Literal["execute", "synthesize"]:
        return "execute" if state.get("pending") else "synthesize"


def build_graph(nodes: GraphNodes) -> Any:
    graph = StateGraph(AssistantState)
    graph.add_node("input_guard", nodes.input_guard)
    graph.add_node("orchestrate", nodes.orchestrate)
    graph.add_node("execute", nodes.execute)
    graph.add_node("validate", nodes.validate)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("finalize", nodes.finalize)
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges("input_guard", nodes.after_guard)
    graph.add_conditional_edges("orchestrate", nodes.after_orchestrate)
    graph.add_edge("execute", "validate")
    graph.add_conditional_edges("validate", nodes.after_validate)
    graph.add_edge("synthesize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def initial_state(
    *,
    request_id: str,
    conversation_id: str,
    query: str,
    history: list[str],
    hints: RequestEntities,
    forced_intent: str | None,
) -> AssistantState:
    return AssistantState(
        request_id=request_id,
        conversation_id=conversation_id,
        query=query,
        history=history,
        hints=hints,
        forced_intent=forced_intent,
    )
