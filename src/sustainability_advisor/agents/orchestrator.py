"""Orchestrator Agent.

Responsibility: understand the request and plan the work. It

1. loads the facility directory (the only data it reads)
2. classifies intent: a forced intent from the API wins; otherwise a model
   classifies against the intents in configuration; if no model is available
   or the model is not confident enough, a keyword classifier built from the
   same configuration decides
3. resolves entities (facilities, period, KPIs, metrics, horizon, levers,
   jurisdiction) in precedence order: explicit API values, model extraction
   validated against known values, deterministic text matching, and finally
   configured defaults anchored on the latest period with data
4. builds a task plan: the intent's agents from configuration, ordered into
   dependency levels so independent agents can run in parallel

The plan is data, not code: changing which agents answer which intent is a
configuration change.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

from pydantic import BaseModel, Field

from sustainability_advisor.agents.base import AgentInput, BaseAgent, RequestEntities, RunState
from sustainability_advisor.domain.errors import LLMError
from sustainability_advisor.domain.models import AgentResult
from sustainability_advisor.services.periods import add_months, month_end, trailing_months
from sustainability_advisor.tools.data_tools import (
    FacilityDirectoryInput,
    FacilityDirectoryOutput,
    FacilityEntry,
)

DIRECTORY_TOOL = "facility_directory"
PROMPT = "intent_classification"

_ISO = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_QUARTER = re.compile(r"\bq([1-4])\s*(\d{4})\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(?:in|for|during|year)\s+(20\d{2}|19\d{2})\b", re.IGNORECASE)
_TRAILING = re.compile(r"\b(?:last|past|previous|trailing)\s+(\d{1,3})\s+(month|year)s?\b", re.I)
_HORIZON = re.compile(r"\b(?:next|coming|following)\s+(\d{1,3})\s+(month|year)s?\b", re.I)


class IntentClassification(BaseModel):
    """What the intent model must return."""

    intent: str
    confidence: float = Field(ge=0, le=1)
    facility_ids: list[str] = Field(default_factory=list)
    period_start: date | None = None
    period_end: date | None = None
    kpis: list[str] | None = None
    metrics: list[str] | None = None
    horizon_months: int | None = None
    levers: dict[str, float] = Field(default_factory=dict)
    jurisdiction: str | None = None


class PlannedTask(BaseModel):
    agent: str
    depends_on: list[str]


class TaskPlan(BaseModel):
    intent: str
    tasks: list[PlannedTask]
    levels: list[list[str]]


class OrchestratorOutput(BaseModel):
    intent: str
    intent_confidence: float
    classification_method: str
    entities: RequestEntities
    plan: TaskPlan
    notes: list[str]


class OrchestratorAgent(BaseAgent):
    name = "orchestrator"
    output_data_model = OrchestratorOutput

    # -- classification ----------------------------------------------------
    def keyword_intent(self, query: str) -> tuple[str, float]:
        text = query.lower()
        scores = {
            name: sum(1 for kw in route.keywords if re.search(rf"\b{re.escape(kw.lower())}", text))
            for name, route in self.settings.orchestration.intents.items()
        }
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, hits = ranked[0]
        if hits == 0:
            return self.settings.orchestration.default_intent, 0.0
        runner_up = ranked[1][1] if len(ranked) > 1 else 0
        return best, round(hits / (hits + runner_up + 1), 4)

    async def _classify(
        self, inp: AgentInput, state: RunState, directory: list[FacilityEntry], notes: list[str]
    ) -> tuple[str, float, str, IntentClassification | None]:
        intents = self.settings.orchestration.intents
        if inp.forced_intent and inp.intent in intents:
            return inp.intent, 1.0, "forced", None
        intent_lines = "\n".join(f"- {name}: {r.description}" for name, r in intents.items())
        facility_lines = "\n".join(
            f"- {e.facility.facility_id}: {e.facility.name}" for e in directory
        )
        latest = max((e.latest_period for e in directory if e.latest_period), default=None)
        try:
            classification = await self.call_llm(
                state,
                PROMPT,
                IntentClassification,
                question=inp.query,
                intents=intent_lines,
                facilities=facility_lines or "none",
                kpis=", ".join(self.settings.kpis.definitions),
                metrics=", ".join(self.settings.anomaly.default_metrics),
                levers=", ".join(self.settings.scenario.levers),
                today=date.today().isoformat(),
                latest_period=latest.isoformat() if latest else "unknown",
                history="\n".join(inp.history) or "none",
            )
        except LLMError as exc:
            notes.append(f"Intent model unavailable ({exc.code}); keyword classification used.")
            intent, confidence = self.keyword_intent(inp.query)
            return intent, confidence, "keywords", None
        minimum = self.settings.orchestration.intent_min_confidence
        if classification.intent in intents and classification.confidence >= minimum:
            return classification.intent, classification.confidence, "model", classification
        notes.append("The intent model was not confident; keyword classification used.")
        intent, confidence = self.keyword_intent(inp.query)
        return intent, confidence, "keywords", classification

    # -- entities ----------------------------------------------------------
    def _facilities(
        self,
        inp: AgentInput,
        directory: list[FacilityEntry],
        extracted: IntentClassification | None,
        notes: list[str],
    ) -> list[str]:
        known = {e.facility.facility_id for e in directory}
        if inp.entities.facility_ids:
            return [f for f in inp.entities.facility_ids if f in known]
        if extracted and extracted.facility_ids:
            valid = [f for f in extracted.facility_ids if f in known]
            if valid:
                return valid
        text = inp.query.lower()
        words = set(re.findall(r"[a-z0-9-]+", text))
        # A facility matches on its id, its full name, or the leading word of
        # its name (conventionally the site location) when that word is unique.
        leading = [e.facility.name.split()[0].lower() for e in directory if e.facility.name]
        matched = [
            e.facility.facility_id
            for e in directory
            if e.facility.facility_id.lower() in text
            or e.facility.name.lower() in text
            or (
                e.facility.name
                and e.facility.name.split()[0].lower() in words
                and leading.count(e.facility.name.split()[0].lower()) == 1
            )
        ]
        if matched:
            return matched
        limit = self.settings.guardrails.max_facility_ids
        notes.append("No facility was named; all active facilities are in scope.")
        return [e.facility.facility_id for e in directory][:limit]

    def _period(
        self,
        inp: AgentInput,
        extracted: IntentClassification | None,
        latest: date | None,
    ) -> tuple[date, date]:
        e = inp.entities
        if e.period_explicit:
            return e.period_start, e.period_end
        if (
            extracted
            and extracted.period_start
            and extracted.period_end
            and extracted.period_start <= extracted.period_end
        ):
            return extracted.period_start, extracted.period_end
        text = inp.query
        isos = _ISO.findall(text)
        if len(isos) >= 2:
            start, end = sorted(date.fromisoformat(d) for d in isos[:2])
            return start, end
        quarter = _QUARTER.search(text)
        if quarter:
            q, year = int(quarter.group(1)), int(quarter.group(2))
            start = date(year, 3 * (q - 1) + 1, 1)
            return start, add_months(start, 3) - timedelta(days=1)
        year_match = _YEAR.search(text)
        if year_match:
            year = int(year_match.group(1))
            return date(year, 1, 1), date(year, 12, 31)
        anchor = latest or date.today()
        trailing = _TRAILING.search(text)
        months = self.settings.orchestration.default_lookback_months
        if trailing:
            months = int(trailing.group(1)) * (12 if trailing.group(2).lower() == "year" else 1)
        window = trailing_months(anchor, months)
        return window.start, month_end(window.end)

    def _entities(
        self,
        inp: AgentInput,
        directory: list[FacilityEntry],
        extracted: IntentClassification | None,
        notes: list[str],
    ) -> RequestEntities:
        e = inp.entities
        facilities = self._facilities(inp, directory, extracted, notes)
        latest = max(
            (
                x.latest_period
                for x in directory
                if x.latest_period and x.facility.facility_id in facilities
            ),
            default=None,
        )
        start, end = self._period(inp, extracted, latest)
        known_kpis = set(self.settings.kpis.definitions)
        kpis = e.kpis or (
            [k for k in (extracted.kpis or []) if k in known_kpis] if extracted else None
        )
        if not kpis:
            text = inp.query.lower()
            kpis = [k for k in known_kpis if k.replace("_", " ") in text] or None
        allowed_metrics = set(self.settings.anomaly.metric_units)
        metrics = e.metrics or (
            [m for m in (extracted.metrics or []) if m.split(":", 1)[0] in allowed_metrics]
            if extracted
            else None
        )
        horizon = e.horizon or (extracted.horizon_months if extracted else None)
        if horizon is None:
            match = _HORIZON.search(inp.query)
            if match:
                horizon = int(match.group(1)) * (12 if match.group(2).lower() == "year" else 1)
        if horizon is not None:
            horizon = min(horizon, self.settings.forecasting.max_horizon)
        levers = e.levers or (
            {k: v for k, v in extracted.levers.items() if k in self.settings.scenario.levers}
            if extracted
            else {}
        )
        return RequestEntities(
            facility_ids=facilities,
            period_start=start,
            period_end=end,
            period_explicit=e.period_explicit,
            kpis=kpis or None,
            metrics=metrics or None,
            horizon=horizon,
            scopes=e.scopes,
            levers=levers,
            jurisdiction=e.jurisdiction or (extracted.jurisdiction if extracted else None),
        )

    # -- planning ----------------------------------------------------------
    def plan(self, intent: str) -> TaskPlan:
        orchestration = self.settings.orchestration
        agents = [
            a for a in orchestration.intents[intent].agents if self.settings.agents[a].enabled
        ]
        in_plan = set(agents)
        tasks = [
            PlannedTask(
                agent=a,
                depends_on=[d for d in orchestration.agent_dependencies.get(a, []) if d in in_plan],
            )
            for a in agents
        ]
        levels: list[list[str]] = []
        placed: set[str] = set()
        remaining = {t.agent: set(t.depends_on) for t in tasks}
        while remaining:
            ready = sorted(a for a, deps in remaining.items() if deps <= placed)
            if not ready:
                raise ValueError(f"Cyclic agent dependencies among {sorted(remaining)}")
            levels.append(ready)
            placed.update(ready)
            for agent in ready:
                remaining.pop(agent)
        return TaskPlan(intent=intent, tasks=tasks, levels=levels)

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        notes: list[str] = []
        directory = await self.call_tool(
            state, DIRECTORY_TOOL, FacilityDirectoryInput(), FacilityDirectoryOutput
        )
        intent, confidence, method, extracted = await self._classify(
            inp, state, directory.facilities, notes
        )
        entities = self._entities(inp, directory.facilities, extracted, notes)
        plan = self.plan(intent)
        output = OrchestratorOutput(
            intent=intent,
            intent_confidence=confidence,
            classification_method=method,
            entities=entities,
            plan=plan,
            notes=notes,
        )
        return self.result(
            state,
            summary=f"Intent {intent} ({method}); plan: "
            + " then ".join("+".join(level) for level in plan.levels),
            confidence=confidence,
            data=output.model_dump(mode="json"),
        )
