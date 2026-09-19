"""Natural language to SQL with routed generation and cross model verification.

Flow for one question:

1. the router scores complexity and picks a generating and a verifying model
2. the generator writes SQL; the guard and validator must accept it. On
   rejection the reason is fed back once to the same model, then the task
   escalates to the verifier model
3. when complexity or the draft's confidence calls for it, the verifier
   reviews the SQL. ``approve`` keeps it; ``revise`` triggers a dry run of
   both statements and a comparison of their results; ``reject`` escalates
   generation to the verifier model
4. the final statement runs on the read only engine with a row cap. A
   database error is fed back for one repair attempt

Nothing produced by a model runs without passing the guard and the validator.
"""

from __future__ import annotations

from dataclasses import dataclass

from sustainability_advisor.config.models import RoutingConfig, SqlSafetyConfig
from sustainability_advisor.domain.errors import LLMError, SQLSafetyError
from sustainability_advisor.llm.factory import ProviderRegistry
from sustainability_advisor.observability.logging import get_logger
from sustainability_advisor.sql.executor import SQLExecutor
from sustainability_advisor.sql.generator import SQLGenerator
from sustainability_advisor.sql.guard import SQLGuard
from sustainability_advisor.sql.models import QueryResult, RoutingDecision, SQLAnswer, SQLDraft
from sustainability_advisor.sql.router import ModelRouter
from sustainability_advisor.sql.validator import SQLValidator

logger = get_logger(__name__)


@dataclass(slots=True)
class _Candidate:
    provider: str
    draft: SQLDraft
    display_sql: str
    execution_sql: str
    tables: list[str]


class NLToSQLService:
    def __init__(
        self,
        *,
        sql_config: SqlSafetyConfig,
        routing_config: RoutingConfig,
        registry: ProviderRegistry,
        router: ModelRouter,
        generator: SQLGenerator,
        guard: SQLGuard,
        validator: SQLValidator,
        executor: SQLExecutor,
        execution_dialect: str,
    ) -> None:
        self._sql = sql_config
        self._routing = routing_config
        self._registry = registry
        self._router = router
        self._generator = generator
        self._guard = guard
        self._validator = validator
        self._executor = executor
        self._execution_dialect = execution_dialect

    # -- validation ------------------------------------------------------
    def prepare(self, sql: str, *, max_rows: int) -> tuple[str, str, list[str]]:
        """Guard and validate SQL. Returns display SQL, execution SQL and tables."""
        guarded = self._guard.check(sql, dialect=self._sql.generation_dialect)
        if not guarded.ok or guarded.expression is None:
            raise SQLSafetyError(
                "The generated SQL was rejected by the safety guard.",
                details={"issues": [i.message for i in guarded.issues]},
            )
        validated = self._validator.validate(
            guarded.expression, execution_dialect=self._execution_dialect, max_rows=max_rows
        )
        if not validated.ok:
            raise SQLSafetyError(
                "The generated SQL failed validation.", details={"issues": validated.issues}
            )
        return validated.display_sql, validated.execution_sql, validated.tables

    # -- generation --------------------------------------------------------
    async def _generate_valid(
        self,
        provider_name: str,
        question: str,
        context: str,
        max_rows: int,
        feedback: str | None = None,
    ) -> _Candidate:
        provider = self._registry.get(provider_name)
        attempts = self._routing.max_repair_attempts + 1
        last_error: SQLSafetyError | None = None
        for _ in range(attempts):
            draft = await self._generator.generate(
                provider, question, context=context, feedback=feedback
            )
            try:
                display, execution, tables = self.prepare(draft.sql, max_rows=max_rows)
            except SQLSafetyError as exc:
                last_error = exc
                feedback = "; ".join(str(i) for i in exc.details.get("issues", [exc.message]))
                logger.info("sql.generation_rejected", provider=provider_name, issues=feedback)
                continue
            return _Candidate(provider_name, draft, display, execution, tables)
        assert last_error is not None
        raise last_error

    async def _initial_candidate(
        self,
        decision: RoutingDecision,
        question: str,
        context: str,
        max_rows: int,
        notes: list[str],
    ) -> _Candidate:
        try:
            return await self._generate_valid(decision.primary, question, context, max_rows)
        except (SQLSafetyError, LLMError) as exc:
            if decision.verifier is None:
                raise
            notes.append(f"{decision.primary} could not produce valid SQL; escalated.")
            logger.info("sql.escalation", source=decision.primary, target=decision.verifier)
            feedback = "; ".join(str(i) for i in exc.details.get("issues", [exc.message]))
            return await self._generate_valid(
                decision.verifier, question, context, max_rows, feedback=feedback
            )

    # -- public API --------------------------------------------------------
    async def answer(
        self, question: str, *, intent: str, context: str = "", max_rows: int | None = None
    ) -> SQLAnswer:
        limit = min(max_rows or self._sql.max_rows, self._sql.max_rows)
        decision = self._router.route(question, intent)
        notes: list[str] = []
        candidate = await self._initial_candidate(decision, question, context, limit, notes)
        verifier = decision.verifier if candidate.provider == decision.primary else decision.primary
        confidence = candidate.draft.confidence
        verification = "not_required"
        verified_by: str | None = None

        if decision.verifier is not None and self._router.should_verify(decision, confidence):
            verified_by = verifier
            candidate, confidence, verification = await self._verify(
                candidate, verifier, question, context, limit, notes
            )

        result = await self._execute_with_repair(candidate, question, context, limit, notes)
        return SQLAnswer(
            question=question,
            sql=candidate.display_sql,
            execution_sql=candidate.execution_sql,
            tables=candidate.tables,
            result=result,
            generated_by=candidate.provider,
            verified_by=verified_by,
            verification=verification,
            complexity=decision.complexity,
            confidence=round(confidence, 4),
            notes=notes,
        )

    async def _verify(
        self,
        candidate: _Candidate,
        verifier: str | None,
        question: str,
        context: str,
        limit: int,
        notes: list[str],
    ) -> tuple[_Candidate, float, str]:
        assert verifier is not None
        provider = self._registry.try_get(verifier)
        if provider is None:
            return candidate, candidate.draft.confidence, "unavailable"
        try:
            review = await self._generator.review(provider, question, candidate.display_sql)
        except LLMError:
            notes.append("The verifier model was unavailable; SQL was not cross checked.")
            return candidate, candidate.draft.confidence, "unavailable"

        mean_confidence = (candidate.draft.confidence + review.confidence) / 2.0
        penalty = self._routing.disagreement_confidence_penalty
        if review.verdict == "approve":
            return candidate, mean_confidence, "approved"

        if review.verdict == "revise" and review.corrected_sql:
            try:
                display, execution, tables = self.prepare(review.corrected_sql, max_rows=limit)
            except SQLSafetyError:
                notes.append("The verifier's correction failed validation; original SQL kept.")
                return candidate, candidate.draft.confidence * penalty, "disagreement"
            revised = _Candidate(
                verifier,
                SQLDraft(
                    sql=review.corrected_sql, confidence=review.confidence, tables_used=tables
                ),
                display,
                execution,
                tables,
            )
            return await self._settle(candidate, revised, mean_confidence, penalty, notes)

        # Rejected, or a revision without SQL: let the verifier write it.
        notes.append("The verifier rejected the SQL; regenerated by the verifier model.")
        try:
            regenerated = await self._generate_valid(
                verifier, question, context, limit, feedback="; ".join(review.issues) or None
            )
        except (SQLSafetyError, LLMError):
            return candidate, candidate.draft.confidence * penalty, "disagreement"
        return regenerated, regenerated.draft.confidence * penalty, "revised"

    async def _settle(
        self,
        original: _Candidate,
        revised: _Candidate,
        mean_confidence: float,
        penalty: float,
        notes: list[str],
    ) -> tuple[_Candidate, float, str]:
        """Dry run both statements and decide which one to keep."""
        dry_rows = self._sql.dry_run_rows
        original_result = await self._try_execute(original.execution_sql, dry_rows)
        revised_result = await self._try_execute(revised.execution_sql, dry_rows)
        if original_result is None and revised_result is not None:
            notes.append("The original SQL failed a dry run; the verifier's correction was used.")
            return revised, revised.draft.confidence, "revised"
        if revised_result is None:
            return original, original.draft.confidence * penalty, "disagreement"
        assert original_result is not None
        if (
            len(original_result.columns) == len(revised_result.columns)
            and original_result.rows == revised_result.rows
        ):
            return original, mean_confidence, "approved"
        winner = revised if revised.provider == self._router.authoritative_provider else original
        notes.append(
            f"The two models' SQL returned different results; kept the {winner.provider} version."
        )
        return (
            winner,
            min(original.draft.confidence, revised.draft.confidence) * penalty,
            ("disagreement"),
        )

    async def _try_execute(self, sql: str, rows: int) -> QueryResult | None:
        try:
            return await self._executor.execute(sql, max_rows=rows)
        except SQLSafetyError:
            return None

    async def _execute_with_repair(
        self,
        candidate: _Candidate,
        question: str,
        context: str,
        limit: int,
        notes: list[str],
    ) -> QueryResult:
        try:
            return await self._executor.execute(candidate.execution_sql, max_rows=limit)
        except SQLSafetyError as exc:
            if self._routing.max_repair_attempts < 1:
                raise
            error = str(exc.details.get("error", exc.message))
            notes.append("The first query failed on the database and was repaired once.")
            repaired = await self._generate_valid(
                self._router.authoritative_provider
                if self._registry.try_get(self._router.authoritative_provider)
                else candidate.provider,
                question,
                context,
                limit,
                feedback=f"The database returned: {error}",
            )
            candidate.display_sql = repaired.display_sql
            candidate.execution_sql = repaired.execution_sql
            candidate.tables = repaired.tables
            candidate.provider = repaired.provider
            return await self._executor.execute(repaired.execution_sql, max_rows=limit)
