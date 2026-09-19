"""SQL generation and review with a specific model provider."""

from __future__ import annotations

from datetime import date

from sustainability_advisor.llm.base import LLMProvider
from sustainability_advisor.llm.structured import complete_structured
from sustainability_advisor.prompts.loader import PromptLibrary
from sustainability_advisor.sql.catalog import SchemaCatalog
from sustainability_advisor.sql.models import SQLDraft, SQLReview

GENERATION_PROMPT = "sql_generation"
VERIFICATION_PROMPT = "sql_verification"
COMMON_PROMPT = "common"


class SQLGenerator:
    def __init__(
        self,
        prompts: PromptLibrary,
        catalog: SchemaCatalog,
        *,
        dialect: str,
        max_rows: int,
        repair_attempts: int,
    ) -> None:
        self._prompts = prompts
        self._catalog = catalog
        self._dialect = dialect
        self._max_rows = max_rows
        self._repair_attempts = repair_attempts

    def _repair_template(self) -> str:
        return self._prompts.text(COMMON_PROMPT, "json_repair", error="$error")

    async def generate(
        self,
        provider: LLMProvider,
        question: str,
        *,
        context: str,
        feedback: str | None = None,
    ) -> SQLDraft:
        messages = self._prompts.messages(
            GENERATION_PROMPT,
            schema=self._catalog.describe(),
            dialect=self._dialect,
            question=question,
            context=context or "none",
            max_rows=self._max_rows,
            today=date.today().isoformat(),
            feedback=feedback or "none",
        )
        draft, _ = await complete_structured(
            provider,
            messages,
            SQLDraft,
            task=GENERATION_PROMPT,
            repair_template=self._repair_template(),
            repair_attempts=self._repair_attempts,
        )
        return draft

    async def review(self, provider: LLMProvider, question: str, sql: str) -> SQLReview:
        messages = self._prompts.messages(
            VERIFICATION_PROMPT,
            schema=self._catalog.describe(),
            dialect=self._dialect,
            question=question,
            sql=sql,
        )
        review, _ = await complete_structured(
            provider,
            messages,
            SQLReview,
            task=VERIFICATION_PROMPT,
            repair_template=self._repair_template(),
            repair_attempts=self._repair_attempts,
        )
        return review
