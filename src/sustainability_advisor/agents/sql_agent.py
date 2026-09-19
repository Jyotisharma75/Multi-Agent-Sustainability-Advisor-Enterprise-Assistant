"""SQL Agent.

Responsibility: natural language data querying. It sends the question to the
Azure SQL query tool, which routes generation between the local Hugging Face
model and Azure OpenAI by complexity, cross verifies with the other model,
and executes only guarded, validated, read only SQL. The agent turns the
result set into evidence the synthesizer can quote.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from sustainability_advisor.agents.base import AgentInput, BaseAgent, RunState
from sustainability_advisor.agents.evidence import compact_rows, evidence_id
from sustainability_advisor.domain.models import AgentResult, AgentStatus, Evidence, EvidenceKind
from sustainability_advisor.sql.models import SQLAnswer
from sustainability_advisor.tools.data_tools import SQLQueryInput

SQL_TOOL = "azure_sql_query"
EVIDENCE_ROWS = 25
SUMMARY_ROWS = 10


class SQLAgentOutput(BaseModel):
    sql: str
    tables: list[str]
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    generated_by: str
    verified_by: str | None
    verification: str
    complexity: float
    notes: list[str]


class SQLAgent(BaseAgent):
    name = "sql_agent"
    output_data_model = SQLAgentOutput

    async def _run(self, inp: AgentInput, state: RunState) -> AgentResult:
        e = inp.entities
        context_parts = [f"reporting period {e.period_start} to {e.period_end}"]
        if e.facility_ids:
            context_parts.append(f"facilities {', '.join(e.facility_ids)}")
        answer = await self.call_tool(
            state,
            SQL_TOOL,
            SQLQueryInput(question=inp.query, intent=inp.intent, context="; ".join(context_parts)),
            SQLAnswer,
        )
        result = answer.result
        rows = compact_rows(result.columns, result.rows, EVIDENCE_ROWS)
        numeric: dict[str, float | None] = {}
        for index, row in enumerate(rows):
            for column, value in row.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    numeric[f"r{index}.{column}"] = float(value)
        summary = (
            f"The query over {', '.join(answer.tables)} returned {result.row_count} rows"
            + (" (truncated)" if result.truncated else "")
            + f". Columns: {', '.join(result.columns)}."
        )
        if rows:
            shown = rows[:SUMMARY_ROWS]
            summary += " Rows: " + "; ".join(
                ", ".join(f"{k}={v}" for k, v in row.items()) for row in shown
            )
            if len(rows) > len(shown):
                summary += f"; and {len(rows) - len(shown)} more"
            summary += "."
        evidence = Evidence(
            evidence_id=evidence_id(EvidenceKind.SQL_RESULT, answer.execution_sql),
            kind=EvidenceKind.SQL_RESULT,
            source=SQL_TOOL,
            summary=summary,
            values=numeric,
            attributes={
                "sql": answer.sql,
                "rows": rows,
                "generated_by": answer.generated_by,
                "verified_by": answer.verified_by,
                "verification": answer.verification,
            },
            confidence=answer.confidence,
        )
        return self.result(
            state,
            summary=summary,
            confidence=answer.confidence,
            status=AgentStatus.SUCCESS if result.row_count else AgentStatus.PARTIAL,
            evidence=[evidence],
            data=SQLAgentOutput(
                sql=answer.sql,
                tables=answer.tables,
                columns=result.columns,
                rows=rows,
                row_count=result.row_count,
                truncated=result.truncated,
                generated_by=answer.generated_by,
                verified_by=answer.verified_by,
                verification=answer.verification,
                complexity=answer.complexity,
                notes=answer.notes,
            ).model_dump(mode="json"),
        )
