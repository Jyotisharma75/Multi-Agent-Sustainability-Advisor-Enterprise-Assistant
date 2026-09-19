"""Tool permission policy: an agent may call only the tools its spec allows."""

from __future__ import annotations

from sustainability_advisor.config.models import AgentSpec, ToolSpec
from sustainability_advisor.domain.errors import ToolPermissionError


class ToolPermissionPolicy:
    def __init__(self, agents: dict[str, AgentSpec], tools: dict[str, ToolSpec]) -> None:
        self._agents = agents
        self._tools = tools

    def check(self, agent: str, tool: str) -> None:
        spec = self._agents.get(agent)
        if spec is None or not spec.enabled:
            raise ToolPermissionError(f"Agent {agent!r} is not enabled.")
        tool_spec = self._tools.get(tool)
        if tool_spec is None or not tool_spec.enabled:
            raise ToolPermissionError(f"Tool {tool!r} is not enabled.")
        if tool not in spec.allowed_tools:
            raise ToolPermissionError(
                f"Agent {agent!r} is not permitted to use tool {tool!r}.",
                details={"agent": agent, "tool": tool},
            )

    def allowed(self, agent: str) -> list[str]:
        spec = self._agents.get(agent)
        return list(spec.allowed_tools) if spec else []
