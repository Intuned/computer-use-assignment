"""Collection classes for managing multiple tools."""

from typing import Any

from anthropic.types.beta import BetaToolUnionParam

from .base import BaseAnthropicTool
from .base import ToolError
from .base import ToolFailure
from .base import ToolResult


class ToolCollection:
    """A collection of anthropic-defined tools."""

    def __init__(self, *tools: BaseAnthropicTool):
        self.tools = tools
        self.tool_map = {tool.to_params()["name"]: tool for tool in tools}

    def add_tool(self, tool: BaseAnthropicTool):
        self.tools = (*self.tools, tool)
        self.tool_map[tool.to_params()["name"]] = tool

    def to_params(
        self,
    ) -> list[BetaToolUnionParam]:
        return [tool.to_params() for tool in self.tools]

    async def run(self, *, name: str, tool_input: dict[str, Any]) -> ToolResult:
        tool = self.tool_map.get(name)
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid")
        try:
            return await tool(**tool_input)
        except ToolError as e:
            return ToolFailure(error=e.message)
