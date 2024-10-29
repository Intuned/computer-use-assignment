import json
from typing import ClassVar, Literal, TypedDict, Union

from computer_use_demo.tools import ToolResult
from computer_use_demo.tools.base import BaseAnthropicTool, ToolError

from playwright.async_api import Page


class GotoPageAction(TypedDict):
    type: Literal["goto"]
    url: str


class ScrollPageAction(TypedDict):
    type: Literal["scroll"]
    x: int
    y: int


class Action(TypedDict):
    action: Union[GotoPageAction, ScrollPageAction]


class PlaywrightTool(BaseAnthropicTool):
    """
    A tool that allows the agent to do playwright actions
    """

    name: ClassVar[Literal["playwright"]] = "playwright"

    def __init__(self, page: Page):
        self.page = page
        super().__init__()

    async def __call__(
        self, action: Union[GotoPageAction, ScrollPageAction], **kwargs
    ):
        if isinstance(action, str):
            action = json.loads(action)
        if action["type"] == "goto":
            try:
                await self.page.goto(action["url"])
                return ToolResult(
                    output=f"Navigated to page {action["url"]}",
                    error=None
                )
            except Exception as e:
                return ToolResult(
                    output=None,
                    error=f"Failed to navigate to page {action["url"]}: {e}"
                )
        if action["type"] == "scroll":
            try:
                dx, dy = action["dx"], action["dy"]
                await self.page.mouse.wheel(dx, dy)
                return ToolResult(
                    output=f"Scrolled",
                    error=None
                )
            except Exception as e:
                return ToolResult(
                    output=None,
                    error=f"Failed to scroll to bottom of page: {e}"
                )
        else:
            raise ToolError(f"Invalid action: {action}")

    def to_params(self) -> dict:
        return {
            "name": self.name,
            "input_schema": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["goto"]
                                    },
                                    "url": {
                                        "type": "string"
                                    }
                                },
                                "required": ["type", "url"]
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["scroll"]
                                    },
                                    "dx": {
                                        "type": "integer"
                                    },
                                    "dy": {
                                        "type": "integer"
                                    }
                                },
                                "required": ["type", "x", "y"]
                            }
                        ]
                    }
                }
            },
        }
