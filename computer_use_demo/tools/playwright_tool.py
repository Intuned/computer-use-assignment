import json
from typing import ClassVar, Literal, TypedDict, Union, cast

from computer_use_demo.tools import ToolResult
from computer_use_demo.tools.base import BaseAnthropicTool, ToolError, ToolFailure

from playwright.async_api import Page

from computer_use_demo.utils.screenshot import take_screenshot


class GotoPageAction(TypedDict):
    type: Literal["goto"]
    url: str


class ScrollPageAction(TypedDict):
    type: Literal["scroll"]
    dx: int
    dy: int


class ZoomPageAction(TypedDict):
    type: Literal["zoom"]
    scale: float


class Action(TypedDict):
    action: Union[GotoPageAction, ScrollPageAction, ZoomPageAction]


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
        if "type" not in action:
            return ToolFailure(error="Action must have a type")
        if action["type"] == "goto":
            _action = cast(GotoPageAction, action)
            return await self._goto(_action["url"])
        if action["type"] == "scroll":
            _action = cast(ScrollPageAction, action)
            return await self._scroll(_action["dx"], _action["dy"])
        if action["type"] == "zoom":
            _action = cast(ZoomPageAction, action)
            return await self._zoom(_action["scale"])
        return ToolFailure(error=f"Invalid action: {action}. Valid actions: goto, scroll, zoom")

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
                                "required": ["type", "dx", "dy"]
                            },

                            {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["zoom"]
                                    },
                                    "scale": {
                                        "type": "number",
                                        "minimum": 0,
                                        "description": "The zoom scale to set the page to in percentage. "
                                                       "100 means original scale."
                                    }
                                },
                                "required": ["type", "scale"]
                            }
                        ]
                    }
                }
            },
        }

    async def _goto(self, url: str):
        try:
            await self.page.goto(url)
            return ToolResult(output=f"Navigated to page {url}", base64_image=await take_screenshot(self.page))
        except Exception as e:
            return ToolFailure(error=f"Failed to navigate to page {url}: {e}")

    async def _scroll(self, dx: int, dy: int):
        try:
            await self.page.mouse.wheel(dx, dy)
            return ToolResult(output=f"Scrolled by {dx} in x and {dy} in y", base64_image=await take_screenshot(self.page))
        except Exception as e:
            return ToolFailure(error=f"Failed to scroll to bottom of page: {e}")

    async def _zoom(self, scale: float):
        try:
            await self.page.evaluate(f"document.body.style.zoom = '{scale}%'")
            return ToolResult(output=f"Zoomed to {scale}%", base64_image=await take_screenshot(self.page))
        except Exception as e:
            return ToolFailure(error=f"Failed to zoom to {scale}%: {e}")
