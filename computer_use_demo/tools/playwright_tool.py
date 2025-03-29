# type: ignore
import json
from datetime import datetime
from typing import cast
from typing import ClassVar
from typing import Literal
from typing import TypedDict
from typing import Union

from playwright.async_api import Page

from ..utils.browser_utils import scrolling_position
from ..utils.browser_utils import take_screenshot
from .base import BaseAnthropicTool
from .base import ToolFailure
from .base import ToolResult


class GotoPageAction(TypedDict):
    type: Literal["goto"]
    url: str


class ZoomPageAction(TypedDict):
    type: Literal["zoom"]
    scale: float


class Action(TypedDict):
    action: Union[GotoPageAction, ZoomPageAction]


class PlaywrightTool(BaseAnthropicTool):
    """
    A tool that allows the agent to do playwright actions
    """

    name: ClassVar[Literal["playwright"]] = "playwright"

    def __init__(self, page: Page):
        self.page = page
        super().__init__()

    async def __call__(self, action: Union[GotoPageAction], **kwargs):
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                return ToolFailure(error="Invalid action json")
        if "type" not in action:
            return ToolFailure(error="Action must have a type")
        if action["type"] == "goto":
            _action = cast(GotoPageAction, action)
            return await self._goto(_action["url"])
        if action["type"] == "zoom":
            _action = cast(ZoomPageAction, action)
            return await self._zoom(_action["scale"])
        return ToolFailure(error=f"Invalid action: {action}. Valid actions: goto, zoom")

    def to_params(self) -> dict:
        return {
            "name": self.name,
            "input_schema": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "oneOf": [
                            {"type": "object", "properties": {"type": {"type": "string", "enum": ["goto"]}, "url": {"type": "string"}}, "required": ["type", "url"]},
                            {
                                "type": "object",
                                "properties": {"type": {"type": "string", "enum": ["zoom"]}, "scale": {"type": "number", "minimum": 0, "description": "The zoom scale to set the page to in percentage. " "100 means original scale."}},
                                "required": ["type", "scale"],
                            },
                        ]
                    }
                },
            },
        }

    async def _goto(self, url: str):
        try:
            await self.page.goto(url)
            return ToolResult(
                system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                output=f"Navigated to page {url}",
                base64_image=await take_screenshot(self.page),
            )
        except Exception as e:
            return ToolFailure(error=f"Failed to navigate to page {url}: {e}")

    async def _zoom(self, scale: float):
        try:
            await self.page.evaluate(f"document.body.style.zoom = '{scale}%'")
            return ToolResult(
                system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                output=f"Zoomed to {scale}%",
                base64_image=await take_screenshot(self.page),
            )
        except Exception as e:
            return ToolFailure(error=f"Failed to zoom to {scale}%: {e}")

    async def scrolling_position(self) -> str:
        return await scrolling_position(self.page)

    def page_url(self) -> str:
        return self.page.url
