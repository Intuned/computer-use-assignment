import asyncio
import os
from datetime import datetime
from enum import StrEnum
from typing import cast
from typing import get_args
from typing import Literal
from typing import TypedDict

from anthropic.types.beta import BetaToolUnionParam
from playwright.async_api import Page

from ..utils.browser_utils import scrolling_position
from ..utils.browser_utils import take_screenshot
from .base import BaseAnthropicTool
from .base import ToolError
from .base import ToolResult

OUTPUT_DIR = "/tmp/outputs"

TYPING_DELAY_MS = 12
TYPING_GROUP_SIZE = 50

Action = Literal[
    "key",
    "type",
    "mouse_move",
    "left_click",
    "left_click_drag",
    "right_click",
    "middle_click",
    "double_click",
    "screenshot",
    "cursor_position",
    "left_mouse_down",
    "left_mouse_up",
    "scroll",
    "hold_key",
    "wait",
    "triple_click",
]

ScrollDirection = Literal["up", "down", "left", "right"]


class Resolution(TypedDict):
    width: int
    height: int


# sizes above XGA/WXGA are not recommended (see README.md)
# scale down to one of these targets if ComputerTool._scaling_enabled is set
MAX_SCALING_TARGETS: dict[str, Resolution] = {
    "XGA": Resolution(width=1024, height=768),  # 4:3
    "WXGA": Resolution(width=1280, height=800),  # 16:10
    "FWXGA": Resolution(width=1366, height=768),  # ~16:9
}


CLICK_BUTTONS = {
    "left_click": 1,
    "right_click": 3,
    "middle_click": 2,
    "double_click": "--repeat 2 --delay 10 1",
    "triple_click": "--repeat 3 --delay 10 1",
}


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20250124"] = "computer_20250124"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(ScalingSource.COMPUTER, self.width, self.height)
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self):
        return cast(
            BetaToolUnionParam,
            {"name": self.name, "type": self.api_type, **self.options},
        )

    def __init__(self, page: Page):
        super().__init__()

        self.page = page
        self.width = int(os.getenv("WIDTH") or 1024)
        self.height = int(os.getenv("HEIGHT") or 768)
        self.mouse_x = 0
        self.mouse_y = 0
        assert self.width and self.height, "WIDTH, HEIGHT must be set"
        if (display_num := os.getenv("DISPLAY_NUM")) is not None:
            self.display_num = int(display_num)
            self._display_prefix = f"DISPLAY=:{self.display_num} "
        else:
            self.display_num = None
            self._display_prefix = ""

        self.xdotool = f"{self._display_prefix}xdotool"

    def map_xdotool_key_to_playwright_key(self, text: str) -> str:
        xdotool_to_playwright_key_map = {
            "A": "KeyA",
            "B": "KeyB",
            "C": "KeyC",
            "D": "KeyD",
            "E": "KeyE",
            "F": "KeyF",
            "G": "KeyG",
            "H": "KeyH",
            "I": "KeyI",
            "J": "KeyJ",
            "K": "KeyK",
            "L": "KeyL",
            "M": "KeyM",
            "N": "KeyN",
            "O": "KeyO",
            "P": "KeyP",
            "Q": "KeyQ",
            "R": "KeyR",
            "S": "KeyS",
            "T": "KeyT",
            "U": "KeyU",
            "V": "KeyV",
            "W": "KeyW",
            "X": "KeyX",
            "Y": "KeyY",
            "Z": "KeyZ",
            "1": "Digit1",
            "2": "Digit2",
            "3": "Digit3",
            "4": "Digit4",
            "5": "Digit5",
            "6": "Digit6",
            "7": "Digit7",
            "8": "Digit8",
            "9": "Digit9",
            "0": "Digit0",
            "Return": "Enter",
            "Escape": "Escape",
            "BackSpace": "Backspace",
            "Tab": "Tab",
            "space": "Space",
            "minus": "Minus",
            "equal": "Equal",
            "bracketleft": "BracketLeft",
            "bracketright": "BracketRight",
            "backslash": "Backslash",
            "semicolon": "Semicolon",
            "apostrophe": "Quote",
            "grave": "Backquote",
            "comma": "Comma",
            "period": "Period",
            "slash": "Slash",
            "Shift_L": "ShiftLeft",
            "Shift_R": "ShiftRight",
            "Control_L": "ControlLeft",
            "Control_R": "ControlRight",
            "Alt_L": "AltLeft",
            "Alt_R": "AltRight",
            "Meta_L": "MetaLeft",
            "Meta_R": "MetaRight",
            # Non-left/right-specific mappings (mapped to left keys)
            "Shift": "ShiftLeft",
            "Control": "ControlLeft",
            "Alt": "AltLeft",
            "Meta": "MetaLeft",
            "Left": "ArrowLeft",
            "Up": "ArrowUp",
            "Right": "ArrowRight",
            "Down": "ArrowDown",
            "Insert": "Insert",
            "Delete": "Delete",
            "Home": "Home",
            "End": "End",
            "Page_Up": "PageUp",
            "Page_Down": "PageDown",
            "Caps_Lock": "CapsLock",
            "Num_Lock": "NumLock",
            "Print": "PrintScreen",
            "Scroll_Lock": "ScrollLock",
            # Add other keys as needed
        }
        return xdotool_to_playwright_key_map.get(text, text)

    async def __call__(
        self,
        *,
        action: Action | str | None = None,
        text: str | None = None,
        coordinate: tuple[int, int] | None = None,
        scroll_direction: ScrollDirection | None = None,
        scroll_amount: int | None = None,
        duration: int | float | None = None,
        **_,
    ):
        if action is None:
            raise ToolError("Computer tool cannot be called with no action")
        if action == "":  # type: ignore
            raise ToolError("Computer tool cannot be called with an empty action")
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")

            x, y = self.validate_and_get_coordinates(coordinate)
            dx = x - self.mouse_x
            dy = y - self.mouse_y
            self.mouse_x = x
            self.mouse_y = y

            if action == "mouse_move":
                await self.page.mouse.move(dx, dy)
                return ToolResult(
                    system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                    output=f"Moved mouse to {self.mouse_x}, {self.mouse_y}",
                    error=None,
                    base64_image=await self.screenshot(),
                )
            elif action == "left_click_drag":
                try:
                    await self.page.mouse.down()
                    await self.page.mouse.move(dx, dy)
                    return ToolResult(
                        system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                        output=f"Dragged mouse to {self.mouse_x}, {self.mouse_y}",
                        error=None,
                        base64_image=await self.screenshot(),
                    )
                finally:
                    await self.page.mouse.up()

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):  # type: ignore
                raise ToolError(message=f"{text} must be a string")

            if action == "key":
                try:
                    await self.page.keyboard.press(self.map_xdotool_key_to_playwright_key(text))
                    return ToolResult(
                        system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                        output=f"Pressed key: {text}",
                        error=None,
                        base64_image=await self.screenshot(),
                    )
                except Exception as e:
                    return ToolResult(output=None, error=f"Failed to press key: {text}: {e}")
            elif action == "type":
                await self.page.keyboard.type(text, delay=TYPING_DELAY_MS)
                return ToolResult(
                    system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                    output=f"Typed: {text}",
                    error=None,
                    base64_image=await self.screenshot(),
                )

        if action in ("left_mouse_down", "left_mouse_up"):
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action=}.")

            await self.page.mouse.down() if action == "left_mouse_down" else await self.page.mouse.up()
            return ToolResult(
                system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                output=f"{'mousedown' if action == 'left_mouse_down' else 'mouseup'} 1",
                error=None,
                base64_image=await self.screenshot(),
            )

        if action == "scroll":
            if scroll_direction is None or scroll_direction not in get_args(ScrollDirection):
                raise ToolError(f"{scroll_direction=} must be 'up', 'down', 'left', or 'right'")
            if not isinstance(scroll_amount, int) or scroll_amount < 0:
                raise ToolError(f"{scroll_amount=} must be a non-negative int")

            if coordinate is not None:
                x, y = self.validate_and_get_coordinates(coordinate)
                # Move mouse to the specified coordinates first
                await self.page.mouse.move(x - self.mouse_x, y - self.mouse_y)
                self.mouse_x, self.mouse_y = x, y

            # Determine scroll direction
            delta_x = 0
            delta_y = 0
            if scroll_direction == "up":
                delta_y = -scroll_amount
            elif scroll_direction == "down":
                delta_y = scroll_amount
            elif scroll_direction == "left":
                delta_x = -scroll_amount
            elif scroll_direction == "right":
                delta_x = scroll_amount

            await self.page.mouse.wheel(delta_x=delta_x, delta_y=delta_y)
            return ToolResult(
                system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                output=f"Scrolled {scroll_amount} {scroll_direction}",
                error=None,
                base64_image=await self.screenshot(),
            )

        if action in ("hold_key", "wait"):
            if duration is None or not isinstance(duration, (int, float)):
                raise ToolError(f"{duration=} must be a number")
            if duration < 0:
                raise ToolError(f"{duration=} must be non-negative")
            if duration > 100:
                raise ToolError(f"{duration=} is too long.")

            if action == "hold_key":
                if text is None:
                    raise ToolError(f"text is required for {action}")

                await self.page.keyboard.press((self.map_xdotool_key_to_playwright_key(text)), delay=duration)
                return ToolResult(
                    system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                    output=f"Held key: {text} for {duration} seconds",
                    error=None,
                    base64_image=await self.screenshot(),
                )

            if action == "wait":
                await self.page.wait_for_timeout(duration)
                return await self.screenshot()

        if action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "screenshot",
            "cursor_position",
            "triple_click",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                self.mouse_x, self.mouse_y = self.validate_and_get_coordinates(coordinate)

            if action == "screenshot":
                return ToolResult(
                    system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                    output="Taken screenshot",
                    error="",
                    base64_image=await self.screenshot(),
                )

            elif action == "cursor_position":
                return ToolResult(
                    output=f"X={self.mouse_x}, Y={self.mouse_y}",
                )
            else:
                x = self.mouse_x
                y = self.mouse_y
                # # Define the rectangle area around the click point (100x100 px)
                # clip = {"x": max(0, x - 50), "y": max(0, y - 50), "width": 100, "height": 100}
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # filename = f"click_area_{action}_{timestamp}.png"
                # # Take and save the screenshot of the area
                # await self.page.screenshot(path=filename, clip=clip)

                # Set up click event listener to capture element's XPath before clicking
                button = action.split("_")[0]

                # perform a fake click to get the xpath
                xpath, _ = await asyncio.gather(get_xpath_of_clicked_element(self.page), _perform_click(self.page, x, y, "left"))

                # perform the actual click
                click_count = await _perform_click(self.page, x, y, button)

                return ToolResult(
                    system=f" | Current url: {self.page_url()} | {await self.scrolling_position()} | Current time: {datetime.now().strftime('%H:%M:%S')}",
                    output=f"Clicked {button} button ({click_count}x) at {self.mouse_x}, {self.mouse_y}. Xpath = {xpath}",
                    error=None,
                    base64_image=await self.screenshot(),
                )

        raise ToolError(f'Invalid action: "{action}"')

    def validate_and_get_coordinates(self, coordinate: tuple[int, int] | None = None):
        if not isinstance(coordinate, list) or len(coordinate) != 2:
            raise ToolError(f"{coordinate} must be a tuple of length 2")
        if not all(isinstance(i, int) and i >= 0 for i in coordinate):
            raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

        return self.scale_coordinates(ScalingSource.API, coordinate[0], coordinate[1])

    async def screenshot(self):
        return await take_screenshot(self.page)

    async def scrolling_position(self) -> str:
        return await scrolling_position(self.page)

    def page_url(self) -> str:
        return self.page.url

    def scale_coordinates(self, source: ScalingSource, x: int, y: int):
        """Scale coordinates to a target maximum resolution."""
        if not self._scaling_enabled:
            return x, y
        ratio = self.width / self.height
        target_dimension = None
        for dimension in MAX_SCALING_TARGETS.values():
            # allow some error in the aspect ratio - not ratios are exactly 16:9
            if abs(dimension["width"] / dimension["height"] - ratio) < 0.02:
                if dimension["width"] < self.width:
                    target_dimension = dimension
                break
        if target_dimension is None:
            return x, y
        # should be less than 1
        x_scaling_factor = target_dimension["width"] / self.width
        y_scaling_factor = target_dimension["height"] / self.height
        if source == ScalingSource.API:
            if x > self.width or y > self.height:
                raise ToolError(f"Coordinates {x}, {y} are out of bounds")
            # scale up
            return round(x / x_scaling_factor), round(y / y_scaling_factor)
        # scale down
        return round(x * x_scaling_factor), round(y * y_scaling_factor)


async def get_xpath_of_clicked_element(page: Page):
    xpath = await page.evaluate("""() => {
                    return new Promise((resolve) => {
                        const clickHandler = function(event) {
                            // Prevent the default action and stop propagation
                            event.preventDefault();
                            event.stopPropagation();
                            
                            // Get the element that was clicked
                            const element = event.target;
                            console.log(element);
                            // Function to get XPath of an element
                            function getElementXPath(element) {
    if (!element || !element.parentNode || element.nodeName === "#document") {
        return null;
    }
    let siblingsCount = 1;
    const parent = element.parentNode;
    const nodeName = element.nodeName.toLowerCase();
    const siblings = Array.from(parent.childNodes).filter((node) => node.nodeType === 1 // Node.ELEMENT_NODE
    );
    for (const sibling of siblings) {
        if (sibling === element) {
            break;
        }
        if (sibling.nodeName.toLowerCase() === nodeName) {
            siblingsCount++;
        }
    }
    const parentXPath = getElementXPath(parent);
    if (element.nodeName === "#text") {
        return parentXPath;
    }
    
    // Special handling for SVG and path elements
    let nodeXPath;
    if (nodeName === 'svg' || nodeName === 'path') {
        nodeXPath = `*[name()='${nodeName}']`;
    } else {
        nodeXPath = `${nodeName}[${siblingsCount}]`;
    }
    
    return parentXPath
        ? `${parentXPath}/${nodeXPath}`
        : nodeXPath;
}
                            
                            // Get XPath of clicked element
                            const xpath = getElementXPath(element);
                            console.log(xpath);
                            // Clean up the event listener
                            document.removeEventListener('click', clickHandler, true);
                            
                            // Return the XPath to Python
                            resolve(xpath);
                            
                        };
                        
                        // Add the event listener with capture phase to get it before it reaches the element
                        document.addEventListener('click', clickHandler, true);
                        
                        // If no click happens within a timeout, resolve with null
                        setTimeout(() => {
                            document.removeEventListener('click', clickHandler, true);
                            resolve(null);
                        }, 5000);
                    });
                }""")
    return xpath


async def _perform_click(page: Page, x: int, y: int, button: str):
    await asyncio.sleep(0.5)
    click_count = 1

    if button == "double":
        click_count = 2

        async def double_click_func():
            await page.mouse.dblclick(x, y)

        await double_click_func()
    elif button == "triple":
        click_count = 3

        async def triple_click_func():
            await page.mouse.click(x, y, click_count=3)

        await triple_click_func()
    else:

        async def click_func():
            await page.mouse.click(x, y, button=button)

        await click_func()

    return click_count
