import asyncio
import base64
import os
import shutil
from enum import StrEnum
from pathlib import Path
from typing import Literal, TypedDict
from uuid import uuid4

from anthropic.types.beta import BetaToolComputerUse20241022Param

from .base import BaseAnthropicTool, ToolError, ToolResult

from playwright.async_api import Playwright, Page

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
]


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


class ScalingSource(StrEnum):
    COMPUTER = "computer"
    API = "api"


class ComputerToolOptions(TypedDict):
    display_height_px: int
    display_width_px: int
    display_number: int | None


def chunks(s: str, chunk_size: int) -> list[str]:
    return [s[i: i + chunk_size] for i in range(0, len(s), chunk_size)]


class ComputerTool(BaseAnthropicTool):
    """
    A tool that allows the agent to interact with the screen, keyboard, and mouse of the current computer.
    The tool parameters are defined by Anthropic and are not editable.
    """

    name: Literal["computer"] = "computer"
    api_type: Literal["computer_20241022"] = "computer_20241022"
    width: int
    height: int
    display_num: int | None

    _screenshot_delay = 2.0
    _scaling_enabled = True

    @property
    def options(self) -> ComputerToolOptions:
        width, height = self.scale_coordinates(
            ScalingSource.COMPUTER, self.width, self.height
        )
        return {
            "display_width_px": width,
            "display_height_px": height,
            "display_number": self.display_num,
        }

    def to_params(self) -> BetaToolComputerUse20241022Param:
        return {"name": self.name, "type": self.api_type, **self.options}

    def __init__(self, page: Page):
        super().__init__()

        self.page = page
        self.width = int(os.getenv("WIDTH") or 0)
        self.height = int(os.getenv("HEIGHT") or 0)
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

    def map_xdotool_key_to_playwright_key(self, text):
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
            action: Action | None = None,
            text: str | None = None,
            coordinate: tuple[int, int] | None = None,
            **kwargs,
    ):
        if action is None:
            raise ToolError("Computer tool cannot be called with no action")
        if action == '':
            raise ToolError("Computer tool cannot be called with an empty action")
        if action in ("mouse_move", "left_click_drag"):
            if coordinate is None:
                raise ToolError(f"coordinate is required for {action}")
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if not isinstance(coordinate, list) or len(coordinate) != 2:
                raise ToolError(f"{coordinate} must be a tuple of length 2")
            if not all(isinstance(i, int) and i >= 0 for i in coordinate):
                raise ToolError(f"{coordinate} must be a tuple of non-negative ints")

            x, y = self.scale_coordinates(
                ScalingSource.API, coordinate[0], coordinate[1]
            )
            dx = x - self.mouse_x
            dy = y - self.mouse_y
            self.mouse_x = x
            self.mouse_y = y

            if action == "mouse_move":
                await self.page.mouse.move(dx, dy)
                return ToolResult(
                    output=f"Moved mouse to {self.mouse_x}, {self.mouse_y}",
                    error=None,
                    base64_image=(await self.screenshot())[1]
                )
            # return await self.shell(f"{self.xdotool} mousemove --sync {x} {y}")
            elif action == "left_click_drag":
                try:
                    await self.page.mouse.down()
                    await self.page.mouse.move(dx, dy)
                    return ToolResult(
                        output=f"Dragged mouse to {self.mouse_x}, {self.mouse_y}",
                        error=None,
                        base64_image=(await self.screenshot())[1]
                    )
                finally:
                    await self.page.mouse.up()

        if action in ("key", "type"):
            if text is None:
                raise ToolError(f"text is required for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")
            if not isinstance(text, str):
                raise ToolError(message=f"{text} must be a string")

            if action == "key":
                try:
                    await self.page.keyboard.press(self.map_xdotool_key_to_playwright_key(text))
                    return ToolResult(
                        output=f"Pressed key: {text}",
                        error=None,
                        base64_image=(await self.screenshot())[1]
                    )
                except Exception as e:
                    return ToolResult(
                        output=None,
                        error=f"Failed to press key: {text}: {e}"
                    )
                # return await self.shell(f"{self.xdotool} key -- {text}")
            elif action == "type":
                await self.page.keyboard.type(text, delay=TYPING_DELAY_MS)
                return ToolResult(
                    output=f"Typed: {text}",
                    error=None,
                    base64_image=(await self.screenshot())[1]
                )
                # results: list[ToolResult] = []
                # for chunk in chunks(text, TYPING_GROUP_SIZE):
                #     cmd = f"{self.xdotool} type --delay {TYPING_DELAY_MS} -- {shlex.quote(chunk)}"
                #     results.append(await self.shell(cmd, take_screenshot=False))
                # screenshot_base64 = (await self.screenshot()).base64_image
                # return ToolResult(
                #     output="".join(result.output or "" for result in results),
                #     error="".join(result.error or "" for result in results),
                #     base64_image=screenshot_base64,
                # )

        if action in (
                "left_click",
                "right_click",
                "double_click",
                "middle_click",
                "screenshot",
                "cursor_position",
        ):
            if text is not None:
                raise ToolError(f"text is not accepted for {action}")
            if coordinate is not None:
                raise ToolError(f"coordinate is not accepted for {action}")

            if action == "screenshot":
                return (await self.screenshot())[0]
            elif action == "cursor_position":
                return ToolResult(
                    output=f"X={self.mouse_x}, Y={self.mouse_y}",
                )
                # result = await self.shell(
                #     f"{self.xdotool} getmouselocation --shell",
                #     take_screenshot=False,
                # )
                # output = result.output or ""
                # x, y = self.scale_coordinates(
                #     ScalingSource.COMPUTER,
                #     int(output.split("X=")[1].split("\n")[0]),
                #     int(output.split("Y=")[1].split("\n")[0]),
                # )
                # return result.replace(output=f"X={x},Y={y}")
            else:
                button = action.split("_")[0]
                if button == "double":
                    await self.page.mouse.click(
                        self.mouse_x, self.mouse_y
                    )
                    await self.page.wait_for_timeout(500)
                    button = "left"
                await self.page.mouse.click(
                    self.mouse_x, self.mouse_y, button=button
                )
                return ToolResult(
                    output=f"Clicked {button} button at {self.mouse_x}, {self.mouse_y}",
                    error=None,
                    base64_image=(await self.screenshot())[1]
                )
                # click_arg = {
                #     "left_click": "1",
                #     "right_click": "3",
                #     "middle_click": "2",
                #     "double_click": "--repeat 2 --delay 500 1",
                # }[action]
                # return await self.shell(f"{self.xdotool} click {click_arg}")

        raise ToolError(f"Invalid action: \"{action}\"")

    async def screenshot(self):
        screenshot = await self.page.screenshot(
            timeout=0
        )
        # TODO scaling if needed
        screenshot_b64 = base64.b64encode(screenshot).decode()
        return ToolResult(
            output="Taken screenshot",
            error="",
            base64_image=screenshot_b64,
        ), screenshot_b64
        # """Take a screenshot of the current screen and return the base64 encoded image."""
        # output_dir = Path(OUTPUT_DIR)
        # output_dir.mkdir(parents=True, exist_ok=True)
        # path = output_dir / f"screenshot_{uuid4().hex}.png"
        #
        # # Try gnome-screenshot first
        # if shutil.which("gnome-screenshot"):
        #     screenshot_cmd = f"{self._display_prefix}gnome-screenshot -f {path} -p"
        # else:
        #     # Fall back to scrot if gnome-screenshot isn't available
        #     screenshot_cmd = f"{self._display_prefix}scrot -p {path}"
        #
        # result = await self.shell(screenshot_cmd, take_screenshot=False)
        # if self._scaling_enabled:
        #     x, y = self.scale_coordinates(
        #         ScalingSource.COMPUTER, self.width, self.height
        #     )
        #     await self.shell(
        #         f"convert {path} -resize {x}x{y}! {path}", take_screenshot=False
        #     )
        #
        # if path.exists():
        #     return result.replace(
        #         base64_image=base64.b64encode(path.read_bytes()).decode()
        #     )
        # raise ToolError(f"Failed to take screenshot: {result.error}")

    # async def shell(self, command: str, take_screenshot=True) -> ToolResult:
    #     """Run a shell command and return the output, error, and optionally a screenshot."""
    #     _, stdout, stderr = await run(command)
    #     base64_image = None
    #
    #     if take_screenshot:
    #         # delay to let things settle before taking a screenshot
    #         await asyncio.sleep(self._screenshot_delay)
    #         base64_image = (await self.screenshot()).base64_image
    #
    #     return ToolResult(output=stdout, error=stderr, base64_image=base64_image)

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
