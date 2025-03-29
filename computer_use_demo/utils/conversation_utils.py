# type: ignore
import asyncio
import json
import traceback
from typing import Any
from typing import cast
from typing import Iterable
from typing import Literal

from anthropic.types.beta import BetaMessageParam
from playwright.async_api import async_playwright
from termcolor import colored

from computer_use_demo.tools import ComputerTool
from computer_use_demo.tools import PlaywrightTool
from computer_use_demo.tools import ToolCollection


def save_conversation(messages: list[BetaMessageParam], path: str):
    with open(path, "w") as f:
        json.dump(messages, f)


def load_conversation(path: str):
    with open(path) as f:
        messages = json.load(f)
    return messages


async def replay_conversation(messages: list[BetaMessageParam]):
    playwright = await async_playwright().start()

    browser = await playwright.chromium.launch(
        headless=False,
    )
    page = await browser.new_page(
        viewport={"width": 1024, "height": 768},
    )
    context = page.context

    tools = ToolCollection(ComputerTool(page), PlaywrightTool(page))

    tool_uses = []

    try:
        for message in messages:
            if not isinstance(message, dict):
                continue
            if not isinstance(message.get("content"), list):
                continue
            for content in message.get("content"):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "tool_use":
                    result = await tools.run(
                        name=content.get("name"),
                        tool_input=cast(dict[str, Any], content.get("input")),
                    )
                    await asyncio.sleep(0.25)
                    tool_uses.append({"tool_use_id": content.get("id"), "name": content.get("name"), "input": content.get("input"), "result": result.output or result.error})
                if content.get("type") == "tool_result":
                    tool = next((tool for tool in tool_uses if tool.get("tool_use_id") == content.get("tool_use_id")), None)
                    if tool:
                        content = content.get("content")
                        if isinstance(content, list):
                            tool["expected_result"] = []
                            for content_part in content:
                                tool["expected_result"].append({**content_part})
                                if "source" in content_part and "data" in content_part.get("source"):
                                    tool["expected_result"][-1]["source"] = {**content_part.get("source"), "data": ""}
                        else:
                            tool["expected_result"] = content
    except:
        traceback.print_exc()
        print("Error occurred during replay, state of browser may be inconsistent!")

    return playwright, browser, context, page, tools, tool_uses


def format_conversation(messages: list[BetaMessageParam], format: Literal["markdown", "tty", "html"] = "markdown") -> str:
    result = ""

    # Define color mappings at function level
    color_map = {"green": "#2ecc71", "red": "#e74c3c", "yellow": "#f1c40f", "cyan": "#3498db", "black": "#000000"}

    bg_color_map = {"on_green": "#d5f5e3", "on_red": "#fadbd8"}

    def acc(content, *, color=None, on_color=None, attrs=None):
        nonlocal result, format
        if format == "markdown":
            result += content + "\n\n"
        elif format == "tty":
            result += colored(content, color, on_color, attrs) + "\n"
        elif format == "html":
            style = ""
            if color:
                style += f"color: {color_map.get(color, color)};"
            if on_color:
                style += f"background-color: {bg_color_map.get(on_color, on_color)};"
            if attrs and "bold" in attrs:
                style += "font-weight: bold;"

            if style:
                result += f'<div style="{style}">{content}</div>\n'
            else:
                result += f"<div>{content}</div>\n"

    # HTML header if needed
    if format == "html":
        result += """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: system-ui, -apple-system, sans-serif; line-height: 1.5; padding: 20px; }
        .message { margin-bottom: 20px; padding: 15px; border-radius: 8px; }
        img { max-width: 100%; height: auto; }
    </style>
</head>
<body>
"""

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        on_color: Literal["on_green", "on_red"] = "on_green" if role == "user" else "on_red"
        color: Literal["green", "red"] = "green" if role == "user" else "red"

        role_text = "🧑 User" if role == "user" else "🤖 Assistant" if role == "assistant" else role

        if format == "html":
            acc(f'<div class="message" style="background-color: {bg_color_map.get(on_color, "#f8f9fa")};">')
            acc(f'<strong style="color: {color_map.get(color, "#000000")}">{role_text}</strong>')
        elif format == "markdown":
            acc(f"#### {role_text}")
        elif format == "tty":
            acc(f"[{role_text}]:", color="black", on_color=on_color, attrs=["bold"])

        def format_content(content, prefix=""):
            nonlocal acc, format
            if isinstance(content, str):
                acc(content, color=color)
                return

            for content_part in content:
                if isinstance(content_part, str):
                    acc(content_part, color=color)
                    continue
                c_type = content_part.get("type")
                if c_type == "text":
                    acc(content_part.get("text"), color=color)
                elif c_type == "image":
                    data = content_part.get("source").get("data")
                    if format == "markdown":
                        acc(f'<img src="data:image/png;base64,{data}" width="500">')
                    elif format == "tty":
                        acc(f"Image <{data[:10]}...>", color="yellow")
                    elif format == "html":
                        acc(f'<img src="data:image/png;base64,{data}" style="max-width: 500px">')
                elif c_type == "tool_use":
                    name, input = content_part.get("name"), content_part.get("input")
                    if format == "markdown":
                        acc(f"🛠 Tool use:\n- **Name**: `{name}`\n- **Input**: `{input}`")
                    elif format == "tty":
                        acc(f"Using tool `{name}` with input `{input}`", color="cyan")
                    elif format == "html":
                        acc(f'<div style="color: {color_map["cyan"]}">🛠 Tool use:<br>- <strong>Name</strong>: <code>{name}</code><br>- <strong>Input</strong>: <code>{input}</code></div>')
                elif c_type == "tool_result":
                    if format == "html":
                        acc(f'<div style="color: {color_map["cyan"]}">Tool result:</div>')
                    else:
                        acc("Tool result:", color="cyan")
                    format_content(content_part.get("content"), prefix + "  ")

        format_content(content)

        if format == "html":
            acc("</div>")  # Close message div
        elif format == "markdown":
            acc("")
            acc("---")
        elif format == "tty":
            acc("\n")

    # Close HTML document if needed
    if format == "html":
        result += """
</body>
</html>
"""
    return result


def display_conversation_markdown(messages):
    try:
        from IPython.display import display_markdown  # type: ignore

        markdown = format_conversation(messages, format="markdown")
        if markdown.strip() != "":
            display_markdown(markdown, raw=True)
    except ImportError:
        print("IPython is not installed, cannot display conversation as markdown.")


def print_diff(old_messages: list[BetaMessageParam], new_messages: list[BetaMessageParam], format: Literal["markdown", "tty"] = "tty"):
    messages_to_print = [message for message in new_messages if message not in old_messages]

    print(format_conversation(messages_to_print, format=format))

    old_messages.extend(messages_to_print)
