import asyncio
import json
import traceback
from typing import cast, Any, Literal

from playwright.async_api import async_playwright

from computer_use_demo.tools import ToolCollection, ComputerTool
from computer_use_demo.tools.playwright_tool import PlaywrightTool

from IPython.display import display_markdown
from termcolor import colored


def save_conversation(messages, path: str):
    with open(path, 'w') as f:
        json.dump(messages, f)


def load_conversation(path: str):
    with open(path, 'r') as f:
        messages = json.load(f)
    return messages


async def replay_conversation(messages):
    playwright = await async_playwright().start()

    browser = await playwright.chromium.launch(
        headless=False,
    )
    page = await browser.new_page(
        viewport={"width": 1024, "height": 1024},
    )
    context = page.context

    tools = ToolCollection(
        ComputerTool(page),
        PlaywrightTool(page)
    )

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
                    tool_uses.append({
                        "tool_use_id": content.get("id"),
                        "name": content.get("name"),
                        "input": content.get("input"),
                        "result": result.output or result.error
                    })
                if content.get("type") == "tool_result":
                    tool = next((tool for tool in tool_uses if tool.get("tool_use_id") == content.get("tool_use_id")),
                                None)
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


def format_conversation(messages, format: Literal["markdown", "tty"] = "markdown") -> str:
    result = ""

    def acc(content, *, color=None, on_color=None, attrs=None):
        nonlocal result, format
        if format == "markdown":
            result += content + "\n\n"
        elif format == "tty":
            result += colored(content, color, on_color, attrs) + "\n"

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        on_color: Literal["on_green", "on_red"] = "on_green" if role == "user" else "on_red"
        color: Literal["green", "red"] = "green" if role == "user" else "red"

        role_text = f"🧑 User" if role == "user" \
            else f"🤖 Assistant" if role == "assistant" \
            else role
        if format == "markdown":
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
                # elif c_type == "base64":
                #     prefixed_print(
                #         colored(f"[{content_part.get("media_type")}]: <{content_part.get("data")[:10]}...>",
                #                 "yellow"), prefix)
                elif c_type == "tool_use":
                    name, input = content_part.get("name"), content_part.get("input")
                    if format == "markdown":
                        acc(f"🛠 Tool use:\n- **Name**: `{name}`\n- **Input**: `{input}`")
                    elif format == "tty":
                        acc(
                            f"Using tool `{name}` with input `{input}`",
                            color="cyan"
                        )
                elif c_type == "tool_result":
                    acc(f"Tool result:", color="cyan")
                    format_content(content_part.get("content"), prefix + "  ")

        format_content(content)
        if format == "markdown":
            acc("")
            acc("---")
        elif format == "tty":
            acc("\n")
    return result


def display_conversation_markdown(messages):
    markdown = format_conversation(messages, format="markdown")
    display_markdown(markdown, raw=True)