"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any
from typing import cast
from typing import Type
from typing import TypeVar

import httpx
from anthropic import Anthropic
from anthropic import AnthropicBedrock
from anthropic import AnthropicVertex
from anthropic import APIError
from anthropic import APIResponseValidationError
from anthropic import APIStatusError
from anthropic.types.beta import BetaCacheControlEphemeralParam
from anthropic.types.beta import BetaContentBlockParam
from anthropic.types.beta import BetaImageBlockParam
from anthropic.types.beta import BetaMessage
from anthropic.types.beta import BetaMessageParam
from anthropic.types.beta import BetaTextBlock
from anthropic.types.beta import BetaTextBlockParam
from anthropic.types.beta import BetaToolResultBlockParam
from anthropic.types.beta import BetaToolUseBlockParam
from playwright.async_api import Page
from pydantic import BaseModel

from computer_use_demo.tools import BaseAnthropicTool
from computer_use_demo.tools import ToolCollection
from computer_use_demo.tools import ToolResult
from computer_use_demo.tools.computer import ComputerTool
from computer_use_demo.tools.playwright_tool import PlaywrightTool
from computer_use_demo.tools.submit_results_tool import SubmitResultsTool
from computer_use_demo.tools.submit_results_tool import TerminateLoopToolResult

COMPUTER_USE_BETA_FLAG = "computer-use-2025-01-24"
PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


PROVIDER_TO_DEFAULT_MODEL_NAME: dict[APIProvider, str] = {
    APIProvider.ANTHROPIC: "claude-3-7-sonnet-20250219",
    APIProvider.BEDROCK: "anthropic.claude-3-7-sonnet-20250219:0",
    APIProvider.VERTEX: "claude-3-7-sonnet@20250219",
}

SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising a browser.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* You have a browser tool that you use to perform specialized actions on the browser. You use it to:
   1. navigate to a webpage. call it with action `goto` with the URL.
   2. scroll the page. call it with action `scroll` with the dy and dx. Positive values scroll down and right. Negative values scroll up and left.
   For any other actions, you use the computer tool.
* with every screenshot, you will be provided with current url, scrolling positions and possible scroll options.
* You are given the screenshot of the current view port, in many cases, what you need to do requires you scroll to see more. Keep scrolling until you are sure that there is nothing more to see. Don't give up until you are 100% confident the task is not possible.
* In some cases, the scrollbar appears on a certin section of the screen (localized scroll), if that is the case, you need to focus the area of the screen where the scrollbar appears and then scroll.
* In some cases, you will get global scrolling options, for those, you can scroll without focusing.
* The computer tool with "click" action has additional functionality - it provides the xpath to the element clicked.
* When asked to provide the xpath, only use the xpath obtained from the computer tool with click action.
* If you receive a screenshot that looks malformed or empty when it should not be (for example, empty screenshot after a scroll), try to take another screenshot.
* You never call the tools with empty actions.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>
"""

SYSTEM_PROMPT_VALIDATION_SUFFIX = """
You will be given a tool called execute_function_to_validate.
You will also will be given what to validate.
Based on the current state of the page and what to validate - come up with a plan of validation.
The plan for validation should include:
- what you need to do (scroll, click, etc.) before executing the validation function. if you are trying to validate something not on the screen, find it before executing the validation function.
- what you need to after executing the validation function.
- how is the validation going to work?
Then - call the execute_function_to_validate tool.
After the tool execution, validate that the function execution worked as exected.
When you have the answer, submit it using the submit_results tool.
"""

SYSTEM_PROMPT_SUBMIT_RESULTS_SUFFIX = """
after executing the USER_TASK, you should submit the result using the submit_results tool.

"""


T = TypeVar("T", bound=BaseModel)


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[[httpx.Request, httpx.Response | object | None, Exception | None], None],
    api_key: str,
    page: Page,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    thinking_budget: int | None = None,
    token_efficient_tools_beta: bool = False,
    submit_results_model: Type[T] | None = None,
    function_to_validate: Callable[[Page], Any] | None = None,
    additional_tools: list[BaseAnthropicTool] = [],
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    system_promot = SYSTEM_PROMPT

    tool_collection = ToolCollection(
        ComputerTool(page),
        PlaywrightTool(page),
    )

    if function_to_validate and submit_results_model:
        tool_collection.add_tool(SubmitResultsTool(submit_results_model))
        system_promot = system_promot + SYSTEM_PROMPT_VALIDATION_SUFFIX
    elif submit_results_model:
        tool_collection.add_tool(SubmitResultsTool(submit_results_model))
        system_promot = system_promot + SYSTEM_PROMPT_SUBMIT_RESULTS_SUFFIX
    if additional_tools:
        for tool in additional_tools:
            tool_collection.add_tool(tool)

    system = BetaTextBlockParam(
        type="text",
        text=f"{system_promot} {system_prompt_suffix if system_prompt_suffix else ''}",
    )

    while True:
        response_params = await completion(
            model=model,
            provider=provider,
            messages=messages,
            api_response_callback=api_response_callback,
            system=system,
            tool_collection=tool_collection,
            api_key=api_key,
            only_n_most_recent_images=only_n_most_recent_images,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
            token_efficient_tools_beta=token_efficient_tools_beta,
        )
        messages.append(
            {
                "role": "assistant",
                "content": response_params,
            }
        )

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in response_params:
            output_callback(content_block)
            if content_block["type"] == "tool_use":
                result = await tool_collection.run(
                    name=content_block["name"],
                    tool_input=cast(dict[str, Any], content_block["input"]),
                )

                if isinstance(result, TerminateLoopToolResult):
                    return messages, result.data

                tool_result_content.append(_make_api_tool_result(result, content_block["id"]))
                tool_output_callback(result, content_block["id"])

        if not tool_result_content:
            messages.append({"content": "Please execute the user task. If you are done with it - submit the results using the submit_results tool.", "role": "user"})

        messages.append({"content": tool_result_content, "role": "user"})


async def completion(
    *,
    model: str,
    provider: APIProvider,
    messages: list[BetaMessageParam],
    api_response_callback: Callable[[httpx.Request, httpx.Response | object | None, Exception | None], None],
    system: BetaTextBlockParam | None = None,
    tool_collection: ToolCollection,
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    thinking_budget: int | None = None,
    token_efficient_tools_beta: bool = False,
):
    enable_prompt_caching = True
    betas = [COMPUTER_USE_BETA_FLAG]

    if token_efficient_tools_beta:
        betas.append("token-efficient-tools-2025-02-19")

    image_truncation_threshold = 10
    system = system or BetaTextBlockParam(
        type="text",
        text=SYSTEM_PROMPT,
    )

    if provider == APIProvider.ANTHROPIC:
        client = Anthropic(api_key=api_key)
        enable_prompt_caching = True
    elif provider == APIProvider.VERTEX:
        client = AnthropicVertex()
    elif provider == APIProvider.BEDROCK:
        client = AnthropicBedrock(
            aws_region="us-west-2",
            aws_profile="default",
        )

    if enable_prompt_caching:
        betas.append(PROMPT_CACHING_BETA_FLAG)
        _inject_prompt_caching(messages)
        # Is it ever worth it to bust the cache with prompt caching?
        image_truncation_threshold = 50
        system["cache_control"] = {"type": "ephemeral"}

    if only_n_most_recent_images:
        _maybe_filter_to_n_most_recent_images(
            messages,
            only_n_most_recent_images,
            min_removal_threshold=image_truncation_threshold,
        )

    extra_body = {}
    if thinking_budget:
        # Ensure we only send the required fields for thinking
        extra_body = {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}

    # Call the API
    # we use raw_response to provide debug information to streamlit. Your
    # implementation may be able call the SDK directly with:
    # `response = client.messages.create(...)` instead.
    try:
        raw_response = client.beta.messages.with_raw_response.create(
            max_tokens=max_tokens,
            messages=messages,
            model=model,
            system=[system],
            tools=tool_collection.to_params(),
            betas=betas,
            extra_body=extra_body,
        )
    except (APIStatusError, APIResponseValidationError) as e:
        api_response_callback(e.request, e.response, e)
        raise e
        # return messages
    except APIError as e:
        api_response_callback(e.request, e.body, e)
        raise e
        # return messages

    api_response_callback(raw_response.http_response.request, raw_response.http_response, None)

    response = raw_response.parse()

    return _response_to_params(response)


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:  # type: ignore
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [item for message in messages for item in (message["content"] if isinstance(message["content"], list) else []) if isinstance(item, dict) and item.get("type") == "tool_result"],  # type: ignore
    )

    total_images = sum(1 for tool_result in tool_result_blocks for content in tool_result.get("content", []) if isinstance(content, dict) and content.get("type") == "image")

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaContentBlockParam]:
    res: list[BetaContentBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                res.append(BetaTextBlockParam(type="text", text=block.text))
            elif getattr(block, "type", None) == "thinking":
                # Handle thinking blocks - include signature field
                thinking_block = {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None),
                }
                if hasattr(block, "signature"):
                    thinking_block["signature"] = getattr(block, "signature", None)
                res.append(cast(BetaContentBlockParam, thinking_block))
        else:
            # Handle tool use blocks normally
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(content := message["content"], list):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                if content[-1] is not None:
                    # Use type ignore to bypass TypedDict check until SDK types are updated
                    content[-1]["cache_control"] = BetaCacheControlEphemeralParam(  # type: ignore
                        {"type": "ephemeral"}
                    )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(result: ToolResult, tool_use_id: str) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
