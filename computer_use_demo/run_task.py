import asyncio
import os
from typing import Literal
from typing import Optional
from typing import Type
from typing import TypeVar

from anthropic import BadRequestError as AnthropicBadRequestError
from anthropic.types.beta import BetaMessageParam
from playwright.async_api import Page
from pydantic import BaseModel

from .common import create_screenshot_messages
from .loop import APIProvider
from .loop import PROVIDER_TO_DEFAULT_MODEL_NAME
from .loop import sampling_loop
from .tools.base import BaseAnthropicTool
from .utils.conversation_utils import print_diff
from .utils.print_utils import noop  # type: ignore

T = TypeVar("T", bound=BaseModel)


async def execute_task_on_website(*, page: Page, task: str, submit_results_model: Type[T], format: Literal["markdown", "tty"] = "tty", additional_tools: list[BaseAnthropicTool] = []) -> tuple[Optional[T], list[BetaMessageParam], Optional[str]]:
    messages: list[BetaMessageParam] = []

    screenshot_messages = await create_screenshot_messages(page)
    messages.extend(screenshot_messages)

    async def converse():
        result_from_call: Optional[T] = None
        old_messages: list[BetaMessageParam] = [*messages]
        try:
            _, result_from_call = await sampling_loop(
                model=PROVIDER_TO_DEFAULT_MODEL_NAME[APIProvider.ANTHROPIC],
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                provider=APIProvider.ANTHROPIC,
                messages=messages,
                system_prompt_suffix="",
                output_callback=lambda _: print_diff(old_messages, messages, format=format),
                tool_output_callback=lambda _, __: print_diff(old_messages, messages, format=format),
                api_response_callback=noop,  # type: ignore
                page=page,
                submit_results_model=submit_results_model,
                additional_tools=additional_tools,
            )
        except AnthropicBadRequestError:
            import traceback

            traceback.print_exc()

            return None, "failed because of safety reasons"

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(e)
            raise e
        await asyncio.sleep(0.1)

        if result_from_call is not None:
            print("result_from_call", result_from_call)
            return result_from_call, None

        return None, None

    user_message = task
    messages.append({"role": "user", "content": user_message})

    result_from_call, failure_reason = await converse()

    return result_from_call, messages, failure_reason
