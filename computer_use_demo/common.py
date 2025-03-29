from datetime import datetime

from anthropic.types.beta import BetaMessageParam
from anthropic.types.beta import BetaToolResultBlockParam
from anthropic.types.beta import BetaToolUseBlockParam
from playwright.async_api import Page

from .utils.browser_utils import scrolling_position
from .utils.browser_utils import take_screenshot


async def create_screenshot_messages(page: Page) -> list[BetaMessageParam]:
    """Create screenshot tool use and result messages."""
    # any hardcoded tool id should work fine.
    tool_use_id = "toolu_013wXXEckrdnFkkw222hw2Dz"
    tool_use_message_block = BetaToolUseBlockParam(type="tool_use", id=tool_use_id, name="computer", input={"action": "screenshot"})

    tool_use_message = BetaMessageParam(content=[tool_use_message_block], role="assistant")

    screenshot_b64 = await take_screenshot(page)

    tool_result_message_block = BetaToolResultBlockParam(
        type="tool_result",
        tool_use_id=tool_use_id,
        content=[
            {"type": "text", "text": f" | Current URL: {page.url} | {await scrolling_position(page)} | Timestamp: {datetime.now().strftime('%H:%M:%S')}"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64}},
        ],
    )

    tool_result_message = BetaMessageParam(content=[tool_result_message_block], role="user")

    return [tool_use_message, tool_result_message]
