import base64

from playwright.async_api import Page


async def take_screenshot(page: Page) -> str:
    # await self.page.wait_for_timeout(1000)
    try:
        screenshot = await page.screenshot()
    except:
        screenshot = await page.screenshot(
            timeout=0
        )
    # TODO scaling if needed
    screenshot_b64 = base64.b64encode(screenshot).decode()
    return screenshot_b64
