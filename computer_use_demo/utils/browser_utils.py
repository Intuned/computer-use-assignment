import base64
import math
from typing import List

from playwright.async_api import Page


async def take_screenshot(page: Page) -> str:
    # await self.page.wait_for_timeout(1000)
    try:
        screenshot = await page.screenshot()
    except:
        screenshot = await page.screenshot(timeout=0)
    # TODO scaling if needed
    screenshot_b64 = base64.b64encode(screenshot).decode()
    return screenshot_b64


async def scrolling_position(page: Page) -> str:
    return await page.evaluate("""() => {
        // Get the current scroll positions
        const scrollX = window.scrollX || window.pageXOffset;
        const scrollY = window.scrollY || window.pageYOffset;

        // Get the viewport size
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        // Get the total document size
        const totalWidth = document.documentElement.scrollWidth;
        const totalHeight = document.documentElement.scrollHeight;

        // Calculate scroll percentages
        const horizontalScrollPercent = ((scrollX / (totalWidth - viewportWidth)) * 100).toFixed(2);
        const verticalScrollPercent = ((scrollY / (totalHeight - viewportHeight)) * 100).toFixed(2);

        // Calculate viewport coverage percentages
        const horizontalCoveragePercent = ((viewportWidth / totalWidth) * 100).toFixed(2);
        const verticalCoveragePercent = ((viewportHeight / totalHeight) * 100).toFixed(2);

        // Determine available scroll options
        const canScrollUp = scrollY > 0;
        const canScrollDown = (scrollY + viewportHeight) < totalHeight;
        const canScrollLeft = scrollX > 0;
        const canScrollRight = (scrollX + viewportWidth) < totalWidth;

        // Calculate total page coverage
        const viewportArea = viewportWidth * viewportHeight;
        const totalPageArea = totalWidth * totalHeight;
        const totalCoveragePercent = ((viewportArea / totalPageArea) * 100).toFixed(2);

        // Create the output string
        let output = `Global scrolling: Currently viewing ${totalCoveragePercent}% of full page size\n\n`;

        if (canScrollUp || canScrollDown || canScrollLeft || canScrollRight) {
            output += `Scroll Options: \n`;
            if (canScrollUp)
                output += `- Up: ${canScrollUp ? 'Available' : 'Not Available'}\n`;
            if (canScrollDown)
                output += `- Down: ${canScrollDown ? 'Available' : 'Not Available'}\n`;
            if (canScrollLeft)
                output += `- Left: ${canScrollLeft ? 'Available' : 'Not Available'}\n`;
            if (canScrollRight)
                output += `- Right: ${canScrollRight ? 'Available' : 'Not Available'}\n`;

            return output;
        }

        return ""
    }""")


async def take_screenshot_with_scroll(page: Page) -> List[str]:
    height = await page.evaluate("""() => {
        return Math.max(
            document.body.scrollHeight,
            document.documentElement.scrollHeight,
            document.body.offsetHeight,
            document.documentElement.offsetHeight
        );
    }""")
    viewport_height = page.viewport_size.get("height")
    print(f"height: {height}, viewport_height: {viewport_height}")
    await page.mouse.wheel(0, -height)
    await page.wait_for_timeout(1000)
    times = math.floor(height / viewport_height)
    screenshot = await take_screenshot(page)
    screenshots: List[str] = [screenshot]
    for _ in range(times):
        await page.mouse.wheel(0, viewport_height)
        await page.wait_for_timeout(1000)
        screenshot = await take_screenshot(page)
        screenshots.append(screenshot)
        # Check if we're at the end of the page (with 100px buffer)
        current_scroll = await page.evaluate("window.scrollY")
        if current_scroll + viewport_height + 100 >= height:
            break
    return screenshots
