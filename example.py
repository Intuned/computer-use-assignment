from computer_use_demo import execute_task_on_website, format_conversation
from dotenv import load_dotenv
import sys
import os
from pydantic import BaseModel
import logging
from datetime import datetime
import uuid

load_dotenv()
import asyncio
from playwright.async_api import async_playwright

class SubmitResultsInput(BaseModel):
    result: str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_results')
    logger.info(f"Results will be saved to {results_dir}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(viewport={"width": 1024, "height": 768})
        result, messages, error = await execute_task_on_website(page=page, task="What is the weather in Tokyo?", submit_results_model=SubmitResultsInput)
        if error:
            logger.error(error)
        else:
            logger.info(result)
        formated_result = format_conversation(messages, format="html")
        
        # Create run_results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename with current date and random uuid
        current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{current_date}_{uuid.uuid4().hex[:8]}.html"
        filepath = os.path.join(results_dir, filename)
        
        # Write results to file
        with open(filepath, 'w') as f:
            f.write(f"Result: {result}\n\n")
            f.write("Conversation:\n")
            f.write(formated_result)
        
        logger.info(f"Results saved to {filepath}")
        
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
