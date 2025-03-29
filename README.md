# Playwright Computer Use Demo

## Dependencies
- Create venv: `python3 -m venv .venv`
- Activate venv: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Make sure playwright is installed: `playwright install`
- Copy/paste `.env.example` and name it `.env`. Add Anthropic API key to it.

## Experiments:
- Run `python ./example.py`

## Modifications
- Shell and edit tools are commented out, they are not relevant.
- Computer tool is modified to work on Playwright instead of a computer.
- Added a Playwright tool because the AI assistant can't do stuff like click on tabs, click on search bar, etc...
- Modified system prompt to instruct the AI about this different environment.