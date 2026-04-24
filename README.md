# Agentic Browser

Drive a real browser with plain English. The agent reads the page,
picks the next action, executes it, and repeats until done.

FastAPI + Playwright + a three-layer LLM cascade (Groq, Gemini,
local Ollama) so it keeps working when free-tier quotas run out.

## Quick start

Runs locally on your machine. No hosted version.

```bash
git clone https://github.com/harihkk/agentic-browser.git
cd agentic-browser
cp .env.example .env       # add your Groq key
make dev                   # venv + deps + Chromium
make run                   # server on :8000
```

Open `http://localhost:8000` once it's up.

## How it works

Per step the orchestrator:

1. Reads page state (URL, title, visible text, interactive elements)
2. Asks the LLM for one action: navigate / click / type / scroll /
   press_key / extract / select / done
3. Executes it via Playwright, takes a screenshot
4. Repeats until the model says done, or max steps, or a loop is detected

## Provider cascade

Free-tier APIs run out. The agent tries each layer in order:

```
Groq llama-3.3-70b   ->   Gemini 2.0 Flash   ->   local Ollama
   (fast, daily cap)      (15 req/min free)       (no quota)
```

Daily quota errors short-circuit instantly so the agent doesn't burn
30s of retries on errors that won't clear for hours.

## Configuration

`.env`:

```
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=...               # optional
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b         # optional, install Ollama
BROWSER_HEADLESS=true
```

## Layout

```
api/main.py             FastAPI + WebSocket + REST
core/
  ai_agent.py           provider cascade, JSON parsing
  browser_engine.py     Playwright wrapper, CDP launch
  task_orchestrator.py  the agentic loop
  task_templates.py     parameterized presets
  workflow_engine.py    chained tasks with conditions
  scheduler.py          cron-style recurring tasks
  session_recorder.py   export to runnable Playwright Python
  data_extractor.py     CSV / JSON / Markdown export
database/db.py          async SQLite
frontend/index.html     single-page UI
tests/unit/             smoke tests
```

## Tests

```bash
make test
```

## License

MIT
