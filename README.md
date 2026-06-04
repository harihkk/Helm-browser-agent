# Helm

Drive a real browser with plain English. Helm turns a natural-language
instruction into a structured intent, plans one safe browser action at a time,
executes it, observes the page, and only reports the task done when a validator
can prove it from the visible page.

FastAPI + Playwright + a three-layer LLM cascade (Groq, Gemini, local Ollama)
so it keeps working when free-tier quotas run out.

## Quick start

Runs locally on your machine. No hosted version.

```bash
git clone https://github.com/harihkk/Helm-agentic-browser.git
cd Helm-agentic-browser
cp .env.example .env       # add Groq, Gemini, or Ollama settings
make dev                   # venv + deps + Chromium
make run                   # server on :8000
```

Open `http://localhost:8000` once it's up.

## How it works

Every task runs through one pipeline, never a raw prompt to the browser:

```
raw prompt
  -> normalize (typos, slang, filler, URLs, quoted content)
  -> structured intent (command vs content, target site/URL, query, content,
       constraints, success condition, risk level, validation strategy)
  -> risk check (high-impact actions pause for confirmation)
  -> plan ONE registered action from the intent + the current page
  -> action-registry validation (the engine can only run known actions)
  -> execute via Playwright, then observe the new page state
  -> adaptive recovery / loop detection
  -> validator gates completion
       completed  : proven from visible page evidence
       unverified : did work but could not prove the outcome
       blocked    : a precise, structured blocker (login, captcha, 404, ...)
```

A deterministic planner handles high-confidence routes (direct URLs, site
searches, common destinations); the LLM decides the next action only when no
high-confidence route applies. Either way the same registry, observation,
validator, and blocker layers apply.

The agent never types the raw instruction into a search box, never marks a task
done just because a page loaded, and never invents an action the engine cannot
execute.

## Safety

High-impact actions pause and ask before acting: purchases, checkout, sending
messages or emails, submitting forms, deleting content, posting publicly, and
account or financial changes. Reading, searching, navigating, playing media,
and writing into a note you explicitly asked for run without a prompt. Walls
that cannot be automated (sign-in, captcha, 404, no results) are reported as
structured blockers, not faked completions.

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
  prompt_normalizer.py  stage 1: normalize messy prompts, structural signals
  intent_planner.py     structured intent + command/content separation + planner
  action_registry.py    the only browser actions the engine will execute
  risk.py               risk classification + confirmation gate
  validators.py         composable, evidence-based completion validators
  blockers.py           structured blocker schema and classifier
  task_orchestrator.py  the agentic loop (plan, execute, observe, validate)
  ai_agent.py           provider cascade, JSON parsing
  browser_engine.py     Playwright wrapper, CDP launch
  task_templates.py     parameterized presets
  workflow_engine.py    chained tasks with conditions
  scheduler.py          cron-style recurring tasks
  session_recorder.py   export to runnable Playwright Python
  data_extractor.py     CSV / JSON / Markdown export
database/db.py          async SQLite
frontend/index.html     single-page UI (run-id safe, shows validation/blockers)
tests/unit/             unit, command/content, validator, blocker, risk,
                        executor, fuzz/paraphrase, and end-to-end WebSocket tests
tests/live_check.py     live HTTP + WebSocket check against a running server
```

## Tests

```bash
make test          # full offline suite (unit + integration + E2E + fuzz)
make lint          # syntax check across the codebase
```

`tests/live_check.py` boots the real server and drives the real `/ws/advanced`
WebSocket end to end with a deterministic fake browser, so the pipeline can be
verified without an AI key or live websites.

## License

MIT
