# Changelog

All notable changes to this project are documented here.
Format loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Universal agent pipeline as first-class, composable modules:
  prompt normalizer, structured intent with command/content separation,
  action registry, risk/confirmation layer, evidence-based validators, and a
  structured blocker schema
- `unverified` terminal state for runs that did work but could not prove
  completion (no more fake "completed")
- Risk/confirmation gate: high-impact actions (cart, checkout, send, submit,
  delete, post, commit, account/financial changes) pause for confirmation;
  pass `options.confirmed` to proceed
- Validators gate completion from visible page evidence; completed runs carry
  validation evidence, blocked/failed/unverified runs carry a structured
  blocker (type, message, suggested next step, evidence)
- Frontend renders validation evidence and structured blocker details, and
  distinguishes blocked / unverified / failed states
- Broad test suite: prompt normalization, command/content separation, intent
  extraction, planner contract, action registry, executor, validators,
  blockers, risk/confirmation, fuzz/paraphrase, and end-to-end WebSocket tests
- `tests/live_check.py` for live HTTP + WebSocket verification against a real
  running server

### Fixed
- Critical: `extract` and `select` actions were silently rewritten to
  non-executable planner concepts by an alias collision and rejected at
  runtime, which stalled tasks on basic pages. Corrected the alias direction,
  guarded executable names, and added an import-time registry audit that fails
  loudly on any future collision
- Intent extraction no longer leaks command scaffolding ("go to", "can you")
  into search queries, and note content is preserved verbatim (interior words
  like "to" are no longer stripped)
- Run lock is created lazily so the orchestrator binds it to the active event
  loop (and can be constructed outside one)

### Removed
- Dead code and unused dependencies: `core/ai_providers.py` (superseded by the
  inline provider cascade), the unused `config/` package (settings never
  imported, prompt templates never loaded), the empty `systems/` package, and
  the `pydantic-settings`, `jinja2`, `aiofiles`, and `pillow` dependencies

## [0.1.0] - 2026-04-22

First working release.

### Added
- FastAPI app with WebSocket task streaming and REST endpoints
- Playwright browser engine with Chromium and CDP-based switching to
  Brave / Chrome / Vivaldi / Edge using a dedicated temp profile
- Three-layer AI cascade: Groq -> Gemini -> local Ollama
- Task orchestrator with the agentic plan-act-evaluate loop
- SQLite persistence for tasks, steps, recordings, templates,
  workflows, scheduled tasks
- Session recorder with Playwright Python and JSON export
- Task templates with variable substitution
- Multi-step workflow engine with conditional branching
- Cron-style task scheduler that survives restarts
- Structured data extractor with CSV / JSON / Markdown output
- Single-page frontend with live preview, history, analytics tabs
- Voice input via Web Speech API
- Smoke test suite (9 tests) covering AI parsing, loop detection,
  retry-after parsing, Python export escaping, DB seeding
- GitHub Actions CI workflow
- Makefile for common dev tasks

### Reliability
- Retry-After header parsing for both Groq and Gemini rate limits
- Daily-quota detection short-circuits retries that would burn time
- Browser auto-restart watchdog if it dies mid-task
- Loop detection: trips on identical repeated actions and idle
  scroll/wait/extract patterns
- `_run_lock` serializes the shared browser across all execution paths
  (tasks, templates, workflows, scheduled jobs, browser switches)
- Type action drills into wrappers when the AI picks a `<form>` or
  `<div>` instead of the inner `<input>`
- Auto-press-Enter after typing into search-like inputs
- Last-failure feedback in the next prompt so the model stops
  retrying broken selectors
