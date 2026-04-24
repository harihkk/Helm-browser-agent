# Changelog

All notable changes to this project are documented here.
Format loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
