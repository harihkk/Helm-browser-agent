"""Deterministic, offline test doubles for the browser and AI layers.

These let the executor / E2E tests drive the real orchestrator and the real
validator/blocker/risk pipeline through the exact same code path the live
WebSocket uses - without a real browser, network, or AI provider.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.browser_engine import PageState
from core.intent_planner import IntentPlanner
from core.action_registry import (
    UnsupportedActionError, normalize_action_name, validate_action,
)


def page(url="about:blank", title="", content="", elements=None, error=""):
    return PageState(url, title, content, elements or [], error=error)


class FakeBrowser:
    """A scriptable stand-in for AdvancedBrowserEngine.

    ``routes`` maps a URL substring to the PageState a navigate lands on.
    ``responder`` (optional) overrides per-action behaviour; it receives
    ``(action_type, params, current_state)`` and returns
    ``(result_dict, next_state_or_None)``.
    """

    def __init__(self, routes=None, responder=None, start_state=None):
        self.is_alive = True
        self.routes = routes or {}
        self.responder = responder
        self._state = start_state or page()
        self.state_calls = 0
        self.action_calls = []      # list of (action_type, params)
        self.screens = []           # step numbers screenshotted

    async def restart(self):
        self.is_alive = True

    async def get_page_state(self, context_id="default"):
        self.state_calls += 1
        return self._state

    async def take_screenshot(self, context_id="default", task_id=None, step=None, quality=80):
        self.screens.append(step)
        return "ZmFrZQ=="  # base64("fake")

    async def get_page_diff(self, context_id="default"):
        return {"changed": True, "diff_summary": "+1 -0 lines changed"}

    def _resolve_route(self, url):
        for key, st in self.routes.items():
            if key in url:
                return st
        return page(url=url, title=url, content="")

    async def execute_action(self, context_id, action_type, params):
        action_type = normalize_action_name(action_type)
        params = params or {}
        self.action_calls.append((action_type, dict(params)))

        # Mirror the real engine: reject unsupported/non-executable actions
        # before doing anything.
        try:
            validate_action(action_type, params, executable_only=True)
        except UnsupportedActionError as e:
            return {"success": False, "error": str(e), "fatal": False}

        if self.responder is not None:
            result, next_state = self.responder(action_type, params, self._state)
            if next_state is not None:
                self._state = next_state
            return result

        if action_type == "navigate":
            url = params.get("url", "")
            self._state = self._resolve_route(url)
            if self._state.is_error:
                return {"success": False, "action": "navigate", "url": url,
                        "error": self._state.error}
            return {"success": True, "action": "navigate", "url": url}
        if action_type == "extract":
            return {"success": True, "action": "extract", "data": {
                "url": self._state.url, "title": self._state.title,
                "content": self._state.content[:3000],
                "element_count": len(self._state.elements)}}
        return {"success": True, "action": action_type}


class ScriptedAI:
    """A stand-in for GroqAIAgent that returns a fixed sequence of decisions.

    Each entry of ``steps`` is either an analysis dict or a callable
    ``(state_dict, goal, context) -> analysis dict``. When the script is
    exhausted it returns a harmless ``extract`` so the orchestrator's own
    step/loop limits take over.
    """

    def __init__(self, steps, intent_planner=None, completion=None):
        self.steps = list(steps)
        self._i = 0
        self.intent_planner = intent_planner or IntentPlanner()
        self._completion = completion or {"completed": False, "confidence": 0.0, "summary": ""}
        self.analyze_calls = 0

    async def analyze_page_text(self, state, goal, context):
        self.analyze_calls += 1
        if self._i < len(self.steps):
            step = self.steps[self._i]
            self._i += 1
            return step(state, goal, context) if callable(step) else dict(step)
        return {"action": "extract", "parameters": {"target": "page"},
                "reasoning": "no more scripted steps", "confidence": 0.3,
                "task_complete": False}

    async def check_completion(self, *args, **kwargs):
        return dict(self._completion)

    def get_token_stats(self):
        return {"total_tokens": 0, "total_cost": 0.0, "api_calls": self.analyze_calls}


def drain(orchestrator, description, options=None):
    """Run a task stream to completion and return the list of events."""
    async def go():
        events = []
        async for ev in orchestrator.execute_task_stream(description, options or {}):
            events.append(ev)
        return events
    return asyncio.run(go())


def terminal(events):
    """Return the final terminal event (task_completed / task_failed)."""
    for ev in reversed(events):
        if ev.get("type") in ("task_completed", "task_failed"):
            return ev
    return {}
