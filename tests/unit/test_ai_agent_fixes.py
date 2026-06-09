"""Regression tests for ai_agent fixes: honest fallback and ambiguous-task handling.

These pin the two behavioural changes:
  * `_fallback_analysis` must never claim completion when it could not decide
    the next action (it extracts evidence instead).
  * an undecidable success condition surfaces as a clean structured signal,
    not an exception, and the orchestrator turns it into an
    `ambiguous_instruction` blocker.
"""

import asyncio
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.ai_agent import GroqAIAgent, ANALYSIS_SYSTEM_PROMPT
from core.intent_planner import MissingSuccessConditionError
from core.task_orchestrator import SophisticatedTaskOrchestrator
from tests.unit.fakes import FakeBrowser, ScriptedAI, drain, terminal, page


def _offline_agent():
    """A GroqAIAgent wired only to a (never-called) local Ollama, so the
    constructor's provider guard passes without any network."""
    return GroqAIAgent(api_key="", ollama_url="http://localhost:11434",
                       ollama_model="dummy")


class HonestFallback(unittest.TestCase):
    def test_fallback_does_not_falsely_claim_completion(self):
        agent = _offline_agent()
        # Mid-run, no usable input element, not an idle pattern: the fallback
        # genuinely cannot decide the next action.
        state = {"url": "https://example.com/page", "title": "Page",
                 "content": "some text", "elements": []}
        ctx = {"action_history": [{"action": "click", "success": True}]}
        result = agent._fallback_analysis("do an unclear thing", state, ctx)
        self.assertFalse(result["task_complete"],
                         "fallback must not mark an undecided task complete")
        self.assertEqual(result["action"], "extract")

    def test_fallback_still_navigates_when_starting_fresh(self):
        # Unchanged behaviour: with no history it should navigate to a target.
        agent = _offline_agent()
        result = agent._fallback_analysis("go to wikipedia", {"url": "about:blank"}, {})
        self.assertEqual(result["action"], "navigate")
        self.assertFalse(result["task_complete"])


class AmbiguousInstruction(unittest.TestCase):
    def test_analyze_returns_clean_error_instead_of_raising(self):
        agent = _offline_agent()
        # The deterministic-planner fallback (no AI configured) is where an
        # undecidable success condition is turned into a clean signal rather
        # than an exception. Force that path.
        agent.client = None
        agent._gemini_key = ""
        agent._ollama_url = ""
        agent._ollama_model = ""

        def boom(*args, **kwargs):
            raise MissingSuccessConditionError("no success condition")

        agent.intent_planner.plan = boom  # force the undecidable path
        result = asyncio.run(agent.analyze_page_text(
            {"url": "about:blank", "elements": []}, "something vague", {}))
        self.assertEqual(result.get("error"), "ambiguous_instruction")
        self.assertFalse(result.get("task_complete"))

    def test_orchestrator_maps_ambiguous_error_to_blocker(self):
        browser = FakeBrowser(routes={
            "wikipedia.org": page("https://en.wikipedia.org/wiki/Python",
                                   "Python", "Python programming language"),
        })
        # A concrete goal so the upfront success-condition guard passes, then
        # the AI reports the task is ambiguous on the first analysis.
        ai = ScriptedAI([
            {"error": "ambiguous_instruction", "message": "cannot decide",
             "task_complete": False},
        ])
        events = drain(orch_with(browser, ai),
                       "go to wikipedia and read about python", {"max_steps": 4})
        end = terminal(events)
        self.assertEqual(end["type"], "task_failed")
        self.assertEqual(end["status"], "blocked")
        self.assertEqual(end["blocker_type"], "ambiguous_instruction")


class SystemPromptWiring(unittest.TestCase):
    def test_analysis_passes_static_system_prompt_and_lean_user_message(self):
        agent = _offline_agent()
        captured = {}

        async def fake_call_groq(prompt, model=None, system=None, **kw):
            captured["prompt"] = prompt
            captured["system"] = system
            return '{"action": "extract", "parameters": {}, "task_complete": false}'

        # Force the LLM path (skip the deterministic planner) and capture args.
        agent._quick_action = lambda *a, **k: None
        agent._call_groq = fake_call_groq

        result = asyncio.run(agent.analyze_page_text(
            {"url": "https://example.com", "title": "Example",
             "content": "hello world", "elements": []},
            "do a weird unroutable thing", {}))

        # Static principles travel in the system message...
        self.assertEqual(captured["system"], ANALYSIS_SYSTEM_PROMPT)
        # ...and the per-step user message carries only the dynamic context,
        # not the per-site routing tables (those belong to the planner).
        self.assertIn("GOAL: do a weird unroutable thing", captured["prompt"])
        self.assertIn("https://example.com", captured["prompt"])
        self.assertNotIn("github", captured["prompt"].lower())
        self.assertNotIn("amazon", captured["prompt"].lower())
        self.assertEqual(result["action"], "extract")


def orch_with(browser, ai):
    return SophisticatedTaskOrchestrator(browser, ai)


if __name__ == "__main__":
    unittest.main()
