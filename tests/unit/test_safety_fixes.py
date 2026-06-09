"""Regression tests for the integrity/safety fixes:

  * the every-5-steps completion check is gated by the evidence validator
    (no weaker second route to COMPLETED),
  * completion validation uses the planner's intent, never the model-reported
    one (a model cannot fabricate an easy success condition),
  * the browser engine refuses to type into password fields.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import core.validators as V
from core.browser_engine import _is_password_field
from core.task_orchestrator import SophisticatedTaskOrchestrator, AdvancedTask
from tests.unit.fakes import FakeBrowser, ScriptedAI, drain, terminal, page


def orch(browser, ai):
    return SophisticatedTaskOrchestrator(browser, ai)


class PeriodicCompletionIsValidated(unittest.TestCase):
    def test_unproven_periodic_completion_is_ignored(self):
        # The cheap completion model insists the task is done, but the page
        # never shows the requested figure. The validator must override it.
        browser = FakeBrowser(routes={
            "site.com": page("https://site.com", "Site", "some unrelated content"),
        })
        ai = ScriptedAI(
            [{"action": "navigate", "parameters": {"url": "https://site.com"}}],
            completion={"completed": True, "confidence": 0.97, "summary": "fake done"},
        )
        events = drain(orch(browser, ai),
                       "find the quarterly revenue figure on site.com",
                       {"max_steps": 6})
        end = terminal(events)
        self.assertNotEqual(end["status"], "completed")
        self.assertIn(end["status"], ("unverified", "blocked", "failed"))


class ValidationUsesPlannerIntent(unittest.TestCase):
    def test_model_reported_intent_cannot_override_planner_intent(self):
        o = orch(FakeBrowser(), ScriptedAI([]))
        task = AdvancedTask("t1", "desc", {})
        task.intent = {"success_condition": "planner truth",
                       "validation_strategy": "validate_text_visible",
                       "search_query": "planner truth"}

        captured = {}
        original = V.validate_completion

        def spy(intent, state, history, extracted):
            captured["intent"] = intent
            return original(intent, state, history, extracted)

        V.validate_completion = spy
        try:
            o._validation_result(
                task, page("https://x.com", "X", "irrelevant"),
                {"intent": {"success_condition": "model lie",
                            "validation_strategy": "validate_text_visible"}})
        finally:
            V.validate_completion = original

        self.assertEqual(captured["intent"], task.intent)
        self.assertNotEqual(captured["intent"].get("success_condition"), "model lie")


class PasswordFieldRefusal(unittest.TestCase):
    def test_password_type_is_detected(self):
        self.assertTrue(_is_password_field("password"))
        self.assertTrue(_is_password_field("PASSWORD"))

    def test_password_autocomplete_is_detected(self):
        self.assertTrue(_is_password_field("text", "current-password"))
        self.assertTrue(_is_password_field("", "new-password"))

    def test_ordinary_fields_are_allowed(self):
        self.assertFalse(_is_password_field("text"))
        self.assertFalse(_is_password_field("search"))
        self.assertFalse(_is_password_field("email", "email"))
        self.assertFalse(_is_password_field(""))


if __name__ == "__main__":
    unittest.main()
