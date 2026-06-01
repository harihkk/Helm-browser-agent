"""Executor / adaptive-loop tests.

These drive the *real* SophisticatedTaskOrchestrator with a fake browser and a
scripted AI, exercising observation, recovery, validation gating, structured
blockers, and the risk/confirmation gate through the same loop the live
WebSocket uses.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.task_orchestrator import SophisticatedTaskOrchestrator, TaskStatus
from tests.unit.fakes import FakeBrowser, ScriptedAI, drain, terminal, page


def orch(browser, ai):
    return SophisticatedTaskOrchestrator(browser, ai)


class NavigationFlow(unittest.TestCase):
    def test_navigation_completes_with_validation_evidence(self):
        browser = FakeBrowser(routes={
            "example.com/docs": page("https://example.com/docs", "Docs", "Documentation home"),
        })
        ai = ScriptedAI([
            {"action": "navigate", "parameters": {"url": "https://example.com/docs"}, "confidence": 0.95},
            {"action": "done", "task_complete": True, "parameters": {"summary": "Opened docs"}},
        ])
        events = drain(orch(browser, ai), "open https://example.com/docs", {"max_steps": 5})
        end = terminal(events)
        self.assertEqual(end["type"], "task_completed")
        self.assertEqual(end["status"], "completed")
        # Completed runs must carry validation evidence (requirement #6).
        self.assertEqual(end["validation"]["status"], "completed")
        self.assertTrue(end["validation"]["success_condition"])

    def test_observes_page_after_each_action(self):
        browser = FakeBrowser(routes={
            "example.com": page("https://example.com", "Example", "Example domain content"),
        })
        ai = ScriptedAI([
            {"action": "navigate", "parameters": {"url": "https://example.com"}},
            {"action": "extract", "parameters": {"target": "page"}},
            {"action": "done", "task_complete": True, "parameters": {"summary": "read"}},
        ])
        drain(orch(browser, ai), "open https://example.com and read it", {"max_steps": 6})
        # One observation at start + one after each executed action.
        self.assertGreaterEqual(browser.state_calls, 3)


class FailureAndBlockers(unittest.TestCase):
    def test_failed_step_is_recorded_not_skipped(self):
        def responder(action, params, state):
            if action == "click":
                return {"success": False, "action": "click",
                        "error": "Could not find a click element: #ghost"}, None
            if action == "navigate":
                return {"success": True, "action": "navigate"}, page(
                    "https://site.com", "Site", "hello")
            return {"success": True, "action": action}, None

        browser = FakeBrowser(responder=responder)
        ai = ScriptedAI([
            {"action": "navigate", "parameters": {"url": "https://site.com"}},
            {"action": "click", "parameters": {"selector": "#ghost"}},
            {"action": "done", "task_complete": True, "parameters": {"summary": "x"}},
        ])
        events = drain(orch(browser, ai), "open https://site.com and click ghost", {"max_steps": 6})
        clicks = [e for e in events if e.get("type") == "step_executed" and e.get("action") == "click"]
        self.assertTrue(clicks, "the failed click step must be reported, not skipped")
        self.assertFalse(clicks[0]["success"])
        self.assertIn("click element", clicks[0]["error"])

    def test_unsupported_action_yields_structured_blocker(self):
        browser = FakeBrowser()
        ai = ScriptedAI([
            {"action": "launch_spaceship", "parameters": {}, "confidence": 0.9},
        ])
        events = drain(orch(browser, ai), "do something impossible", {"max_steps": 4})
        end = terminal(events)
        self.assertEqual(end["type"], "task_failed")
        self.assertEqual(end["blocker_type"], "unsupported_action")
        self.assertTrue(end["suggested_next_step"])
        self.assertIn("blocker", end)

    def test_navigation_to_404_is_blocked_not_completed(self):
        browser = FakeBrowser(routes={
            "broken": page("https://x.com/broken", "404 Not Found", "Page Not Found"),
        })
        ai = ScriptedAI([
            {"action": "navigate", "parameters": {"url": "https://x.com/broken"}},
            {"action": "done", "task_complete": True, "parameters": {"summary": "opened"}},
        ])
        events = drain(orch(browser, ai), "open https://x.com/broken", {"max_steps": 3})
        end = terminal(events)
        self.assertEqual(end["status"], "blocked")
        self.assertEqual(end["blocker_type"], "page_not_found")

    def test_login_wall_is_reported_blocked(self):
        browser = FakeBrowser(routes={
            "members": page("https://site.com/members", "Sign in",
                            "Please sign in with your password to continue. Create account."),
        })
        ai = ScriptedAI([
            {"action": "navigate", "parameters": {"url": "https://site.com/members"}},
            {"action": "done", "task_complete": True, "parameters": {"summary": "opened"}},
        ])
        events = drain(orch(browser, ai), "open https://site.com/members", {"max_steps": 3})
        end = terminal(events)
        self.assertEqual(end["status"], "blocked")
        self.assertEqual(end["blocker_type"], "login_required")

    def test_run_out_of_steps_is_unverified_not_falsely_completed(self):
        # The agent keeps extracting a generic page it can't prove satisfies
        # the goal -> should end UNVERIFIED, never a fake "completed".
        browser = FakeBrowser(routes={
            "site.com": page("https://site.com", "Site", "some unrelated content"),
        })
        ai = ScriptedAI([
            {"action": "navigate", "parameters": {"url": "https://site.com"}},
        ])  # then exhausted -> repeated extracts
        events = drain(orch(browser, ai),
                       "find the quarterly revenue figure on site.com", {"max_steps": 4})
        end = terminal(events)
        self.assertIn(end["status"], ("unverified", "blocked", "failed"))
        self.assertNotEqual(end["status"], "completed")


class ConfirmationGate(unittest.TestCase):
    def _cart_ai(self):
        return ScriptedAI([
            {"action": "navigate", "parameters": {"url": "https://www.amazon.com/s?k=airpods"}},
            {"action": "add_amazon_item_to_cart", "parameters": {"query": "airpods"}},
        ])

    def _cart_browser(self):
        def responder(action, params, state):
            if action == "navigate":
                return {"success": True, "action": "navigate"}, page(
                    "https://www.amazon.com/s?k=airpods", "Amazon airpods", "results")
            if action == "add_amazon_item_to_cart":
                return ({"success": True, "action": "add_amazon_item_to_cart",
                         "task_complete": True, "summary": "Added AirPods to cart.",
                         "data": {"product_match": True, "cart_confirmed": True}},
                        page("https://www.amazon.com/cart", "Cart", "Added to cart subtotal"))
            return {"success": True, "action": action}, None
        return FakeBrowser(responder=responder)

    def test_high_impact_action_pauses_for_confirmation(self):
        browser = self._cart_browser()
        events = drain(orch(browser, self._cart_ai()),
                       "go to amazon and add airpods to cart", {"max_steps": 5})
        end = terminal(events)
        self.assertEqual(end["status"], "blocked")
        self.assertEqual(end["blocker_type"], "confirmation_required")
        # The mutating action must NOT have executed.
        self.assertNotIn("add_amazon_item_to_cart",
                         [a for a, _ in browser.action_calls])

    def test_preconfirmed_run_executes_high_impact_action(self):
        browser = self._cart_browser()
        events = drain(orch(browser, self._cart_ai()),
                       "go to amazon and add airpods to cart",
                       {"max_steps": 5, "confirmed": True})
        end = terminal(events)
        self.assertEqual(end["status"], "completed")
        self.assertIn("add_amazon_item_to_cart",
                      [a for a, _ in browser.action_calls])


class SuccessConditionGuard(unittest.TestCase):
    def test_missing_success_condition_fails_before_execution(self):
        # Force the planner to produce no success condition; the orchestrator
        # must refuse to execute and emit ambiguous_instruction.
        browser = FakeBrowser()
        ai = ScriptedAI([{"action": "extract", "parameters": {}}])
        planner = ai.intent_planner
        planner._success_condition_for = lambda *a, **k: ""  # type: ignore
        events = drain(orch(browser, ai), "do the thing", {"max_steps": 3})
        end = terminal(events)
        self.assertEqual(end["status"], "failed")
        self.assertEqual(end["blocker_type"], "ambiguous_instruction")
        self.assertEqual(browser.action_calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
