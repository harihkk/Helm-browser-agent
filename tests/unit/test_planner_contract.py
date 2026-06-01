"""Planner contract tests.

Every plan the planner emits must: consume structured intent, name a registered
action, validate its parameters, carry a success condition and a validation
method, and never emit an unexecutable action (except the terminal 'done').
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.intent_planner import IntentPlanner, MissingSuccessConditionError
from core.action_registry import (
    is_supported_action, is_executable_action, validate_action,
)


PROMPTS_AND_STATES = [
    ("open https://example.com/docs", {"url": "about:blank"}),
    ("search the web for playwright locators", {"url": "about:blank"}),
    ("go to youtube and search for lofi music and play it", {"url": "about:blank"}),
    ("write buy milk tomorrow in google keep", {"url": "about:blank"}),
    ("search for auth logic inside https://github.com/harihkk/helm repo", {"url": "about:blank"}),
    ("go to amazon and add airpods to cart", {"url": "about:blank"}),
    ("open apple website and look for iphone 16 256 gb price", {"url": "about:blank"}),
    ("go to wikipedia and search machine learning", {"url": "about:blank"}),
    ("go to reddit.com and search for local llms", {"url": "about:blank"}),
    ("can u go over to linkedin and look for google recruiters", {"url": "about:blank"}),
    ("go to github and look for torvalds profile", {"url": "about:blank"}),
    ("find docs about browser automation reliability", {"url": "about:blank"}),
    ("compare iphone 16 and pixel 9 cameras", {"url": "about:blank"}),
]


class PlanContract(unittest.TestCase):
    def setUp(self):
        self.p = IntentPlanner()

    def test_every_plan_satisfies_the_contract(self):
        for prompt, state in PROMPTS_AND_STATES:
            with self.subTest(prompt=prompt):
                plan = self.p.plan(prompt, state)
                self.assertIsNotNone(plan, f"no plan for {prompt!r}")
                action = plan["action"]
                # Registered.
                self.assertTrue(is_supported_action(action), f"{action} not registered")
                # Executable (or the terminal done).
                self.assertTrue(
                    is_executable_action(action) or action == "done",
                    f"{action} is not executable")
                # Parameters validate against the registry (won't raise).
                validate_action(action, plan.get("parameters", {}))
                # Success condition + validation method present.
                self.assertTrue(plan["success_condition"], f"no success_condition for {prompt!r}")
                self.assertTrue(plan["validation_method"], f"no validation_method for {prompt!r}")
                # Structured intent attached.
                self.assertIsNotNone(plan["intent"])
                self.assertTrue(plan["intent"]["success_condition"])

    def test_plan_fails_without_success_condition(self):
        p = IntentPlanner()
        p._success_condition_for = lambda *a, **k: ""  # type: ignore
        with self.assertRaises(MissingSuccessConditionError):
            p.plan("do the thing", {"url": "about:blank"})

    def test_to_analysis_rejects_unregistered_action(self):
        from core.intent_planner import IntentAction
        from core.action_registry import UnsupportedActionError
        bad = IntentAction(action="launch_spaceship", parameters={},
                           reasoning="x", thinking="y")
        with self.assertRaises(UnsupportedActionError):
            bad.to_analysis()

    def test_intent_object_has_pipeline_fields(self):
        intent = self.p.parse_intent("open https://example.com/docs").to_dict()
        for key in ("run_id", "primary_action", "task_family", "target_domain",
                    "context_scope", "expected_output", "validation_strategy",
                    "success_condition", "needs_user_confirmation",
                    "allowed_action_types", "likely_blockers"):
            self.assertIn(key, intent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
