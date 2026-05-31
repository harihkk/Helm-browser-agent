"""Structured blocker tests - schema, vocabulary, and classification."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import blockers as B


MANDATED_TYPES = [
    "bad_url", "page_not_found", "navigation_failed", "wrong_domain",
    "login_required", "captcha_or_bot_protection", "permission_denied",
    "popup_blocking", "missing_element", "disabled_element", "no_results",
    "unavailable_option", "unsupported_action", "timeout", "validation_failed",
    "ambiguous_instruction", "confirmation_required", "unsafe_action",
    "stale_run_event", "partial_completion",
]


class Vocabulary(unittest.TestCase):
    def test_all_mandated_types_present(self):
        for t in MANDATED_TYPES:
            self.assertIn(t, B.BLOCKER_TYPES)


class Schema(unittest.TestCase):
    def test_to_dict_has_full_schema(self):
        b = B.Blocker(blocker_type="login_required", blocker_message="sign in",
                      current_url="https://x.com", failed_step=3, last_successful_step=2)
        d = b.to_dict()
        for key in ("status", "blocker_type", "blocker_message", "current_url",
                    "page_title", "failed_step", "last_successful_step",
                    "attempted_recoveries", "visible_evidence", "suggested_next_step"):
            self.assertIn(key, d)
        self.assertTrue(d["suggested_next_step"])  # always actionable

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            B.Blocker(blocker_type="totally_made_up", blocker_message="x")

    def test_status_derivation(self):
        self.assertEqual(B.Blocker("login_required", "x").status, "blocked")
        self.assertEqual(B.Blocker("validation_failed", "x").status, "failed")
        self.assertEqual(B.Blocker("page_not_found", "x").status, "blocked")
        self.assertEqual(B.Blocker("unsupported_action", "x").status, "failed")


class Classification(unittest.TestCase):
    def test_classifies_common_walls(self):
        self.assertEqual(
            B.classify_blocker(title="About", content="reCAPTCHA unusual traffic").blocker_type,
            "captcha_or_bot_protection")
        self.assertEqual(
            B.classify_blocker(title="Login", content="please sign in").blocker_type,
            "login_required")
        self.assertEqual(
            B.classify_blocker(title="404", content="Page Not Found").blocker_type,
            "page_not_found")
        self.assertEqual(
            B.classify_blocker(content="Your search did not match any results").blocker_type,
            "no_results")
        self.assertEqual(
            B.classify_blocker(last_error="Timed out waiting for selector").blocker_type,
            "timeout")
        self.assertEqual(
            B.classify_blocker(last_error="Unsupported browser action: x").blocker_type,
            "unsupported_action")
        self.assertEqual(
            B.classify_blocker(last_error="Refusing to type a full command").blocker_type,
            "unsafe_action")
        self.assertEqual(
            B.classify_blocker(last_error="Could not find a click element: #x").blocker_type,
            "missing_element")

    def test_error_page_is_bad_url(self):
        self.assertEqual(
            B.classify_blocker(url="error", last_error="net::ERR", is_error=True).blocker_type,
            "bad_url")

    def test_carries_step_context(self):
        b = B.classify_blocker(title="Login", content="sign in", failed_step=4,
                               last_successful_step=3, attempted_recoveries=["scroll"])
        self.assertEqual(b.failed_step, 4)
        self.assertEqual(b.last_successful_step, 3)
        self.assertEqual(b.attempted_recoveries, ["scroll"])


class Confirmation(unittest.TestCase):
    def test_confirmation_blocker(self):
        b = B.confirmation_blocker("Confirm the purchase", url="https://x.com")
        self.assertEqual(b.blocker_type, "confirmation_required")
        self.assertEqual(b.status, "blocked")
        self.assertTrue(b.suggested_next_step)


if __name__ == "__main__":
    unittest.main(verbosity=2)
