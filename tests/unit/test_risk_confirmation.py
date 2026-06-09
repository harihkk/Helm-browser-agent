"""Risk classification and confirmation-gate tests (unit level)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import risk as R
from core.intent_planner import IntentPlanner


class ClassifyRisk(unittest.TestCase):
    def test_high_impact_verbs_require_confirmation(self):
        for prompt in [
            "buy the iphone 16 on apple",
            "checkout my amazon cart",
            "send an email to my boss",
            "submit the contact form",
            "delete my last tweet",
            "post this update to my feed",
            "place an order for groceries",
            "commit and push the changes",
        ]:
            with self.subTest(prompt=prompt):
                self.assertTrue(R.classify_risk(prompt)["requires_confirmation"], prompt)

    def test_safe_verbs_do_not_require_confirmation(self):
        for prompt in [
            "open https://example.com",
            "search the web for cats",
            "read the article on this page",
            "summarize this page",
            "play lofi beats on youtube",
            "find the refund policy",
        ]:
            with self.subTest(prompt=prompt):
                self.assertFalse(R.classify_risk(prompt)["requires_confirmation"], prompt)

    def test_safe_family_overrides_content_verbs(self):
        # A high-impact word ("buy") inside note content must NOT trigger
        # confirmation once the task family is known to be safe.
        self.assertFalse(
            R.classify_risk("write buy milk in google keep",
                            task_type="note_creation")["requires_confirmation"])
        self.assertFalse(
            R.classify_risk("search for how to cancel my subscription",
                            task_type="web_search")["requires_confirmation"])

    def test_add_to_cart_is_high_impact_but_plain_add_is_not(self):
        self.assertTrue(R.classify_risk("add airpods to cart")["requires_confirmation"])
        self.assertTrue(R.classify_risk("add the book to my basket")["requires_confirmation"])
        self.assertFalse(R.classify_risk("add a column to the table")["requires_confirmation"])


class ActionGate(unittest.TestCase):
    def test_high_impact_action_always_gated(self):
        self.assertTrue(R.action_requires_confirmation(
            "add_amazon_item_to_cart", {}, {"requires_confirmation": False}))

    def test_lead_up_actions_not_gated(self):
        intent = {"requires_confirmation": True}
        for a in ("navigate", "extract", "scroll", "wait", "press_key", "done"):
            self.assertFalse(R.action_requires_confirmation(a, {}, intent), a)

    def test_only_the_commit_action_is_gated(self):
        intent = {"requires_confirmation": True}
        # The genuine pay / place-order / confirm step pauses...
        for params in ({"text": "Pay now"}, {"text": "Place order"},
                       {"selector": "#confirm-and-pay"}, {"text": "Buy now"},
                       {"text": "Proceed to payment"}):
            self.assertTrue(R.action_requires_confirmation("click", params, intent), params)
        # ...but cookie banners, searches, and option-picking flow freely, even
        # inside a high-impact task like booking a flight.
        for params in ({}, {"selector": "#cookie-accept"}, {"text": "Accept"},
                       {"selector": "#search-flights"}, {"text": "Search flights"},
                       {"text": "Select this flight"}):
            self.assertFalse(R.action_requires_confirmation("click", params, intent), params)

    def test_no_gate_for_low_risk_non_commit_action(self):
        intent = {"requires_confirmation": False}
        self.assertFalse(R.action_requires_confirmation("click", {}, intent))
        # A pay click is gated regardless of the task's overall risk.
        self.assertTrue(R.action_requires_confirmation("click", {"text": "Pay now"}, intent))

    def test_confirmation_message_non_empty(self):
        self.assertTrue(R.confirmation_message({"task_type": "cart_update", "search_query": "ipad"}))


class IntentReflectsRisk(unittest.TestCase):
    def test_cart_intent_flags_confirmation(self):
        intent = IntentPlanner().parse_intent("go to amazon and add airpods to cart")
        self.assertTrue(intent.requires_confirmation)
        self.assertEqual(intent.risk_level, "high")
        self.assertTrue(intent.to_dict()["needs_user_confirmation"])

    def test_search_intent_is_low_risk(self):
        intent = IntentPlanner().parse_intent("search the web for cats")
        self.assertFalse(intent.requires_confirmation)
        self.assertEqual(intent.risk_level, "low")

    def test_note_with_buy_word_is_safe(self):
        # End-to-end: the note pipeline knows "buy milk" is content, not a buy.
        intent = IntentPlanner().parse_intent("write buy milk tomorrow in google keep")
        self.assertEqual(intent.task_type, "note_creation")
        self.assertFalse(intent.requires_confirmation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
