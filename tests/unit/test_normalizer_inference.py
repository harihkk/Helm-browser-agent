"""Inference tests for PromptNormalizer: infer_target_domain, has_purchase_intent,
has_media_intent.

These cover the boundaries that matter: multi-word product signals shadowing
shorter ones, purchase verbs that sit inside a note/reminder clause (and the
4-token window edge), and media-vs-everything-else. Parameterized with
subTest so each row is reported independently (unittest is this project's test
runner; pytest is not installed and CI runs `unittest discover -s tests/unit`).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.prompt_normalizer import PromptNormalizer


class InferTargetDomain(unittest.TestCase):
    def setUp(self):
        self.n = PromptNormalizer()

    def test_maps_product_signal_to_domain(self):
        cases = [
            # Apple hardware families
            ("buy me an iPhone 16 Pro Max", "apple.com"),
            ("what does the MacBook Air M3 cost", "apple.com"),
            ("MacBook Pro 14 inch", "apple.com"),
            ("AirPods Pro 2 price", "apple.com"),
            ("an iPad mini for my kid", "apple.com"),
            ("mac mini base model", "apple.com"),
            ("imac 24 inch", "apple.com"),
            # Multi-word entries must win over the shorter ones they overlap
            ("Apple Watch Series 9", "apple.com"),
            ("Pixel Watch 2 review", "store.google.com"),
            ("Pixel Buds Pro", "store.google.com"),
            ("Galaxy Watch 6 classic", "samsung.com"),
            ("Galaxy Tab S9", "samsung.com"),
            # Phone lines
            ("Google Pixel 9 Pro", "store.google.com"),
            ("Samsung Galaxy S24 Ultra", "samsung.com"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(self.n.infer_target_domain(text), expected)

    def test_no_signal_returns_empty(self):
        for text in (
            "find me a good Python book",
            "order some AA batteries",
            "get me a USB-C hub",
            "summarize this article",
            "",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.n.infer_target_domain(text), "")


class HasPurchaseIntent(unittest.TestCase):
    def setUp(self):
        self.n = PromptNormalizer()

    def test_true_for_genuine_purchase(self):
        for text in (
            "buy me an iPhone",
            "order some AA batteries",
            "purchase a standing desk",
            "find me a good Python book",
            "get me a USB-C hub",
            "what is the price of the MacBook Air",
            "how much does a yoga mat cost",
            "cheapest noise cancelling headphones",
            # note word present, but OUTSIDE the 4-token window before the verb
            "write a long detailed shopping list and then buy everything",
        ):
            with self.subTest(text=text):
                self.assertTrue(self.n.has_purchase_intent(text))

    def test_false_when_verb_is_inside_a_note_clause(self):
        for text in (
            "write a note to buy milk",
            "set a reminder to order pizza",
            "remember to purchase the tickets",
            "write a note in keep to buy groceries",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.n.has_purchase_intent(text))

    def test_false_when_no_purchase_verb(self):
        for text in (
            "open reddit and read the top post",
            "play blinding lights",
            "search for python tutorials",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.n.has_purchase_intent(text))


class HasMediaIntent(unittest.TestCase):
    def setUp(self):
        self.n = PromptNormalizer()

    def test_true_for_media_verbs(self):
        for text in (
            "play blinding lights",
            "watch the new trailer",
            "listen to lofi beats",
            "stream the game",
        ):
            with self.subTest(text=text):
                self.assertTrue(self.n.has_media_intent(text))

    def test_false_for_non_media(self):
        for text in (
            "buy me an iPhone",
            "search for python tutorials",
            "open reddit",
            "read this article",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(self.n.has_media_intent(text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
