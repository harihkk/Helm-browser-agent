"""Validator system tests.

Completion requires the validator to return COMPLETED with real evidence;
uncertain outcomes return UNVERIFIED; hard page walls return BLOCKED.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core import validators as V
from core.browser_engine import PageState


def st(url="", title="", content="", error=""):
    return PageState(url, title, content, [], error=error)


class AtomicValidators(unittest.TestCase):
    def test_url_and_domain(self):
        s = st("https://www.example.com/docs", "Docs", "hi")
        self.assertTrue(V.validate_url(s, "https://example.com/docs/"))
        self.assertFalse(V.validate_url(s, "https://other.com/docs"))
        self.assertTrue(V.validate_domain(s, "example.com"))
        self.assertFalse(V.validate_domain(s, "other.com"))

    def test_text_and_results_visible(self):
        s = st("https://x.com", "Cats page", "all about cats and kittens")
        self.assertTrue(V.validate_text_visible(s, "cats kittens"))
        self.assertFalse(V.validate_text_visible(s, "dogs puppies"))
        results = st("https://www.google.com/search?q=cats", "cats - Search", "...")
        self.assertTrue(V.validate_search_results_visible(results, "cats"))

    def test_page_not_error(self):
        self.assertTrue(V.validate_page_not_error(st("https://x.com", "ok", "ok")))
        self.assertFalse(V.validate_page_not_error(st("error", "x", "", error="boom")))

    def test_registry_lists_composable_validators(self):
        for name in ("validate_url", "validate_domain", "validate_text_visible",
                     "validate_search_results_visible", "validate_page_not_error",
                     "validate_data_extracted"):
            self.assertIn(name, V.VALIDATORS)
            self.assertTrue(callable(V.VALIDATORS[name]))


class CompletionGating(unittest.TestCase):
    def test_navigation_completed_and_failed(self):
        intent = {"task_type": "navigation", "target_url": "https://x.com/p"}
        ok = V.validate_completion(intent, st("https://x.com/p", "P", "welcome"), [], [])
        self.assertEqual(ok.status, V.COMPLETED)
        self.assertTrue(ok.ok)
        bad = V.validate_completion(intent, st("https://x.com/other", "O", "welcome"), [], [])
        self.assertEqual(bad.status, V.FAILED)
        self.assertFalse(bad.ok)

    def test_navigation_to_404_is_blocked(self):
        intent = {"task_type": "navigation", "target_url": "https://x.com/p"}
        r = V.validate_completion(intent, st("https://x.com/p", "404", "Page Not Found"), [], [])
        self.assertEqual(r.status, V.BLOCKED)
        self.assertEqual(r.blocker["blocker_type"], "page_not_found")

    def test_login_wall_blocks_any_task(self):
        intent = {"task_type": "web_search", "search_query": "cats"}
        r = V.validate_completion(
            intent, st("https://x.com", "Sign in", "sign in with your password to continue"),
            [], [{"url": "x"}])
        self.assertEqual(r.status, V.BLOCKED)
        self.assertEqual(r.blocker["blocker_type"], "login_required")

    def test_web_search_completed_only_when_results_visible(self):
        intent = {"task_type": "web_search", "search_query": "playwright locators"}
        good = V.validate_completion(
            intent, st("https://www.google.com/search?q=playwright+locators",
                       "playwright locators", "results about playwright locators"), [], [])
        self.assertEqual(good.status, V.COMPLETED)
        weak = V.validate_completion(
            intent, st("https://random.com", "Random", "unrelated content"),
            [{"action": "navigate", "success": True}], [{"x": 1}])
        self.assertEqual(weak.status, V.UNVERIFIED)

    def test_media_playback_requires_playback_validator(self):
        intent = {"task_type": "media_playback", "search_query": "song"}
        without = V.validate_completion(intent, st("https://youtube.com/watch", "v", ""), [], [])
        self.assertEqual(without.status, V.UNVERIFIED)
        with_play = V.validate_completion(
            intent, st("https://youtube.com/watch", "v", ""),
            [{"action": "ensure_youtube_playback", "success": True}], [])
        self.assertEqual(with_play.status, V.COMPLETED)

    def test_note_creation(self):
        intent = {"task_type": "note_creation", "content_to_type": "buy milk"}
        ok = V.validate_completion(
            intent, st("https://keep.google.com", "Keep", ""),
            [{"action": "write_google_keep_note", "success": True}], [])
        self.assertEqual(ok.status, V.COMPLETED)
        unk = V.validate_completion(intent, st("https://keep.google.com", "Keep", ""), [], [])
        self.assertEqual(unk.status, V.UNVERIFIED)

    def test_cart_requires_match_confirmation_and_reviews(self):
        intent = {"task_type": "cart_update", "search_query": "ipad",
                  "constraints": {"open_reviews": True}}
        hist = [{"action": "add_amazon_item_to_cart", "success": True,
                 "data": {"product_match": True, "cart_confirmed": True, "reviews_opened": False}}]
        no_reviews = V.validate_completion(intent, st("https://amazon.com/dp/X", "iPad", "added to cart"), hist, [])
        self.assertEqual(no_reviews.status, V.UNVERIFIED)
        hist[0]["data"]["reviews_opened"] = True
        with_reviews = V.validate_completion(
            intent, st("https://amazon.com/product-reviews/X", "Reviews", "customer reviews"), hist, [])
        self.assertEqual(with_reviews.status, V.COMPLETED)

    def test_information_extraction_and_generic(self):
        ie = {"task_type": "information_extraction"}
        self.assertEqual(
            V.validate_completion(ie, st("https://x.com", "x", "data"), [], [{"x": 1}]).status,
            V.COMPLETED)
        self.assertEqual(
            V.validate_completion(ie, st("https://x.com", "x", "data"), [], []).status,
            V.UNVERIFIED)
        gen = {"task_type": "generic_browser_task"}
        self.assertEqual(
            V.validate_completion(gen, st("https://x.com", "x", "data"),
                                  [{"action": "click", "success": True}], []).status,
            V.UNVERIFIED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
