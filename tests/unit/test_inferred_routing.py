"""Inferred routing reaches the DIRECT site, not a Google intermediary.

These assert the routing CONTRACT, not memorized URLs: a no-site-name product /
brand / media prompt must produce a navigate whose host is the inferred site and
whose path is that site's own search route - never google.com/search?q=site:...
(the regression these guard against).
"""

import os
import sys
import unittest
from urllib.parse import urlparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.intent_planner import IntentPlanner


def _first_nav_host_path(planner, goal):
    action = planner.plan(goal, {"url": "about:blank"}) or {}
    url = (action.get("parameters", {}) or {}).get("url", "")
    parsed = urlparse(url)
    return action.get("action", ""), parsed.netloc.lower().removeprefix("www."), parsed.path.lower(), url


class InferredDirectRouting(unittest.TestCase):
    def setUp(self):
        self.p = IntentPlanner()

    # (prompt, expected host, a substring the path must contain)
    CASES = [
        ("buy me an iPhone 16 Pro Max", "apple.com", "/search"),
        ("how much is a MacBook Air", "apple.com", "/search"),
        ("whats the cheapest Samsung Galaxy S24", "samsung.com", "/search"),
        ("order some AA batteries", "amazon.com", "/s"),
        ("play Blinding Lights", "youtube.com", "/results"),
    ]

    def test_routes_directly_to_inferred_site(self):
        for prompt, host, path_part in self.CASES:
            with self.subTest(prompt=prompt):
                action, got_host, got_path, url = _first_nav_host_path(self.p, prompt)
                self.assertEqual(action, "navigate", f"{prompt!r} -> {url!r}")
                self.assertEqual(got_host, host, f"{prompt!r} -> {url!r}")
                self.assertIn(path_part, got_path, f"{prompt!r} -> {url!r}")

    def test_never_routes_through_a_google_site_search(self):
        for prompt, host, _ in self.CASES:
            with self.subTest(prompt=prompt):
                _, got_host, _, url = _first_nav_host_path(self.p, prompt)
                self.assertNotIn("google.com", got_host,
                                 f"{prompt!r} fell back to a Google intermediary: {url!r}")

    def test_pixel_product_name_survives_query_cleaning(self):
        # Regression: "pixel" is a store.google.com alias but also the product;
        # it must not be stripped out of the search query.
        _, host, _, url = _first_nav_host_path(self.p, "get me a Pixel 9 Pro")
        self.assertEqual(host, "store.google.com", url)
        self.assertIn("pixel", url.lower(), url)


if __name__ == "__main__":
    unittest.main()
