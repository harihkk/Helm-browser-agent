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


class QueryIsClean(unittest.TestCase):
    """The query must never carry a site name, a URL, or leading command words,
    regardless of how the user phrases it (the systemic bug behind the reddit
    'reddit.com uber vs lyft' leak)."""

    def setUp(self):
        self.p = IntentPlanner()

    # (prompt, tokens that MUST appear, tokens that must NOT)
    CASES = [
        ("can u open reddit.com and look for uber vs lyft", ["uber", "lyft"], ["reddit", "reddit.com", "look", "for"]),
        ("buy me an iPhone 16 Pro Max", ["iphone"], ["buy", "me"]),
        ("order some AA batteries", ["batteries"], ["order", "some"]),
        ("get me a Pixel 9 Pro", ["pixel"], ["get", "me"]),
        ("find python decorators on stackoverflow", ["python", "decorators"], ["find", "stackoverflow", "on"]),
    ]

    def test_query_carries_payload_not_scaffolding(self):
        for prompt, must, must_not in self.CASES:
            with self.subTest(prompt=prompt):
                q = (self.p.parse_intent(prompt).search_query or "").lower()
                for tok in must:
                    self.assertIn(tok, q, f"{prompt!r} -> query {q!r} missing {tok!r}")
                for tok in must_not:
                    self.assertNotIn(tok, q.split(), f"{prompt!r} -> query {q!r} leaked {tok!r}")

    def test_no_bare_domain_survives_in_query(self):
        for prompt in ("open reddit.com and look for uber vs lyft",
                       "search amazon.com for a yoga mat"):
            with self.subTest(prompt=prompt):
                q = self.p.parse_intent(prompt).search_query or ""
                self.assertNotRegex(q.lower(), r"\b[a-z0-9-]+\.(?:com|org|net)\b",
                                    f"{prompt!r} leaked a domain into {q!r}")


if __name__ == "__main__":
    unittest.main()
