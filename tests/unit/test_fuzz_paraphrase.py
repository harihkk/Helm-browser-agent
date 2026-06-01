"""Fuzz / paraphrase tests.

For a set of canonical task intents, generate messy variations (typos, slang,
politeness, anger, filler, reordering, direct URL vs site name) and assert the
extracted structured intent stays stable.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.intent_planner import IntentPlanner


def variants(core_phrase):
    """Produce paraphrase variations of a core instruction phrase."""
    return [
        core_phrase,
        f"can you {core_phrase}",
        f"cna u {core_phrase} plz",
        f"hey, {core_phrase} for me",
        f"just {core_phrase} already",
        f"please {core_phrase} when you get a chance",
        f"{core_phrase}!!!",
        f"  {core_phrase}  ",
    ]


class StableIntent(unittest.TestCase):
    def setUp(self):
        self.p = IntentPlanner()

    def test_media_playback_stable(self):
        for v in variants("go to youtube and play despacito"):
            with self.subTest(v=v):
                intent = self.p.parse_intent(v)
                self.assertEqual(intent.task_type, "media_playback")
                self.assertEqual(intent.target_site, "youtube.com")
                self.assertIn("despacito", intent.search_query.lower())

    def test_note_creation_stable_content(self):
        for v in variants("write buy milk tomorrow in google keep"):
            with self.subTest(v=v):
                intent = self.p.parse_intent(v)
                self.assertEqual(intent.task_type, "note_creation")
                self.assertIn("buy milk tomorrow", intent.content_to_type)
                self.assertNotIn("keep", intent.content_to_type.lower())

    def test_navigation_stable_for_url(self):
        for v in variants("open https://example.com/docs"):
            with self.subTest(v=v):
                intent = self.p.parse_intent(v)
                self.assertEqual(intent.target_url, "https://example.com/docs")

    def test_web_search_query_stable_and_clean(self):
        for v in variants("search the web for fastapi websockets"):
            with self.subTest(v=v):
                intent = self.p.parse_intent(v)
                q = intent.search_query.lower()
                self.assertIn("fastapi", q)
                self.assertIn("websockets", q)
                for bad in ("can you", "can u", "please", "go to", "http"):
                    self.assertNotIn(bad, q)

    def test_cart_intent_stable_and_high_risk(self):
        for v in variants("go to amazon and add airpods to cart"):
            with self.subTest(v=v):
                intent = self.p.parse_intent(v)
                self.assertEqual(intent.task_type, "cart_update")
                self.assertTrue(intent.requires_confirmation)

    def test_site_name_vs_direct_url_same_family(self):
        by_name = self.p.parse_intent("go to reddit and search for local llms")
        by_url = self.p.parse_intent("go to reddit.com and search for local llms")
        self.assertEqual(by_name.target_site, by_url.target_site)
        self.assertEqual(by_name.task_type, by_url.task_type)
        self.assertIn("local llms", by_name.search_query.lower())
        self.assertIn("local llms", by_url.search_query.lower())

    def test_every_variant_has_success_condition(self):
        cores = [
            "go to youtube and play despacito",
            "write buy milk in google keep",
            "open https://example.com",
            "search the web for cats",
            "go to amazon and add airpods to cart",
        ]
        for core in cores:
            for v in variants(core):
                with self.subTest(v=v):
                    self.assertTrue(self.p.parse_intent(v).success_condition)


if __name__ == "__main__":
    unittest.main(verbosity=2)
