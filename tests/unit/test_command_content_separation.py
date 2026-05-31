"""Command vs content separation tests.

Prove the system never types command scaffolding as content, and never leaves
destination/URL/politeness tokens inside a search query.
"""

import os
import re
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.intent_planner import IntentPlanner


FORBIDDEN_IN_QUERY = [
    "go to", "goto", "can you", "can u", "please", "look up",
    "search for", "inside https", "http://", "https://",
]


class QueryDoesNotContainCommands(unittest.TestCase):
    def setUp(self):
        self.p = IntentPlanner()

    def _check_query_clean(self, prompt):
        intent = self.p.parse_intent(prompt)
        q = (intent.search_query or "").lower()
        for bad in FORBIDDEN_IN_QUERY:
            self.assertNotIn(bad, q, f"query for {prompt!r} leaked command token {bad!r}: {q!r}")
        return intent

    def test_youtube_query_is_clean(self):
        intent = self._check_query_clean(
            "cna u go to youtube and search for drop dead by olivia rodrigo and actually play that song")
        self.assertEqual(intent.search_query, "drop dead by olivia rodrigo")

    def test_repo_search_query_is_clean_and_url_removed(self):
        intent = self._check_query_clean(
            "search for auth logic inside https://github.com/harihkk/helm")
        self.assertEqual(intent.search_query, "auth logic")
        self.assertNotIn("github.com", intent.search_query)

    def test_site_search_query_drops_site_and_scaffolding(self):
        intent = self._check_query_clean(
            "go to reddit.com and search for netherlands carrots season")
        self.assertNotIn("reddit", intent.search_query.lower())
        self.assertIn("carrots", intent.search_query.lower())

    def test_web_search_query_is_clean(self):
        self._check_query_clean("please can you search the web for playwright locators")

    def test_batch_of_prompts_keep_queries_clean(self):
        prompts = [
            "go to youtube and search for lofi music and actually play it",
            "search the web for fastapi websockets tutorial",
            "look up the capital of france",
            "go to amazon and search for usb c cable",
            "find the refund policy inside https://example.com",
            "can u go over to linkedin and look for google recruiters",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self._check_query_clean(prompt)


class ContentIsNotCommand(unittest.TestCase):
    def setUp(self):
        self.p = IntentPlanner()

    def test_note_content_excludes_destination_and_verb(self):
        intent = self.p.parse_intent("write remember to call Sam tomorrow in my google keep")
        self.assertEqual(intent.content_to_type, "remember to call Sam tomorrow")
        low = intent.content_to_type.lower()
        for bad in ("write", "google keep", "in my", "keep"):
            self.assertNotIn(bad, low)

    def test_note_content_simple(self):
        intent = self.p.parse_intent("write buy milk tomorrow in google keep")
        self.assertEqual(intent.content_to_type, "buy milk tomorrow")

    def test_typed_content_excludes_command_words(self):
        intent = self.p.parse_intent("write call the bank at noon in keep")
        self.assertEqual(intent.content_to_type, "call the bank at noon")
        self.assertNotIn("write", intent.content_to_type.lower())
        self.assertNotIn("keep", intent.content_to_type.lower())


class StructuredFieldsPopulated(unittest.TestCase):
    def setUp(self):
        self.p = IntentPlanner()

    def test_url_site_query_separated(self):
        intent = self.p.parse_intent("search for pricing inside https://example.com")
        self.assertEqual(intent.target_url, "https://example.com")
        self.assertEqual(intent.search_query, "pricing")
        self.assertEqual(intent.context_scope, "provided_url")

    def test_click_target_and_scope_for_url_task(self):
        intent = self.p.parse_intent("open https://example.com and click contact")
        self.assertEqual(intent.target_url, "https://example.com")
        self.assertTrue(intent.success_condition)


if __name__ == "__main__":
    unittest.main(verbosity=2)
