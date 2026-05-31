"""Prompt normalization tests - messy, typo-heavy, angry, polite, vague, direct.

Critically: normalization must not destroy intended typed content.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.prompt_normalizer import PromptNormalizer


class Normalize(unittest.TestCase):
    def setUp(self):
        self.n = PromptNormalizer()

    def test_fixes_common_shorthand_and_typos(self):
        self.assertEqual(self.n.normalize("cna u go to youtube"), "can you go to youtube")
        self.assertEqual(self.n.normalize("plz open reddit"), "please open reddit")
        self.assertEqual(self.n.normalize("got o reddit.com"), "go to reddit.com")
        self.assertIn("and search", self.n.normalize("reddit.com an dsearch for carrots"))
        self.assertIn("apple website", self.n.normalize("open the apply website"))

    def test_collapses_whitespace(self):
        self.assertEqual(self.n.normalize("  open    example.com  "), "open example.com")

    def test_preserves_intended_content(self):
        # The note body must survive normalization intact.
        out = self.n.normalize('write "buy milk and eggs tomorrow" in keep')
        self.assertIn("buy milk and eggs tomorrow", out)

    def test_does_not_invent_words_in_clean_prompts(self):
        msg = "compare the iphone 16 and the pixel 9 prices"
        self.assertEqual(self.n.normalize(msg), msg)


class StructuralSignals(unittest.TestCase):
    def setUp(self):
        self.n = PromptNormalizer()

    def test_detect_urls(self):
        urls = self.n.detect_urls("open https://github.com/a/b and read it.")
        self.assertEqual(urls, ["https://github.com/a/b"])
        self.assertEqual(self.n.detect_urls("no url here"), [])

    def test_detect_quoted(self):
        self.assertEqual(
            self.n.detect_quoted('type "remember to call Sam" in notes'),
            ["remember to call Sam"],
        )

    def test_detect_action_verbs_and_primary_action(self):
        self.assertEqual(self.n.primary_action("go to example.com"), "navigate")
        self.assertEqual(self.n.primary_action("search for lofi beats"), "search")
        self.assertEqual(self.n.primary_action("play despacito on youtube"), "play")
        self.assertEqual(self.n.primary_action("summarize this article"), "read")
        self.assertEqual(self.n.primary_action("extract the table data"), "extract")
        # Outer intent wins: search-then-play is a play task.
        self.assertEqual(
            self.n.primary_action("search for despacito and play it"), "play")

    def test_primary_action_stable_across_tone(self):
        polite = "could you please open https://example.com when you get a chance"
        angry = "just OPEN https://example.com already"
        terse = "https://example.com"
        self.assertEqual(self.n.primary_action(polite), "navigate")
        self.assertEqual(self.n.primary_action(angry), "navigate")
        self.assertEqual(self.n.primary_action(terse), "navigate")


class StripFiller(unittest.TestCase):
    def setUp(self):
        self.n = PromptNormalizer()

    def test_strips_politeness_and_command_scaffolding(self):
        q = self.n.strip_filler("can you please go to and search for lofi hip hop beats")
        self.assertNotIn("can", q.lower().split())
        self.assertNotIn("please", q.lower())
        self.assertNotIn("go to", q.lower())
        self.assertIn("lofi hip hop beats", q)

    def test_removes_urls_from_query(self):
        q = self.n.strip_filler("find pricing inside https://example.com/page")
        self.assertNotIn("http", q)
        self.assertIn("pricing", q)


if __name__ == "__main__":
    unittest.main(verbosity=2)
