"""iPhone buy-page slug tests.

Apple groups models on shared configuration pages: base + Plus on /iphone-<n>,
Pro + Pro Max on /iphone-<n>-pro. A dedicated /iphone-<n>-pro-max page does not
exist (it 404s), so Pro Max must resolve to the -pro slug.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.intent_planner import IntentPlanner


class IphoneBuySlug(unittest.TestCase):
    def setUp(self):
        self.p = IntentPlanner()

    def test_base_model(self):
        self.assertEqual(self.p._iphone_buy_slug("iPhone 16"), "iphone-16")

    def test_plus_shares_base_page(self):
        self.assertEqual(self.p._iphone_buy_slug("iPhone 16 Plus"), "iphone-16")

    def test_pro(self):
        self.assertEqual(self.p._iphone_buy_slug("iPhone 16 Pro"), "iphone-16-pro")

    def test_pro_max_shares_pro_page_not_a_dead_url(self):
        # Regression: must NOT be the 404 slug "iphone-16-pro-max".
        self.assertEqual(self.p._iphone_buy_slug("iPhone 16 Pro Max"), "iphone-16-pro")

    def test_other_generation(self):
        self.assertEqual(self.p._iphone_buy_slug("iPhone 15 Pro Max"), "iphone-15-pro")

    def test_no_version_number_falls_back_cleanly(self):
        # No double dashes even on the fallback path.
        slug = self.p._iphone_buy_slug("iPhone SE")
        self.assertNotIn("--", slug)
        self.assertTrue(slug.startswith("iphone-"))


if __name__ == "__main__":
    unittest.main()
