"""URL safety policy (SSRF guard) tests.

The agent navigates to URLs an LLM picks, possibly influenced by untrusted page
content. These tests pin the deterministic gate that decides what it may open.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.url_policy import check_url, is_safe_url, UrlVerdict


class AllowsPublicHttp(unittest.TestCase):
    def test_plain_https(self):
        self.assertTrue(check_url("https://example.com"))

    def test_http_with_path_query(self):
        self.assertTrue(check_url("http://example.com/search?q=hello&n=2"))

    def test_public_ip(self):
        self.assertTrue(check_url("https://8.8.8.8/"))

    def test_subdomain(self):
        self.assertTrue(check_url("https://maps.google.com/"))

    def test_verdict_is_truthy_and_typed(self):
        v = check_url("https://example.com")
        self.assertIsInstance(v, UrlVerdict)
        self.assertTrue(v.allowed)
        self.assertEqual(v.category, "ok")


class BlocksDangerousSchemes(unittest.TestCase):
    def test_file_scheme(self):
        v = check_url("file:///etc/passwd")
        self.assertFalse(v)
        self.assertEqual(v.category, "bad_scheme")

    def test_data_scheme(self):
        self.assertFalse(check_url("data:text/html,<script>alert(1)</script>"))

    def test_javascript_scheme(self):
        self.assertFalse(check_url("javascript:alert(document.cookie)"))

    def test_about_scheme(self):
        self.assertFalse(check_url("about:config"))

    def test_ftp_scheme(self):
        self.assertFalse(check_url("ftp://example.com/file"))

    def test_relative_url_has_no_scheme(self):
        v = check_url("/just/a/path")
        self.assertFalse(v)
        self.assertEqual(v.category, "bad_scheme")


class BlocksLocalAndPrivate(unittest.TestCase):
    def test_localhost(self):
        v = check_url("http://localhost:8000/admin")
        self.assertFalse(v)
        self.assertEqual(v.category, "private_host")

    def test_localhost_subdomain(self):
        self.assertFalse(check_url("http://api.localhost/"))

    def test_loopback_v4(self):
        self.assertFalse(check_url("http://127.0.0.1/"))

    def test_loopback_v6(self):
        self.assertFalse(check_url("http://[::1]/"))

    def test_unspecified(self):
        self.assertFalse(check_url("http://0.0.0.0:8000/"))

    def test_private_10(self):
        self.assertFalse(check_url("http://10.0.0.5/"))

    def test_private_192_168(self):
        self.assertFalse(check_url("http://192.168.1.1/"))

    def test_private_172_16(self):
        self.assertFalse(check_url("http://172.16.4.4/"))

    def test_link_local(self):
        self.assertFalse(check_url("http://169.254.1.1/"))


class BlocksMetadataEndpoints(unittest.TestCase):
    def test_aws_gcp_metadata_ip(self):
        v = check_url("http://169.254.169.254/latest/meta-data/")
        self.assertFalse(v)
        self.assertEqual(v.category, "metadata")

    def test_gcp_metadata_name(self):
        v = check_url("http://metadata.google.internal/computeMetadata/v1/")
        self.assertFalse(v)
        self.assertEqual(v.category, "metadata")

    def test_metadata_blocked_even_when_private_allowed(self):
        # Metadata exists only to hand out credentials; never reachable.
        v = check_url("http://169.254.169.254/", allow_private=True)
        self.assertFalse(v)
        self.assertEqual(v.category, "metadata")


class BlocksEncodedIpBypasses(unittest.TestCase):
    def test_decimal_packed_loopback(self):
        # 2130706433 == 127.0.0.1
        self.assertFalse(check_url("http://2130706433/"))

    def test_hex_packed_loopback(self):
        # 0x7f000001 == 127.0.0.1
        self.assertFalse(check_url("http://0x7f000001/"))

    def test_decimal_packed_public_is_allowed(self):
        # 134744072 == 8.8.8.8 (public) - should pass.
        self.assertTrue(check_url("http://134744072/"))


class AllowPrivateOptIn(unittest.TestCase):
    def test_localhost_allowed_when_opted_in(self):
        self.assertTrue(check_url("http://localhost:8000/", allow_private=True))

    def test_loopback_allowed_when_opted_in(self):
        self.assertTrue(check_url("http://127.0.0.1:3000/", allow_private=True))

    def test_env_flag_enables_private(self):
        os.environ["HELM_ALLOW_PRIVATE_HOSTS"] = "true"
        try:
            self.assertTrue(check_url("http://127.0.0.1/"))
        finally:
            del os.environ["HELM_ALLOW_PRIVATE_HOSTS"]


class AllowAndDenyLists(unittest.TestCase):
    def test_denylist_blocks_public_host(self):
        v = check_url("https://tracker.example.com/", denylist=["example.com"])
        self.assertFalse(v)
        self.assertEqual(v.category, "denylisted")

    def test_allowlist_only_mode(self):
        self.assertTrue(check_url("https://good.com/", allowlist=["good.com"]))
        v = check_url("https://other.com/", allowlist=["good.com"])
        self.assertFalse(v)
        self.assertEqual(v.category, "not_allowlisted")

    def test_denylist_wins_over_allowlist(self):
        v = check_url("https://bad.com/", allowlist=["bad.com"], denylist=["bad.com"])
        self.assertFalse(v)


class Malformed(unittest.TestCase):
    def test_empty(self):
        v = check_url("")
        self.assertFalse(v)
        self.assertEqual(v.category, "malformed")

    def test_none(self):
        self.assertFalse(check_url(None))

    def test_scheme_only_no_host(self):
        self.assertFalse(check_url("http://"))


class ResolverBlocksRebinding(unittest.TestCase):
    def test_name_resolving_to_loopback_is_blocked(self):
        v = check_url("http://sneaky.example.com/",
                      resolver=lambda h: ["127.0.0.1"])
        self.assertFalse(v)
        self.assertEqual(v.category, "private_host")

    def test_name_resolving_to_metadata_is_blocked(self):
        v = check_url("http://sneaky.example.com/",
                      resolver=lambda h: ["169.254.169.254"])
        self.assertFalse(v)
        self.assertEqual(v.category, "metadata")

    def test_name_resolving_to_public_passes(self):
        self.assertTrue(check_url("http://example.com/",
                                  resolver=lambda h: ["93.184.216.34"]))

    def test_resolver_skipped_when_private_allowed(self):
        # If the operator allows private hosts, the resolver must not override.
        self.assertTrue(check_url("http://example.com/", allow_private=True,
                                  resolver=lambda h: ["127.0.0.1"]))


class BooleanWrapper(unittest.TestCase):
    def test_is_safe_url(self):
        self.assertTrue(is_safe_url("https://example.com"))
        self.assertFalse(is_safe_url("file:///etc/passwd"))


if __name__ == "__main__":
    unittest.main()
