"""Smoke tests - keep fast and free of network/browser dependencies.

Run with:  python -m pytest tests/unit -q
Or:        python tests/unit/test_smoke.py
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


class ImportSmoke(unittest.TestCase):
    def test_core_modules_import(self):
        import core.ai_agent
        import core.browser_engine
        import core.task_orchestrator
        import core.task_templates
        import core.workflow_engine
        import core.session_recorder
        import core.scheduler
        import core.data_extractor
        import database.db
        import api.main  # noqa: F401


class SchedulerIntervalParsing(unittest.TestCase):
    def test_parse_simple_interval(self):
        from core.scheduler import parse_simple_interval
        self.assertEqual(parse_simple_interval('5m'), 300)
        self.assertEqual(parse_simple_interval('1h'), 3600)
        self.assertEqual(parse_simple_interval('2d'), 2 * 86400)
        self.assertEqual(parse_simple_interval('30s'), 30)
        self.assertEqual(parse_simple_interval('10'), 600)  # bare -> minutes
        self.assertIsNone(parse_simple_interval('not-an-interval'))


class JsonParsing(unittest.TestCase):
    def test_parse_direct_and_fenced(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass
        a = Stub()
        self.assertEqual(a._parse_json('{"action":"done"}')['action'], 'done')
        fenced = 'prose\n```json\n{"action":"click","confidence":0.9}\n```\ntrailer'
        self.assertEqual(a._parse_json(fenced)['action'], 'click')
        brace = 'garbage before {"a":1, "b":{"c":2}} garbage after'
        self.assertEqual(a._parse_json(brace)['a'], 1)
        self.assertIn('error', a._parse_json('not json at all'))


class ProviderConfiguration(unittest.TestCase):
    def test_ollama_can_be_primary_without_groq_key(self):
        from core.ai_agent import GroqAIAgent

        agent = GroqAIAgent(
            api_key="",
            ollama_url="http://localhost:11434",
            ollama_model="llama3.1",
        )

        self.assertFalse(agent._groq_enabled)
        self.assertIsNone(agent.client)
        self.assertEqual(agent._ollama_model, "llama3.1")

    def test_requires_at_least_one_provider(self):
        from core.ai_agent import GroqAIAgent

        with self.assertRaises(ValueError):
            GroqAIAgent(api_key="")

    def test_provider_mode_switches_to_local_when_ollama_configured(self):
        from core.ai_agent import GroqAIAgent

        agent = GroqAIAgent(
            api_key="",
            ollama_url="http://localhost:11434",
            ollama_model="llama3.1",
        )

        result = agent.set_provider_mode("local")
        self.assertTrue(result["success"])
        self.assertEqual(agent.provider_mode, "local")

    def test_provider_mode_rejects_local_without_ollama(self):
        from core.ai_agent import GroqAIAgent

        agent = GroqAIAgent(api_key="", gemini_api_key="test-key")

        result = agent.set_provider_mode("local")
        self.assertFalse(result["success"])
        self.assertEqual(agent.provider_mode, "api")


class QuickActionRouting(unittest.TestCase):
    def test_structured_intent_extracts_media_query_not_full_prompt(self):
        from core.intent_planner import IntentPlanner

        intent = IntentPlanner().parse_intent(
            "go to youtube and search for drop dead by olivia rodrigo and actually play that song"
        )
        self.assertEqual(intent.task_type, "media_playback")
        self.assertEqual(intent.target_site, "youtube.com")
        self.assertEqual(intent.search_query, "drop dead by olivia rodrigo")
        self.assertIn("playback", intent.success_condition.lower())
        self.assertNotIn("go to youtube", intent.search_query)

    def test_structured_intent_extracts_note_content_not_command(self):
        from core.intent_planner import IntentPlanner

        intent = IntentPlanner().parse_intent("write buy milk tomorrow in google keep")
        self.assertEqual(intent.task_type, "note_creation")
        self.assertEqual(intent.target_app, "google_keep")
        self.assertEqual(intent.content_to_type, "buy milk tomorrow")
        self.assertNotIn("google keep", intent.content_to_type)

    def test_structured_intent_extracts_repo_search_scope(self):
        from core.intent_planner import IntentPlanner

        intent = IntentPlanner().parse_intent(
            "search for auth logic inside https://github.com/harihkk/helm"
        )
        self.assertEqual(intent.task_type, "repo_search")
        self.assertEqual(intent.target_url, "https://github.com/harihkk/helm")
        self.assertEqual(intent.entity_or_object, "harihkk/helm")
        self.assertEqual(intent.search_query, "auth logic")

    def test_structured_intent_extracts_product_constraints(self):
        from core.intent_planner import IntentPlanner

        intent = IntentPlanner().parse_intent(
            "go to apply website and look for iphone plus 512 gb variant and sow em the final price"
        )
        self.assertEqual(intent.task_type, "product_configuration")
        self.assertEqual(intent.target_site, "apple.com")
        self.assertEqual(intent.constraints["storage"], "512GB")
        self.assertIn("iPhone 16 Plus", intent.entity_or_object)

    def test_amazon_ipad_prompt_extracts_exact_product_and_reviews(self):
        from core.intent_planner import IntentPlanner

        intent = IntentPlanner().parse_intent(
            "go to amazon and add ipad pro m4 base version and open reviews"
        )
        self.assertEqual(intent.task_type, "cart_update")
        self.assertEqual(intent.target_site, "amazon.com")
        self.assertEqual(intent.search_query, "Apple iPad Pro M4 256GB Wi-Fi")
        self.assertEqual(intent.constraints["core_product"], "ipad pro")
        self.assertEqual(intent.constraints["model_generation"], "M4")
        self.assertTrue(intent.constraints["base_version"])
        self.assertTrue(intent.constraints["open_reviews"])
        self.assertTrue(intent.constraints["reject_accessories"])

    def test_generic_fallback_has_success_condition_and_supported_action(self):
        from core.action_registry import is_supported_action
        from core.intent_planner import IntentPlanner

        action = IntentPlanner().plan(
            "find docs about browser automation reliability",
            {"url": "about:blank"},
        )
        self.assertTrue(is_supported_action(action["action"]))
        self.assertTrue(action["success_condition"])
        self.assertIsNotNone(action["intent"])

    def test_planner_never_emits_unregistered_action_for_categories(self):
        from core.action_registry import is_supported_action
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        prompts = [
            "open https://example.com",
            "search the web for playwright locators",
            "write buy milk tomorrow in google keep",
            "search for auth logic inside https://github.com/harihkk/helm",
            "go to amazon and add some flowers to cart",
            "go to youtube and search for lofi music and actually play it",
            "open apple website and look for iphone 16 256 gb price",
            "go to wikipedia and search machine learning",
        ]
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                action = planner.plan(prompt, {"url": "about:blank"})
                self.assertTrue(is_supported_action(action["action"]))
                self.assertTrue(action["success_condition"])

    def test_intent_planner_handles_common_browser_routes(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()

        direct = planner.plan("open https://example.com/docs", {"url": "about:blank"})
        self.assertEqual(direct["action"], "navigate")
        self.assertEqual(direct["parameters"]["url"], "https://example.com/docs")

        search = planner.plan("search google for playwright selectors", {"url": "about:blank"})
        self.assertEqual(search["action"], "navigate")
        self.assertIn("playwright+selectors", search["parameters"]["url"])

    def test_youtube_search_uses_direct_site_search(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "go to youtube and search for openai dev day keynote",
            {"url": "about:blank"},
        )
        self.assertEqual(action["action"], "navigate")
        self.assertEqual(
            action["parameters"]["url"],
            "https://www.youtube.com/results?search_query=openai+dev+day+keynote",
        )

    def test_youtube_play_prompt_extracts_media_query_and_opens_video(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        first = planner.plan(
            "cna u go to youtube and search for drop dead by olivia rodrigo and actually play that song",
            {"url": "about:blank"},
        )
        self.assertEqual(first["action"], "navigate")
        self.assertEqual(
            first["parameters"]["url"],
            "https://www.youtube.com/results?search_query=drop+dead+by+olivia+rodrigo",
        )

        second = planner.plan(
            "cna u go to youtube and search for drop dead by olivia rodrigo and actually play that song",
            {"url": "https://www.youtube.com/results?search_query=drop+dead+by+olivia+rodrigo"},
        )
        self.assertEqual(second["action"], "play_youtube_result")
        self.assertEqual(second["parameters"]["query"], "drop dead by olivia rodrigo")

        third = planner.plan(
            "cna u go to youtube and search for drop dead by olivia rodrigo and actually play that song",
            {"url": "https://www.youtube.com/watch?v=abc123"},
            [{"action": "play_youtube_result", "success": True}],
        )
        self.assertEqual(third["action"], "ensure_youtube_playback")

    def test_reddit_search_uses_direct_site_search_even_with_typos(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "got o reddit.com an dsearch for netherlnads carrots season",
            {"url": "about:blank"},
        )
        self.assertEqual(action["action"], "navigate")
        self.assertEqual(
            action["parameters"]["url"],
            "https://www.reddit.com/search/?q=netherlnads+carrots+season",
        )

    def test_reddit_search_finishes_after_extract(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "got o reddit.com an dsearch for netherlnads carrots season",
            {"url": "https://www.reddit.com/search/?q=netherlnads+carrots+season"},
            [{"action": "extract", "success": True}],
        )
        self.assertEqual(action["action"], "done")
        self.assertTrue(action["task_complete"])

    def test_generic_site_search_opens_first_result_when_requested(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "go to example.com and open the best page about pricing",
            {"url": "https://www.google.com/search?q=site%3Aexample.com+pricing"},
        )
        self.assertEqual(action["action"], "open_first_search_result")
        self.assertEqual(action["parameters"]["domain"], "example.com")

    def test_direct_site_search_opens_first_result_when_requested(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "go to reddit and open the best post about local llms",
            {"url": "https://www.reddit.com/search/?q=local+llms"},
        )
        self.assertEqual(action["action"], "open_first_search_result")
        self.assertEqual(action["parameters"]["domain"], "reddit.com")

    def test_apple_iphone_storage_price_routes_to_configurator(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        first = planner.plan(
            "open apple website and look for iphone 16 and show me the 256 gb config model price",
            {"url": "about:blank"},
        )
        self.assertEqual(first["action"], "navigate")
        self.assertEqual(
            first["parameters"]["url"],
            "https://www.apple.com/shop/buy-iphone/iphone-16",
        )

        second = planner.plan(
            "open apple website and look for iphone 16 and show me the 256 gb config model price",
            {"url": "https://www.apple.com/shop/buy-iphone/iphone-16"},
        )
        self.assertEqual(second["action"], "configure_apple_product")
        self.assertEqual(second["parameters"]["model"], "iPhone 16")
        self.assertEqual(second["parameters"]["storage"], "256GB")

    def test_apple_iphone_plus_uses_shared_family_buy_page(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        first = planner.plan(
            "go to apply website and look for iphone plus 512 gb variant and sow em the final price",
            {"url": "about:blank"},
        )
        self.assertEqual(first["action"], "navigate")
        self.assertEqual(
            first["parameters"]["url"],
            "https://www.apple.com/shop/buy-iphone/iphone-16",
        )

        second = planner.plan(
            "go to apple website and look for iphone plus 512 gb variant and show me the final price",
            {"url": "https://www.apple.com/shop/buy-iphone/iphone-16"},
        )
        self.assertEqual(second["action"], "configure_apple_product")
        self.assertEqual(second["parameters"]["model"], "iPhone 16 Plus")
        self.assertEqual(second["parameters"]["storage"], "512GB")

    def test_apple_page_not_found_recovers_to_correct_buy_page(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "go to apple website and look for iphone 16 plus 512 gb variant and show me the price",
            {
                "url": "https://www.apple.com/shop/buy-iphone/iphone-16-plus",
                "title": "Page Not Found",
                "content": "Page Not Found Apple Store footer",
            },
        )
        self.assertEqual(action["action"], "navigate")
        self.assertEqual(
            action["parameters"]["url"],
            "https://www.apple.com/shop/buy-iphone/iphone-16",
        )

    def test_apple_visible_price_completes_task(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "open apple website and look for iphone 16 and show me the 256 gb config model price",
            {
                "url": "https://www.apple.com/shop/buy-iphone/iphone-16",
                "title": "Buy iPhone 16 and iPhone 16 Plus - Apple",
                "content": (
                    "Model. Which is best for you? iPhone 16 From $699 "
                    "Storage. How much space do you need? "
                    "128 GB From $699 or $29.12/mo. for 24 mo. "
                    "256 GB From $899 or $37.45/mo. for 24 mo."
                ),
            },
            [{"action": "extract", "success": True}],
        )
        self.assertEqual(action["action"], "done")
        self.assertIn("iPhone 16 256GB: $899", action["parameters"]["summary"])

    def test_amazon_add_to_cart_routes_to_search_then_cart_action(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        first = planner.plan(
            "go to amazon and add soem flowers to cart",
            {"url": "about:blank"},
        )
        self.assertEqual(first["action"], "navigate")
        self.assertEqual(
            first["parameters"]["url"],
            "https://www.amazon.com/s?k=flowers",
        )

        second = planner.plan(
            "go to amazon and add soem flowers to cart",
            {"url": "https://www.amazon.com/s?k=flowers"},
        )
        self.assertEqual(second["action"], "add_amazon_item_to_cart")
        self.assertEqual(second["parameters"]["query"], "flowers")

    def test_amazon_add_ipad_routes_with_reviews_constraint(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        first = planner.plan(
            "go to amazon and add ipad pro m4 base version and open reviews",
            {"url": "about:blank"},
        )
        self.assertEqual(first["action"], "navigate")
        self.assertEqual(
            first["parameters"]["url"],
            "https://www.amazon.com/s?k=Apple+iPad+Pro+M4+256GB+Wi-Fi",
        )

        second = planner.plan(
            "go to amazon and add ipad pro m4 base version and open reviews",
            {"url": "https://www.amazon.com/s?k=Apple+iPad+Pro+M4+256GB+Wi-Fi"},
        )
        self.assertEqual(second["action"], "add_amazon_item_to_cart")
        self.assertEqual(second["parameters"]["query"], "Apple iPad Pro M4 256GB Wi-Fi")
        self.assertTrue(second["parameters"]["constraints"]["open_reviews"])

    def test_google_keep_note_routes_to_keep_then_write_action(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        first = planner.plan(
            "write buy milk tomorrow in google keep",
            {"url": "about:blank"},
        )
        self.assertEqual(first["action"], "navigate")
        self.assertEqual(first["parameters"]["url"], "https://keep.google.com/")

        second = planner.plan(
            "write buy milk tomorrow in google keep",
            {"url": "https://keep.google.com/"},
        )
        self.assertEqual(second["action"], "write_google_keep_note")
        self.assertEqual(second["parameters"]["text"], "buy milk tomorrow")

        redirected = planner.plan(
            "write buy milk tomorrow in google keep",
            {"url": "https://accounts.google.com/signin/v2/identifier?service=keep&continue=https%3A%2F%2Fkeep.google.com"},
            [{"action": "navigate", "success": True}],
        )
        self.assertEqual(redirected["action"], "write_google_keep_note")

    def test_github_repo_search_routes_to_code_search(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        first = planner.plan(
            "search for browser engine inside https://github.com/harihkk/helm-agent repo",
            {"url": "about:blank"},
        )
        self.assertEqual(first["action"], "navigate")
        self.assertEqual(
            first["parameters"]["url"],
            "https://github.com/harihkk/helm-agent/search?q=browser+engine&type=code",
        )

        second = planner.plan(
            "open first result for browser engine inside https://github.com/harihkk/helm-agent repo",
            {"url": "https://github.com/harihkk/helm-agent/search?q=browser+engine&type=code"},
        )
        self.assertEqual(second["action"], "open_first_github_code_result")

        global_search = planner.plan(
            "open first result for browser engine inside https://github.com/harihkk/helm-agent repo",
            {"url": "https://github.com/search?q=repo%3Aharihkk%2Fhelm-agent+browser+engine&type=code"},
        )
        self.assertEqual(global_search["action"], "open_first_github_code_result")

    def test_linkedin_recruiter_search_routes_to_people_search(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "can u go over to linkedin and look for google recruiters",
            {"url": "about:blank"},
        )
        self.assertEqual(action["action"], "navigate")
        self.assertEqual(
            action["parameters"]["url"],
            "https://www.linkedin.com/search/results/people/?keywords=google+recruiters",
        )

    def test_github_profile_routes_directly(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass

        agent = Stub()
        action = agent._quick_action(
            "go to github and look for ajay6601 profile",
            {"url": "about:blank", "title": "", "content": ""},
            {"action_history": []},
        )
        self.assertEqual(action["action"], "navigate")
        self.assertEqual(action["parameters"]["url"], "https://github.com/ajay6601")

    def test_github_user_top_repository_routes_to_sorted_repos(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "go to github.com and look for ajay6601 user and open his top repository",
            {"url": "about:blank"},
        )
        self.assertEqual(action["action"], "navigate")
        self.assertEqual(
            action["parameters"]["url"],
            "https://github.com/ajay6601?tab=repositories&sort=stargazers",
        )

    def test_github_user_top_repository_opens_first_sorted_repo(self):
        from core.intent_planner import IntentPlanner

        planner = IntentPlanner()
        action = planner.plan(
            "go to github.com and look for ajay6601 user and open his top repository",
            {"url": "https://github.com/ajay6601?tab=repositories&sort=stargazers"},
        )
        self.assertEqual(action["action"], "open_top_github_repo")
        self.assertEqual(action["parameters"]["user"], "ajay6601")

    def test_linkedin_auth_wall_is_reported_as_blocked(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass

        agent = Stub()
        action = agent._blocked_site_action(
            "can u go over to linkedin and look for google recruiters",
            {
                "url": "https://www.linkedin.com/search/results/people/?keywords=google+recruiters",
                "title": "LinkedIn Login",
                "content": "LinkedIn Sign in Continue with Google Join now",
            },
            {"action_history": [{"action": "navigate", "success": True}]},
        )
        self.assertEqual(action["error"], "site_requires_sign_in")

    def test_google_recaptcha_is_reported_as_blocked(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass

        agent = Stub()
        action = agent._blocked_site_action(
            "search google for reddit carrots",
            {
                "url": "https://www.google.com/search?q=site%3Areddit.com+carrots",
                "title": "About this page",
                "content": "Our systems have detected unusual traffic. I'm not a robot reCAPTCHA",
            },
            {"action_history": [{"action": "navigate", "success": True}]},
        )
        self.assertEqual(action["error"], "site_blocked_by_bot_check")

    def test_github_profile_finishes_on_matching_url(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass

        agent = Stub()
        action = agent._quick_action(
            "go to github and look for ajay6601 profile",
            {"url": "https://github.com/ajay6601", "title": "ajay6601", "content": "Overview"},
            {"action_history": [{"action": "navigate", "success": True}]},
        )
        self.assertEqual(action["action"], "done")
        self.assertTrue(action["task_complete"])

    def test_github_profile_and_repository_routes_to_repo(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass

        agent = Stub()
        action = agent._quick_action(
            "go to github and search for harihkk profile and open his code review arena repository",
            {"url": "https://github.com/repository", "title": "repository", "content": ""},
            {"action_history": [{"action": "navigate", "success": True}]},
        )
        self.assertEqual(action["action"], "navigate")
        self.assertEqual(
            action["parameters"]["url"],
            "https://github.com/harihkk/code-review-arena",
        )

    def test_github_repository_finishes_on_matching_url(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass

        agent = Stub()
        action = agent._quick_action(
            "go to github and search for harihkk profile and open his code review arena repository",
            {"url": "https://github.com/harihkk/code-review-arena", "title": "code-review-arena", "content": "Code"},
            {"action_history": [{"action": "navigate", "success": True}]},
        )
        self.assertEqual(action["action"], "done")
        self.assertTrue(action["task_complete"])


class LoopDetection(unittest.TestCase):
    def test_loop_detection(self):
        from core.task_orchestrator import SophisticatedTaskOrchestrator

        class Stub(SophisticatedTaskOrchestrator):
            def __init__(self):
                pass
        o = Stub()

        # Short history: no loop
        self.assertFalse(o._detect_loop(['click']))
        self.assertFalse(o._detect_loop(['click', 'type']))

        # Three legit typing steps in a row must NOT trip
        self.assertFalse(o._detect_loop(['type', 'type', 'type']))

        # Three scrolls in a row = idle loop
        self.assertTrue(o._detect_loop(['scroll', 'scroll', 'scroll']))

        # scroll/wait alternating = loop
        self.assertTrue(o._detect_loop(
            ['scroll', 'wait', 'scroll', 'wait']))

        # Same failed click with identical params 3x = broken selector loop.
        history = [{'action': 'click', 'parameters': {'selector': '#x'},
                    'success': False}] * 3
        self.assertTrue(o._detect_loop(['click', 'click', 'click'], history))

        # Successful repeat-looking actions are common on dynamic sites; the
        # planner/state should decide the next move instead of pre-failing.
        history = [{'action': 'click', 'parameters': {'selector': '#x'},
                    'success': True}] * 3
        self.assertFalse(o._detect_loop(['click', 'click', 'click'], history))

        # Three same-named actions with DIFFERENT params is NOT a loop
        history = [
            {'action': 'type', 'parameters': {'selector': '#a', 'text': 'foo'}, 'success': True},
            {'action': 'type', 'parameters': {'selector': '#b', 'text': 'bar'}, 'success': True},
            {'action': 'type', 'parameters': {'selector': '#c', 'text': 'baz'}, 'success': True},
        ]
        self.assertFalse(o._detect_loop(['type', 'type', 'type'], history))

    def test_fallback_search_text_does_not_type_full_instruction(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass
        agent = Stub()
        query = agent._clean_search_text(
            "cna u go to youtube and search for drop dead by olivia rodrigo and actually play that song"
        )
        self.assertEqual(query, "drop dead by olivia rodrigo")


class ActionRegistryAndGuards(unittest.TestCase):
    def test_registry_contains_required_public_actions(self):
        from core.action_registry import is_supported_action

        required = [
            "navigate", "search_web", "site_search", "click", "type",
            "press_key", "wait_for_selector", "extract_text", "select_option",
            "observe_page", "validate_url", "validate_text_visible",
            "validate_media_playing", "validate_note_created",
            "validate_product_configured", "validate_cart_updated",
            "recover_from_error_page", "report_blocker",
        ]
        for name in required:
            with self.subTest(action=name):
                self.assertTrue(is_supported_action(name))

    def test_text_guard_rejects_full_commands(self):
        from core.action_registry import UnsafeTextPayloadError, validate_text_payload

        validate_text_payload("drop dead olivia rodrigo")
        validate_text_payload("buy milk tomorrow")
        with self.assertRaises(UnsafeTextPayloadError):
            validate_text_payload(
                "go to youtube and search for drop dead by olivia rodrigo and actually play that song"
            )

    def test_browser_engine_rejects_unsupported_action_before_page_use(self):
        from core.browser_engine import AdvancedBrowserEngine

        engine = AdvancedBrowserEngine()
        result = asyncio.run(engine.execute_action("default", "launch_spaceship", {}))
        self.assertFalse(result["success"])
        self.assertIn("Unsupported browser action", result["error"])

    def test_amazon_candidate_scoring_rejects_ipad_accessories(self):
        from core.browser_engine import AdvancedBrowserEngine

        engine = AdvancedBrowserEngine()
        constraints = {
            "core_product": "ipad pro",
            "brand": "apple",
            "model_generation": "M4",
            "reject_accessories": True,
            "required_terms": ["apple", "ipad", "pro", "m4"],
        }
        accessory = {
            "title": "Paperlike Screen Protector for iPad Pro 13 M4",
            "text": "iPad Pro 13 M4/M3 screen protector accessory",
            "url": "https://www.amazon.com/dp/ACCESSORY1",
        }
        actual = {
            "title": "Apple iPad Pro 13-Inch (M4): Built for Apple Intelligence, 256GB, Wi-Fi",
            "text": "Apple iPad Pro M4 256GB Wi-Fi tablet",
            "url": "https://www.amazon.com/dp/IPADPRO123",
        }

        self.assertLess(
            engine._score_amazon_candidate(accessory, "Apple iPad Pro M4 256GB Wi-Fi", constraints)["score"],
            0,
        )
        self.assertGreater(
            engine._score_amazon_candidate(actual, "Apple iPad Pro M4 256GB Wi-Fi", constraints)["score"],
            0,
        )


class CompletionAndBlockerValidation(unittest.TestCase):
    def test_unvalidated_done_is_rejected_for_navigation(self):
        from core.intent_planner import IntentPlanner
        from core.task_orchestrator import AdvancedTask, SophisticatedTaskOrchestrator
        from core.browser_engine import PageState

        class Agent:
            intent_planner = IntentPlanner()

        class Stub(SophisticatedTaskOrchestrator):
            def __init__(self):
                self.ai_agent = Agent()

        task = AdvancedTask("t1", "open https://example.com/docs", {})
        ok, reason = Stub()._validate_completion(
            task,
            PageState("https://example.com/other", "Example", "", []),
            {"action": "done", "parameters": {"summary": "done"}},
        )
        self.assertFalse(ok)
        self.assertIn("Expected current URL", reason)

    def test_amazon_cart_completion_requires_product_match_and_reviews(self):
        from core.intent_planner import IntentPlanner
        from core.task_orchestrator import AdvancedTask, SophisticatedTaskOrchestrator
        from core.browser_engine import PageState

        class Agent:
            intent_planner = IntentPlanner()

        class Stub(SophisticatedTaskOrchestrator):
            def __init__(self):
                self.ai_agent = Agent()

        task = AdvancedTask(
            "t2",
            "go to amazon and add ipad pro m4 base version and open reviews",
            {},
        )
        task.context["action_history"].append({
            "action": "add_amazon_item_to_cart",
            "success": True,
            "data": {
                "product_match": True,
                "cart_confirmed": True,
                "reviews_opened": False,
                "title": "Apple iPad Pro 13-Inch (M4)",
            },
        })
        ok, reason = Stub()._validate_completion(
            task,
            PageState("https://www.amazon.com/dp/IPADPRO123", "Apple iPad Pro", "Added to cart", []),
            {"action": "done", "parameters": {"summary": "done"}},
        )
        self.assertFalse(ok)
        self.assertIn("reviews", reason.lower())

        task.context["action_history"][-1]["data"]["reviews_opened"] = True
        ok, _ = Stub()._validate_completion(
            task,
            PageState("https://www.amazon.com/product-reviews/IPADPRO123", "Reviews", "Customer reviews", []),
            {"action": "done", "parameters": {"summary": "done"}},
        )
        self.assertTrue(ok)

    def test_blocker_classifier_identifies_login_captcha_404(self):
        from core.task_orchestrator import SophisticatedTaskOrchestrator
        from core.browser_engine import PageState

        class Stub(SophisticatedTaskOrchestrator):
            def __init__(self):
                pass

        o = Stub()
        self.assertEqual(
            o._classify_blocker(PageState("https://x", "Login", "Sign in required", []))["blocker_type"],
            "login_required",
        )
        self.assertEqual(
            o._classify_blocker(PageState("https://x", "About this page", "reCAPTCHA unusual traffic", []))["blocker_type"],
            "captcha_or_bot_protection",
        )
        self.assertEqual(
            o._classify_blocker(PageState("https://x", "Page Not Found", "404", []))["blocker_type"],
            "page_not_found",
        )


class FrontendRunState(unittest.TestCase):
    def test_frontend_uses_client_run_ids_to_ignore_stale_events(self):
        html_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        self.assertIn("activeClientRunId", html)
        self.assertIn("client_run_id:activeClientRunId", html)
        self.assertIn("d.client_run_id!==activeClientRunId", html)
        self.assertIn("resetPreview();", html)

    def test_frontend_renders_validation_evidence_and_structured_blockers(self):
        html_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        # Completed runs surface validation evidence.
        self.assertIn("d.validation", html)
        # Terminal blockers are rendered with type + suggested next step, and
        # blocked/unverified are distinguished from a hard failure.
        self.assertIn("showTerminalBlocker", html)
        self.assertIn("suggested_next_step", html)
        self.assertIn("blocker_type", html)
        self.assertIn("Unverified", html)
        self.assertIn("Blocked", html)


class WebSocketRunId(unittest.TestCase):
    def test_websocket_path_echoes_client_run_id_contract(self):
        api_path = os.path.join(os.path.dirname(__file__), "..", "..", "api", "main.py")
        with open(api_path, "r", encoding="utf-8") as f:
            source = f.read()
        self.assertIn('@app.websocket("/ws/advanced")', source)
        self.assertIn("client_run_id", source)
        self.assertIn("message.get('client_run_id'", source)
        self.assertIn("update['client_run_id'] = client_run_id", source)


class PythonExportEscaping(unittest.TestCase):
    def test_export_handles_quotes_and_backslashes(self):
        from core.session_recorder import SessionRecorder
        rec = {
            'name': 'weird',
            'steps': [
                {'action': 'navigate',
                 'parameters': {'url': 'https://example.com/?a="b"&c=\\d'}},
                {'action': 'type',
                 'parameters': {'selector': '#q',
                                'text': 'hello "world"\nnew line'}},
            ],
        }
        src = SessionRecorder().export_as_python(rec)
        # Must be syntactically valid Python
        compile(src, '<generated>', 'exec')


class DatabaseCreate(unittest.IsolatedAsyncioTestCase):
    async def test_init_and_seed(self):
        from database.db import Database
        tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        tmp.close()
        try:
            db = Database(tmp.name)
            await db.init()
            templates = await db.get_templates()
            self.assertGreater(len(templates), 0)
            analytics = await db.get_analytics()
            self.assertEqual(analytics['total_tasks'], 0)
            await db.close()
        finally:
            os.unlink(tmp.name)


class RetryAfterParsing(unittest.TestCase):
    def test_parse_retry_after_from_message(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass
        a = Stub()

        e1 = Exception("Rate limit reached. Please try again in 6.2s.")
        self.assertAlmostEqual(a._parse_retry_after(e1), 6.2, places=2)

        e2 = Exception("429 rate_limit_exceeded. Retry-After: 12")
        self.assertAlmostEqual(a._parse_retry_after(e2), 12.0, places=2)

        e3 = Exception("please try again in 450ms")
        self.assertAlmostEqual(a._parse_retry_after(e3), 0.45, places=2)

        e4 = Exception("some other error")
        self.assertIsNone(a._parse_retry_after(e4))

    def test_parse_retry_after_from_headers(self):
        from core.ai_agent import GroqAIAgent

        class Stub(GroqAIAgent):
            def __init__(self):
                pass

        class FakeResp:
            headers = {'retry-after': '3.5'}

        err = Exception('429')
        err.response = FakeResp()
        self.assertAlmostEqual(Stub()._parse_retry_after(err), 3.5, places=2)


class TemplateVarResolution(unittest.TestCase):
    def test_resolve_nested(self):
        from core.task_templates import TemplateEngine

        class OrchStub:
            browser = None
        t = TemplateEngine(OrchStub())
        resolved = t._resolve_variables(
            [{'action': 'type',
              'parameters': {'selector': '#q', 'text': '{query}'}}],
            {'query': 'hello'})
        self.assertEqual(resolved[0]['parameters']['text'], 'hello')

    def test_resolve_query_variables_in_urls(self):
        from core.task_templates import TemplateEngine

        class OrchStub:
            browser = None
        t = TemplateEngine(OrchStub())
        resolved = t._resolve_variables(
            [{'action': 'navigate',
              'parameters': {'url': 'https://www.google.com/search?q={query}'}}],
            {'query': 'hello world'})
        self.assertEqual(
            resolved[0]['parameters']['url'],
            'https://www.google.com/search?q=hello+world')


if __name__ == '__main__':
    unittest.main(verbosity=2)
