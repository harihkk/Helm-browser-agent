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

        # Same click with identical params 3x = loop (whether failing or
        # succeeding - both indicate the agent isn't making progress)
        for ok in (True, False):
            history = [{'action': 'click', 'parameters': {'selector': '#x'},
                        'success': ok}] * 3
            self.assertTrue(o._detect_loop(['click', 'click', 'click'], history),
                            f"should detect loop with success={ok}")

        # Three same-named actions with DIFFERENT params is NOT a loop
        history = [
            {'action': 'type', 'parameters': {'selector': '#a', 'text': 'foo'}, 'success': True},
            {'action': 'type', 'parameters': {'selector': '#b', 'text': 'bar'}, 'success': True},
            {'action': 'type', 'parameters': {'selector': '#c', 'text': 'baz'}, 'success': True},
        ]
        self.assertFalse(o._detect_loop(['type', 'type', 'type'], history))


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
