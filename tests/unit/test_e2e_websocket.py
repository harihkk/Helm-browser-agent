"""End-to-end tests through the real /ws/advanced WebSocket handler.

These drive the exact coroutine the live frontend talks to
(``api.main.websocket_endpoint``) with a fake WebSocket transport plus a fake
browser and scripted AI, so the full pipeline - intent, planning, execution,
observation, validation, blockers, and run-id plumbing - is exercised without
a real browser, network, or AI key. (We talk to the handler directly rather
than via Starlette's TestClient to avoid an httpx/starlette version clash.)
"""

import asyncio
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fastapi import WebSocketDisconnect

import api.main as main
from core.task_orchestrator import SophisticatedTaskOrchestrator
from core.session_recorder import SessionRecorder
from tests.unit.fakes import FakeBrowser, ScriptedAI, page

TERMINAL = ("task_completed", "task_failed", "task_cancelled")


class FakeWebSocket:
    """Minimal ASGI-WebSocket stand-in feeding one client message."""

    def __init__(self, incoming):
        self._incoming = list(incoming)
        self.sent = []
        self._closed = asyncio.Event()

    async def accept(self):
        pass

    async def receive_text(self):
        if self._incoming:
            return json.dumps(self._incoming.pop(0))
        await self._closed.wait()
        raise WebSocketDisconnect()

    async def send_text(self, text):
        self.sent.append(json.loads(text))

    async def close(self):
        self._closed.set()


def install(browser, ai):
    main.orchestrator = SophisticatedTaskOrchestrator(browser, ai)
    main.session_recorder = SessionRecorder()
    main.db = None
    main.template_engine = None


def run_ws(description, client_run_id="run-1", options=None, timeout=5.0):
    async def go():
        ws = FakeWebSocket([{
            "type": "execute_advanced_task",
            "client_run_id": client_run_id,
            "description": description,
            "options": options or {},
        }])
        endpoint = asyncio.create_task(main.websocket_endpoint(ws))
        loops = int(timeout / 0.005)
        for _ in range(loops):
            if any(s.get("type") in TERMINAL for s in ws.sent):
                break
            await asyncio.sleep(0.005)
        await ws.close()
        try:
            await asyncio.wait_for(endpoint, timeout=2.0)
        except asyncio.TimeoutError:
            endpoint.cancel()
        return ws.sent
    return asyncio.run(go())


class WebSocketPipeline(unittest.TestCase):
    def test_navigation_completes_end_to_end(self):
        install(
            FakeBrowser(routes={"example.com/docs": page("https://example.com/docs", "Docs", "Documentation")}),
            ScriptedAI([
                {"action": "navigate", "parameters": {"url": "https://example.com/docs"}},
                {"action": "done", "task_complete": True, "parameters": {"summary": "opened docs"}},
            ]),
        )
        events = run_ws("open https://example.com/docs", client_run_id="run-nav")
        terminal = [e for e in events if e.get("type") in TERMINAL][-1]
        self.assertEqual(terminal["type"], "task_completed")
        self.assertEqual(terminal["status"], "completed")
        self.assertIn("validation", terminal)
        self.assertTrue(all(e.get("client_run_id") == "run-nav" for e in events))

    def test_404_surfaces_structured_blocker_end_to_end(self):
        install(
            FakeBrowser(routes={"gone": page("https://x.com/gone", "404 Not Found", "Page Not Found")}),
            ScriptedAI([
                {"action": "navigate", "parameters": {"url": "https://x.com/gone"}},
                {"action": "done", "task_complete": True, "parameters": {"summary": "opened"}},
            ]),
        )
        events = run_ws("open https://x.com/gone", client_run_id="run-404", options={"max_steps": 3})
        terminal = [e for e in events if e.get("type") in TERMINAL][-1]
        self.assertEqual(terminal["type"], "task_failed")
        self.assertEqual(terminal["status"], "blocked")
        self.assertEqual(terminal["blocker_type"], "page_not_found")
        self.assertTrue(terminal["suggested_next_step"])
        self.assertTrue(all(e.get("client_run_id") == "run-404" for e in events))

    def test_confirmation_required_blocks_high_impact_end_to_end(self):
        def responder(action, params, state):
            if action == "navigate":
                return {"success": True, "action": "navigate"}, page(
                    "https://www.amazon.com/s?k=airpods", "airpods", "results")
            return {"success": True, "action": action}, None
        install(
            FakeBrowser(responder=responder),
            ScriptedAI([
                {"action": "navigate", "parameters": {"url": "https://www.amazon.com/s?k=airpods"}},
                {"action": "add_amazon_item_to_cart", "parameters": {"query": "airpods"}},
            ]),
        )
        events = run_ws("go to amazon and add airpods to cart", client_run_id="run-cart",
                        options={"max_steps": 5})
        terminal = [e for e in events if e.get("type") in TERMINAL][-1]
        self.assertEqual(terminal["status"], "blocked")
        self.assertEqual(terminal["blocker_type"], "confirmation_required")

    def test_run_id_is_echoed_on_every_event(self):
        install(
            FakeBrowser(routes={"example.com": page("https://example.com", "Example", "content")}),
            ScriptedAI([
                {"action": "navigate", "parameters": {"url": "https://example.com"}},
                {"action": "done", "task_complete": True, "parameters": {"summary": "ok"}},
            ]),
        )
        events = run_ws("open https://example.com", client_run_id="alpha")
        self.assertTrue(events)
        self.assertTrue(all(e.get("client_run_id") == "alpha" for e in events))
        self.assertFalse(any(e.get("client_run_id") == "beta" for e in events))


if __name__ == "__main__":
    unittest.main(verbosity=2)
