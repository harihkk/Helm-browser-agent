"""Live HTTP + WebSocket verification against a real running server.

Starts the real uvicorn server (lifespan disabled so it does not try to launch
a real Chromium / require an AI key) with a deterministic fake browser + scripted
AI injected, then exercises the exact endpoints the frontend uses:

  * GET /                -> the real frontend HTML
  * GET /api/status      -> health JSON
  * ws://.../ws/advanced -> full task runs (completed / blocked / confirmation)

This is the closest faithful "live frontend/WebSocket path" check possible in
an offline CI environment. Run:  python tests/live_check.py
"""

import asyncio
import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import httpx
import uvicorn
import websockets

import api.main as main
from core.task_orchestrator import SophisticatedTaskOrchestrator
from core.session_recorder import SessionRecorder
from tests.unit.fakes import FakeBrowser, ScriptedAI, page


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def install(browser, ai):
    main.orchestrator = SophisticatedTaskOrchestrator(browser, ai)
    main.session_recorder = SessionRecorder()
    main.db = None
    main.template_engine = None


async def run_scenario(port, description, run_id, options=None):
    uri = f"ws://127.0.0.1:{port}/ws/advanced"
    events = []
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "execute_advanced_task",
            "client_run_id": run_id,
            "description": description,
            "options": options or {},
        }))
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=8)
            ev = json.loads(msg)
            events.append(ev)
            if ev.get("type") in ("task_completed", "task_failed", "task_cancelled"):
                break
    return events


def terminal(events):
    return [e for e in events if e.get("type") in ("task_completed", "task_failed")][-1]


async def main_async(port):
    results = []

    def check(name, cond, detail=""):
        results.append((name, cond, detail))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    # ---- HTTP layer ----
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
        r = await c.get("/")
        check("GET / serves frontend HTML",
              r.status_code == 200 and "<html" in r.text.lower(),
              f"status={r.status_code}, {len(r.text)} bytes")
        s = await c.get("/api/status")
        check("GET /api/status responds healthy",
              s.status_code == 200 and s.json().get("status") == "healthy")

    # ---- WS: navigation completes with validation ----
    install(
        FakeBrowser(routes={"example.com/docs": page("https://example.com/docs", "Docs", "Documentation")}),
        ScriptedAI([
            {"action": "navigate", "parameters": {"url": "https://example.com/docs"}},
            {"action": "done", "task_complete": True, "parameters": {"summary": "opened docs"}},
        ]),
    )
    ev = await run_scenario(port, "open https://example.com/docs", "live-nav")
    t = terminal(ev)
    check("WS navigation -> completed", t["type"] == "task_completed" and t["status"] == "completed")
    check("WS completed carries validation evidence",
          bool(t.get("validation", {}).get("reason")), str(t.get("validation")))
    check("WS every event echoes client_run_id",
          all(e.get("client_run_id") == "live-nav" for e in ev))

    # ---- WS: 404 -> structured blocker ----
    install(
        FakeBrowser(routes={"gone": page("https://x.com/gone", "404 Not Found", "Page Not Found")}),
        ScriptedAI([
            {"action": "navigate", "parameters": {"url": "https://x.com/gone"}},
            {"action": "done", "task_complete": True, "parameters": {"summary": "opened"}},
        ]),
    )
    ev = await run_scenario(port, "open https://x.com/gone", "live-404", {"max_steps": 3})
    t = terminal(ev)
    check("WS 404 -> blocked/page_not_found",
          t["status"] == "blocked" and t["blocker_type"] == "page_not_found",
          f"status={t['status']} type={t.get('blocker_type')}")
    check("WS blocker carries suggested_next_step", bool(t.get("suggested_next_step")))

    # ---- WS: high-impact action requires confirmation ----
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
    ev = await run_scenario(port, "go to amazon and add airpods to cart", "live-cart", {"max_steps": 5})
    t = terminal(ev)
    check("WS cart -> confirmation_required (paused)",
          t["status"] == "blocked" and t["blocker_type"] == "confirmation_required")

    passed = sum(1 for _, ok, _ in results if ok)
    print(f"\n  {passed}/{len(results)} live checks passed")
    return all(ok for _, ok, _ in results)


def main_entry():
    port = free_port()
    install(FakeBrowser(), ScriptedAI([]))  # ensure a non-None orchestrator before startup
    config = uvicorn.Config(main.app, host="127.0.0.1", port=port,
                            lifespan="off", log_level="warning", ws="websockets")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # Wait for startup.
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        print("Server failed to start")
        return 1
    print(f"Live server up on 127.0.0.1:{port} (lifespan off, fakes injected)\n")
    try:
        ok = asyncio.run(main_async(port))
    finally:
        server.should_exit = True
        thread.join(timeout=5)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main_entry())
