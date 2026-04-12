"""
Agentic Browser API - FastAPI + WebSocket streaming
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, HTTPException, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

from core.ai_agent import GroqAIAgent
from core.browser_engine import AdvancedBrowserEngine
from core.task_orchestrator import SophisticatedTaskOrchestrator
from core.session_recorder import SessionRecorder
from core.data_extractor import DataExtractor
from core.task_templates import TemplateEngine
from core.workflow_engine import WorkflowEngine
from core.scheduler import TaskScheduler
from database.db import Database

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Models
# ------------------------------------------------------------------ #
class TaskRequest(BaseModel):
    description: str
    options: Dict[str, Any] = {}

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TemplateExecuteRequest(BaseModel):
    variables: Dict[str, str] = {}

class WorkflowCreateRequest(BaseModel):
    name: str
    description: str = ""
    steps: list = []

class ScheduleTaskRequest(BaseModel):
    name: str
    description: str
    interval: str

class HumanInputRequest(BaseModel):
    task_id: str
    input_text: str


# ------------------------------------------------------------------ #
# Globals
# ------------------------------------------------------------------ #
browser_engine: AdvancedBrowserEngine = None
ai_agent: GroqAIAgent = None
orchestrator: SophisticatedTaskOrchestrator = None
db: Database = None
session_recorder: SessionRecorder = None
data_extractor: DataExtractor = None
template_engine: TemplateEngine = None
workflow_engine: WorkflowEngine = None
scheduler: TaskScheduler = None
active_websockets: Dict[str, WebSocket] = {}


# ------------------------------------------------------------------ #
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------ #
@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser_engine, ai_agent, orchestrator, db
    global session_recorder, data_extractor, template_engine, workflow_engine, scheduler

    logger.info("Starting Agentic Browser...")

    # Database
    try:
        db = Database()
        await db.init()
    except Exception as e:
        logger.error(f"Database init failed: {e}")

    # Browser
    try:
        browser_engine = AdvancedBrowserEngine(
            headless=os.getenv("BROWSER_HEADLESS", "false").lower() == "true",
            screenshots_dir="./screenshots",
        )
        await browser_engine.start()
    except Exception as e:
        logger.error(f"Browser engine failed: {e}")
        browser_engine = AdvancedBrowserEngine(headless=True)

    # AI Agent
    groq_key = os.getenv("GROQ_API_KEY", "")
    try:
        if groq_key and groq_key != "your-groq-api-key-here":
            ai_agent = GroqAIAgent(
                api_key=groq_key,
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                eval_model=os.getenv("GROQ_EVAL_MODEL", "llama-3.1-8b-instant"),
                gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
                gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                ollama_url=os.getenv("OLLAMA_BASE_URL", ""),
                ollama_model=os.getenv("OLLAMA_MODEL", ""),
            )
        else:
            logger.warning("No Groq API key - AI disabled. Set GROQ_API_KEY in .env")
    except Exception as e:
        logger.error(f"AI Agent failed: {e}")

    # Orchestrator
    if browser_engine and ai_agent:
        orchestrator = SophisticatedTaskOrchestrator(browser_engine, ai_agent)
        if db:
            orchestrator.set_database(db)

    # Utilities
    session_recorder = SessionRecorder()
    if browser_engine:
        data_extractor = DataExtractor(browser_engine)
    if orchestrator:
        template_engine = TemplateEngine(orchestrator)
        workflow_engine = WorkflowEngine(orchestrator)
        scheduler = TaskScheduler(orchestrator)
        if db:
            scheduler.set_database(db)
            await scheduler.load_from_db()

    logger.info("Agentic Browser started! http://localhost:8000")
    if not ai_agent:
        logger.warning("AI agent not available - set GROQ_API_KEY to enable")

    try:
        yield
    finally:
        logger.info("Shutting down...")
        if scheduler:
            await scheduler.stop_all()
        for ws in list(active_websockets.values()):
            try:
                await ws.close()
            except Exception:
                pass
        if browser_engine:
            await browser_engine.close()
        if db:
            await db.close()


# ------------------------------------------------------------------ #
# App
# ------------------------------------------------------------------ #
app = FastAPI(title="Agentic Browser", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

os.makedirs("./screenshots", exist_ok=True)
app.mount("/screenshots", StaticFiles(directory="screenshots"), name="screenshots")


# ------------------------------------------------------------------ #
# Main WebSocket - task execution
# ------------------------------------------------------------------ #
@app.websocket("/ws/advanced")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"c_{uuid.uuid4().hex[:8]}"
    await websocket.accept()
    active_websockets[client_id] = websocket
    cancel_event = asyncio.Event()
    task_running = asyncio.Event()
    current_task = None

    async def run_task(description: str, options: dict):
        nonlocal current_task
        if not orchestrator:
            try:
                await websocket.send_text(json.dumps({
                    'type': 'task_failed', 'error': 'System not ready. Check your Groq API key.',
                    'task_id': '', 'steps_taken': 0, 'execution_time': 0
                }))
            except Exception:
                pass
            task_running.clear()
            return

        rec_id = session_recorder.start_recording(task_id="pending", name=description[:50])
        try:
            async for update in orchestrator.execute_task_stream(
                description, options, cancel_event=cancel_event
            ):
                if cancel_event.is_set():
                    break

                # Record steps
                if update.get('type') == 'step_executed':
                    session_recorder.record_step(
                        rec_id, update.get('action', ''),
                        update.get('parameters', {}),
                        update.get('success', False), url='')

                # Send to client (handle large screenshots)
                try:
                    msg = json.dumps(update, default=str)
                    await websocket.send_text(msg)
                except Exception:
                    break

                # Screenshots are sent via the main WS step_executed messages only
                # No separate preview forwarding - prevents blinking
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            try:
                await websocket.send_text(json.dumps({
                    'type': 'task_failed', 'error': str(e),
                    'task_id': '', 'steps_taken': 0, 'execution_time': 0
                }))
            except Exception:
                pass
        finally:
            recording = session_recorder.stop_recording(rec_id)
            if recording and db:
                try:
                    await db.save_recording(
                        rec_id, recording['name'], recording['task_id'],
                        json.dumps(recording['steps']), 0)
                except Exception:
                    pass
            cancel_event.clear()
            task_running.clear()

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue

            msg_type = message.get('type', '')

            if msg_type == 'execute_advanced_task':
                if task_running.is_set():
                    await websocket.send_text(json.dumps({
                        'type': 'warning', 'message': 'A task is already running. Stop it first.'
                    }))
                    continue
                task_running.set()
                cancel_event.clear()
                current_task = asyncio.create_task(run_task(
                    message.get('description', ''),
                    message.get('options', {})))

            elif msg_type == 'stop_task':
                cancel_event.set()
                # Also cancel the asyncio task for immediate effect
                if current_task and not current_task.done():
                    current_task.cancel()
                task_running.clear()
                await websocket.send_text(json.dumps({
                    'type': 'task_cancelled', 'message': 'Task cancelled'
                }))

            elif msg_type == 'human_input':
                if orchestrator:
                    await orchestrator.provide_human_input(
                        message.get('task_id', ''), message.get('input_text', ''))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.pop(client_id, None)
        # Clean up if task was running
        cancel_event.set()
        if current_task and not current_task.done():
            current_task.cancel()


# ------------------------------------------------------------------ #
# REST Endpoints
# ------------------------------------------------------------------ #

@app.post("/api/execute-task")
async def execute_task_api(req: TaskRequest):
    if not orchestrator:
        raise HTTPException(503, "System not initialized - check Groq API key")
    result = await orchestrator.execute_advanced_task(req.description, req.options)
    return TaskResponse(
        task_id=result.get('task_id', ''),
        status=result.get('status', 'unknown'),
        message=f"{result.get('steps_taken', 0)} steps, cost: {result.get('cost_summary', '$0')}")


@app.get("/api/status")
async def get_status():
    providers = []
    if ai_agent:
        providers.append({"name": "groq", "model": ai_agent.model, "primary": True})
        if getattr(ai_agent, "_gemini_key", ""):
            providers.append({"name": "gemini", "model": ai_agent._gemini_model,
                              "primary": False})
        if getattr(ai_agent, "_ollama_url", "") and getattr(ai_agent, "_ollama_model", ""):
            providers.append({"name": "ollama", "model": ai_agent._ollama_model,
                              "primary": False, "local": True})
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "browser": browser_engine is not None and browser_engine.is_alive,
            "ai_agent": ai_agent is not None,
            "orchestrator": orchestrator is not None,
            "database": db is not None,
        },
        "providers": providers,
        "groq_stats": ai_agent.get_token_stats() if ai_agent else {},
        "fallback_counts": {
            "gemini": getattr(ai_agent, "_gemini_fallback_count", 0) if ai_agent else 0,
            "ollama": getattr(ai_agent, "_ollama_fallback_count", 0) if ai_agent else 0,
        },
    }


@app.get("/api/metrics")
async def get_metrics():
    if not orchestrator:
        raise HTTPException(503)
    return orchestrator.get_performance_metrics()


@app.get("/api/tasks/history")
async def get_task_history(limit: int = 50, offset: int = 0):
    if db:
        return await db.get_task_history(limit, offset)
    return orchestrator.get_task_history(limit) if orchestrator else []


@app.get("/api/tasks/{task_id}")
async def get_task_detail(task_id: str):
    if not db:
        raise HTTPException(404)
    task = await db.get_task_detail(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return task


@app.get("/api/analytics")
async def get_analytics():
    if not db:
        raise HTTPException(503)
    return await db.get_analytics()


@app.get("/api/templates")
async def get_templates():
    if not db:
        return []
    return await db.get_templates()


@app.post("/api/templates/{template_id}/execute")
async def execute_template(template_id: int, req: TemplateExecuteRequest):
    if not db or not template_engine:
        raise HTTPException(503)
    template = await db.get_template(template_id)
    if not template:
        raise HTTPException(404, "Template not found")
    await db.increment_template_usage(template_id)
    result = {}
    async for update in template_engine.execute_template(template, req.variables):
        if update.get('type') in ('task_completed', 'task_failed'):
            result = update
    return result or {"status": "completed"}


@app.get("/api/recordings")
async def get_recordings():
    if not db:
        return []
    return await db.get_recordings()


@app.get("/api/recordings/{recording_id}/export")
async def export_recording(recording_id: str, format: str = "python"):
    if not db:
        raise HTTPException(503)
    recording = await db.get_recording(recording_id)
    if not recording:
        raise HTTPException(404)
    if format == "python":
        code = session_recorder.export_as_python(recording)
        return StreamingResponse(
            iter([code]), media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename=auto_{recording_id}.py"})
    elif format == "json":
        wf = session_recorder.export_as_json(recording)
        return StreamingResponse(
            iter([wf]), media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=workflow_{recording_id}.json"})
    raise HTTPException(400, "Format must be 'python' or 'json'")


@app.get("/api/extract")
async def extract_data(format: str = "json"):
    if not data_extractor:
        raise HTTPException(503)
    data = await data_extractor.extract_all()
    if format == "csv":
        csv = data_extractor.to_csv(data)
        return StreamingResponse(
            iter([csv]), media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=data.csv"})
    elif format == "markdown":
        md = data_extractor.to_markdown(data)
        return StreamingResponse(iter([md]), media_type="text/markdown")
    return data


@app.get("/api/workflows")
async def get_workflows():
    if not db:
        return []
    return await db.get_workflows()


@app.post("/api/workflows")
async def create_workflow(req: WorkflowCreateRequest):
    if not db:
        raise HTTPException(503)
    wf_id = str(uuid.uuid4())[:12]
    await db.save_workflow(wf_id, req.name, req.description, json.dumps(req.steps))
    return {"id": wf_id, "name": req.name}


@app.post("/api/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str):
    if not db or not workflow_engine:
        raise HTTPException(503)
    workflow = await db.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(404)
    result = {}
    async for update in workflow_engine.execute_workflow(workflow):
        if update.get('type') == 'workflow_completed':
            result = update
    return result or {"status": "completed"}


@app.get("/api/scheduled")
async def get_scheduled_tasks():
    if not scheduler:
        return []
    return scheduler.get_tasks()


@app.post("/api/scheduled")
async def create_scheduled_task(req: ScheduleTaskRequest):
    if not scheduler:
        raise HTTPException(503)
    return await scheduler.add_task(req.name, req.description, req.interval)


@app.delete("/api/scheduled/{task_id}")
async def delete_scheduled_task(task_id: str):
    if not scheduler:
        raise HTTPException(503)
    await scheduler.remove_task(task_id)
    return {"status": "deleted"}


@app.post("/api/scheduled/{task_id}/toggle")
async def toggle_scheduled_task(task_id: str, enabled: bool = True):
    if not scheduler:
        raise HTTPException(503)
    await scheduler.toggle_task(task_id, enabled)
    return {"status": "toggled", "enabled": enabled}


@app.get("/api/screenshot")
async def get_screenshot():
    if not browser_engine or not browser_engine.is_alive:
        raise HTTPException(503, "Browser not running")
    screenshot = await browser_engine.take_screenshot("default", quality=85)
    if not screenshot:
        raise HTTPException(500, "Screenshot failed")
    return {"screenshot": screenshot}


@app.post("/api/human-input")
async def provide_human_input(req: HumanInputRequest):
    if not orchestrator:
        raise HTTPException(503)
    await orchestrator.provide_human_input(req.task_id, req.input_text)
    return {"status": "input_provided"}


# ------------------------------------------------------------------ #
# Browser selection
# ------------------------------------------------------------------ #

class BrowserLaunchRequest(BaseModel):
    browser: str = "brave"

@app.get("/api/browser/status")
async def browser_status():
    if not browser_engine:
        raise HTTPException(503)
    return {
        "current": browser_engine.browser_name,
        "alive": browser_engine.is_alive,
        "available": browser_engine.get_available_browsers()
    }

@app.post("/api/browser/launch")
async def launch_browser(req: BrowserLaunchRequest):
    if not browser_engine:
        raise HTTPException(503)
    # Serialize against running tasks so we don't race a task's use of
    # the browser with a switch operation. Use the orchestrator's lock.
    lock = getattr(orchestrator, '_run_lock', None) if orchestrator else None
    try:
        if lock:
            try:
                await asyncio.wait_for(lock.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                return {"success": False,
                        "error": "A task is running. Stop it before switching."}
        try:
            # Cap the whole switch at 20s so we never hang the client
            result = await asyncio.wait_for(
                browser_engine.launch_browser(req.browser), timeout=20.0)
        except asyncio.TimeoutError:
            # Make sure built-in is alive even after the timeout
            try:
                await browser_engine.restart()
            except Exception:
                pass
            return {"success": False,
                    "error": "Switch timed out after 20s. Built-in browser restored."}
        return result or {"success": False, "error": "Launch returned no result"}
    except Exception as e:
        logger.error(f"launch_browser failed hard: {e}")
        try:
            await browser_engine.restart()
        except Exception:
            pass
        return {"success": False,
                "error": f"Browser switch failed: {e}. Using built-in."}
    finally:
        if lock and lock.locked():
            try:
                lock.release()
            except RuntimeError:
                pass

@app.post("/api/browser/builtin")
async def switch_to_builtin():
    if not browser_engine:
        raise HTTPException(503)
    result = await browser_engine.switch_to_builtin()
    return result


# ------------------------------------------------------------------ #
# Frontend
# ------------------------------------------------------------------ #
_frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(_frontend_path, "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=500)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
