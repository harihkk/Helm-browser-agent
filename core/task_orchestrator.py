"""
Task Orchestrator - Agentic loop with robust error handling.
plan -> execute -> evaluate -> adapt -> repeat
"""

import asyncio
import json
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime
from enum import Enum

from .ai_agent import GroqAIAgent, ActionPlan, ActionType
from .browser_engine import AdvancedBrowserEngine, PageState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AdvancedTask:
    def __init__(self, task_id: str, description: str, options: Dict):
        self.id = task_id
        self.description = description
        self.options = options
        self.status = TaskStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.steps: List[Dict] = []
        self.context: Dict = {
            'action_history': [],
            'urls_visited': [],
            'errors': [],
            'extracted_data': [],
            'human_inputs': [],
        }
        self.current_page_state: Optional[PageState] = None
        self.max_steps = options.get('max_steps', 25)
        self.context_id = options.get('context_id', 'default')
        self.total_cost = 0.0
        self.result_summary = ""


class SophisticatedTaskOrchestrator:
    def __init__(self, browser_engine: AdvancedBrowserEngine, ai_agent: GroqAIAgent):
        self.browser = browser_engine
        self.ai_agent = ai_agent
        self.active_tasks: Dict[str, AdvancedTask] = {}
        self.task_history: List[Dict] = []
        self.performance_metrics = {
            'total_tasks': 0, 'successful_tasks': 0, 'failed_tasks': 0,
            'average_steps': 0, 'average_execution_time': 0, 'total_cost': 0
        }
        self._cancel_events: Dict[str, asyncio.Event] = {}
        self._db = None
        self._preview_callbacks: List = []
        # Serializes access to the single shared browser across task runs,
        # template runs, workflow runs, and scheduled runs.
        self._run_lock = asyncio.Lock()

    def set_database(self, db):
        self._db = db

    def register_preview_callback(self, callback):
        self._preview_callbacks.append(callback)

    # ------------------------------------------------------------------ #
    # Streaming task execution
    # ------------------------------------------------------------------ #

    async def execute_task_stream(self, description: str, options: Dict = None,
                                   cancel_event: asyncio.Event = None) -> AsyncGenerator:
        options = options or {}
        task_id = str(uuid.uuid4())[:12]
        task = AdvancedTask(task_id, description, options)
        task.status = TaskStatus.PENDING
        self.active_tasks[task_id] = task

        if cancel_event:
            self._cancel_events[task_id] = cancel_event

        # Wait our turn on the shared browser. If the caller cancels while
        # we're queued, bail out cleanly without ever marking EXECUTING.
        try:
            # Peek lock - if held, surface a 'queued' update so the UI
            # knows this task is waiting rather than silently stalling.
            if self._run_lock.locked():
                yield {'type': 'task_queued', 'task_id': task_id,
                       'description': description}
            await self._run_lock.acquire()
        except asyncio.CancelledError:
            self.active_tasks.pop(task_id, None)
            self._cancel_events.pop(task_id, None)
            raise

        task.start_time = time.time()
        task.status = TaskStatus.EXECUTING

        yield {'type': 'task_started', 'task_id': task_id, 'description': description,
               'max_steps': task.max_steps}

        try:
            # Check browser is alive - if not, try to restart before failing.
            if not self.browser.is_alive:
                logger.warning("Browser not alive at task start - attempting auto-restart")
                try:
                    await self.browser.restart()
                except Exception as e:
                    logger.error(f"Auto-restart failed: {e}")
                if not self.browser.is_alive:
                    yield {'type': 'task_failed', 'task_id': task_id,
                           'error': 'Browser could not start. Try stopping the server '
                                    '(Ctrl+C) and running python run.py again.',
                           'steps_taken': 0, 'execution_time': 0}
                    return

            page_state = await self.browser.get_page_state(task.context_id)
            task.current_page_state = page_state

            step_num = 0
            consecutive_failures = 0
            action_log: List[str] = []  # Track action types for loop detection

            while step_num < task.max_steps:
                # -- Cancellation check --
                if self._is_cancelled(cancel_event):
                    task.status = TaskStatus.CANCELLED
                    yield {'type': 'task_cancelled', 'task_id': task_id, 'steps_taken': step_num}
                    break

                step_num += 1

                # -- Loop detection --
                if self._detect_loop(action_log, task.context.get('action_history', [])):
                    screenshot = await self.browser.take_screenshot(task.context_id, task_id=task_id, step=step_num)
                    task.result_summary = f"Completed after {step_num - 1} steps"
                    task.status = TaskStatus.COMPLETED
                    yield {'type': 'step_executed', 'step': step_num, 'action': 'done',
                           'success': True, 'confidence': 0.7, 'task_id': task_id,
                           'reasoning': 'Detected repeated actions - completing task',
                           'thinking': '', 'screenshot': screenshot, 'error': ''}
                    break

                # -- Browser liveness check --
                if page_state.is_error and 'closed' in page_state.error.lower():
                    task.status = TaskStatus.FAILED
                    yield {'type': 'task_failed', 'task_id': task_id,
                           'error': 'Browser was closed. Reconnect and try again.',
                           'steps_taken': step_num, 'execution_time': time.time() - task.start_time}
                    return

                yield {'type': 'step_started', 'step': step_num,
                       'max_steps': task.max_steps, 'task_id': task_id}

                # -- 1. AI analysis (single API call decides action) --
                state_dict = page_state.to_dict()
                analysis = await self.ai_agent.analyze_page_text(
                    state_dict, task.description, task.context
                )

                if self._is_cancelled(cancel_event):
                    task.status = TaskStatus.CANCELLED
                    yield {'type': 'task_cancelled', 'task_id': task_id, 'steps_taken': step_num}
                    break

                # AI provider out of quota - fail the task with a clear message
                # rather than pretending to be "done".
                if analysis.get('error') == 'ai_unavailable':
                    task.status = TaskStatus.FAILED
                    yield {'type': 'task_failed', 'task_id': task_id,
                           'error': analysis.get('message', 'AI provider unavailable'),
                           'steps_taken': len(task.steps),
                           'execution_time': time.time() - task.start_time}
                    return

                # -- Check if AI says done --
                if analysis.get('task_complete') or analysis.get('action') == 'done':
                    summary = (analysis.get('parameters', {}).get('summary', '')
                               or analysis.get('reasoning', 'Task completed'))
                    screenshot = await self.browser.take_screenshot(
                        task.context_id, task_id=task_id, step=step_num)
                    yield {'type': 'step_executed', 'step': step_num, 'action': 'done',
                           'success': True, 'confidence': analysis.get('confidence', 0.9),
                           'reasoning': summary, 'thinking': analysis.get('thinking', ''),
                           'screenshot': screenshot, 'error': '', 'task_id': task_id}
                    task.result_summary = summary
                    task.status = TaskStatus.COMPLETED
                    break

                # -- 2. Execute action directly from analysis (skip separate plan API call) --
                action_type = analysis.get('action', 'extract')
                params = analysis.get('parameters', {})
                confidence = analysis.get('confidence', 0.5)

                yield {'type': 'planning_complete', 'step': step_num,
                       'plans_created': 1,
                       'thinking': analysis.get('thinking', ''),
                       'confidence': confidence,
                       'groq_stats': self.ai_agent.get_token_stats(),
                       'task_id': task_id}

                if self._is_cancelled(cancel_event):
                    break

                exec_result = await self.browser.execute_action(
                    task.context_id, action_type, params)

                # -- Fatal error: browser gone --
                if exec_result.get('fatal'):
                    task.status = TaskStatus.FAILED
                    yield {'type': 'step_executed', 'step': step_num,
                           'action': action_type, 'success': False,
                           'confidence': 0, 'reasoning': 'Browser disconnected',
                           'thinking': '', 'screenshot': None,
                           'error': exec_result.get('error', ''), 'task_id': task_id}
                    yield {'type': 'task_failed', 'task_id': task_id,
                           'error': 'Browser disconnected. Please restart and try again.',
                           'steps_taken': len(task.steps),
                           'execution_time': time.time() - task.start_time}
                    return

                # NOTE: the executor already settles after each action
                # (_smart_wait on navigate, sleeps after click/type/etc.).
                # No extra sleep here - it only adds perceived latency.

                new_state = await self.browser.get_page_state(task.context_id)

                # Screenshots are expensive - only grab one when the action
                # can actually change the visible page. `extract` and `done`
                # don't; skip them.
                if action_type in ('extract', 'done'):
                    screenshot = None
                else:
                    screenshot = await self.browser.take_screenshot(
                        task.context_id, task_id=task_id, step=step_num)
                diff = await self.browser.get_page_diff(task.context_id)

                success = exec_result.get('success', False)

                # Record
                record = {
                    'step': step_num, 'action': action_type, 'parameters': params,
                    'success': success, 'result': json.dumps(exec_result)[:200],
                    'summary': f"{action_type}: {'OK' if success else 'FAILED'}",
                    'evaluation': f"{action_type} {'succeeded' if success else 'failed'}",
                    'timestamp': datetime.now().isoformat()
                }
                task.context['action_history'].append(record)
                task.steps.append(record)
                action_log.append(action_type)

                if success and action_type == 'navigate':
                    task.context['urls_visited'].append(params.get('url', ''))
                if not success:
                    task.context['errors'].append({
                        'step': step_num, 'error': exec_result.get('error', ''),
                        'action': action_type
                    })
                if action_type == 'extract':
                    data = exec_result.get('data', {}) or {}
                    # Skip if we just stored the same URL+content; the AI
                    # sometimes calls extract repeatedly on the same page.
                    prev = task.context['extracted_data'][-1] if task.context['extracted_data'] else None
                    is_dup = (prev
                              and prev.get('url') == data.get('url')
                              and prev.get('content') == data.get('content'))
                    if not is_dup:
                        task.context['extracted_data'].append(data)

                yield {'type': 'step_executed', 'step': step_num,
                       'action': action_type, 'parameters': params,
                       'success': success,
                       'confidence': confidence,
                       'reasoning': analysis.get('reasoning', ''),
                       'thinking': analysis.get('thinking', ''),
                       'screenshot': screenshot, 'diff': diff,
                       'error': exec_result.get('error', ''), 'task_id': task_id}

                # Handle failures
                if not success:
                    consecutive_failures += 1
                    if consecutive_failures >= 5:
                        task.status = TaskStatus.FAILED
                        yield {'type': 'task_failed', 'task_id': task_id,
                               'error': 'Too many consecutive failures. Task aborted.',
                               'steps_taken': len(task.steps),
                               'execution_time': time.time() - task.start_time}
                        return
                else:
                    consecutive_failures = 0

                page_state = new_state
                task.current_page_state = page_state

                # -- Completion check every 5 steps (uses cheap eval model) --
                if step_num % 5 == 0 and step_num > 0:
                    if self._is_cancelled(cancel_event):
                        break
                    completion = await self.ai_agent.check_completion(
                        task.description, task.context['action_history'],
                        page_state.to_dict() if page_state else {})
                    if completion.get('completed') and completion.get('confidence', 0) > 0.7:
                        task.result_summary = completion.get('summary', 'Task completed')
                        task.status = TaskStatus.COMPLETED
                        yield {'type': 'completion_check', 'completed': True,
                               'confidence': completion['confidence'],
                               'summary': task.result_summary, 'task_id': task_id}
                        break

            # -- Finalize --
            task.end_time = time.time()
            exec_time = task.end_time - task.start_time
            stats = self.ai_agent.get_token_stats()
            task.total_cost = stats.get('total_cost', 0)

            if task.status == TaskStatus.EXECUTING:
                task.status = TaskStatus.COMPLETED
                task.result_summary = f"Completed {step_num} steps"

            self._update_metrics(task, exec_time)

            if self._db:
                try:
                    await self._db.save_task(task)
                except Exception as e:
                    logger.error(f"DB save failed: {e}")

            yield {'type': 'task_completed', 'task_id': task_id,
                   'status': task.status.value, 'steps_taken': len(task.steps),
                   'execution_time': exec_time,
                   'cost_summary': f"${task.total_cost:.4f}",
                   'groq_stats': stats, 'result_summary': task.result_summary,
                   'urls_visited': task.context.get('urls_visited', []),
                   'extracted_data': task.context.get('extracted_data', [])}

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.end_time = time.time()
            logger.error(f"Task failed: {e}")
            self.performance_metrics['failed_tasks'] += 1
            yield {'type': 'task_failed', 'task_id': task_id, 'error': str(e),
                   'steps_taken': len(task.steps),
                   'execution_time': task.end_time - (task.start_time or task.end_time)}
        finally:
            if task_id in self.active_tasks:
                self.task_history.append({
                    'task_id': task_id, 'description': description,
                    'status': task.status.value, 'steps': len(task.steps),
                    'started': task.start_time, 'ended': task.end_time,
                    'cost': task.total_cost
                })
                del self.active_tasks[task_id]
            self._cancel_events.pop(task_id, None)
            if self._run_lock.locked():
                try:
                    self._run_lock.release()
                except RuntimeError:
                    pass

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _is_cancelled(self, cancel_event: Optional[asyncio.Event]) -> bool:
        return cancel_event is not None and cancel_event.is_set()

    def _detect_loop(self, action_log: List[str], history: List[Dict] = None) -> bool:
        """Detect if the agent is stuck in a non-productive loop.

        Only trip on actions that indicate spinning, never on legitimately
        repeated productive actions like multiple `type` or `click` steps
        that happen to share an action name but target different elements.
        """
        if len(action_log) < 3:
            return False

        # Three idle-ish actions in a row - scrolling/waiting/extracting with no progress
        last3 = action_log[-3:]
        if len(set(last3)) == 1 and last3[0] in ('scroll', 'wait', 'extract'):
            return True

        # Alternating idle pattern (scroll/wait/scroll/wait etc.)
        if len(action_log) >= 4:
            last4 = action_log[-4:]
            if (last4[0] == last4[2] and last4[1] == last4[3]
                    and last4[0] != last4[1]
                    and set(last4) <= {'scroll', 'wait', 'extract'}):
                return True

        # Same action+params 3x in a row, regardless of success.
        # If they all FAILED we've been retrying a broken selector.
        # If they all SUCCEEDED with identical params, the action isn't
        # making progress (typing the same text into the same box, etc.).
        if history and len(history) >= 3:
            last3h = history[-3:]
            same_action = len({h.get('action', '') for h in last3h}) == 1
            same_params = len({json.dumps(h.get('parameters', {}), sort_keys=True) for h in last3h}) == 1
            if same_action and same_params:
                return True

        return False

    # ------------------------------------------------------------------ #
    # Non-streaming
    # ------------------------------------------------------------------ #

    async def execute_advanced_task(self, description: str, options: Dict = None) -> Dict:
        result = {}
        async for update in self.execute_task_stream(description, options):
            if update['type'] in ('task_completed', 'task_failed'):
                result = update
        return result or {'task_id': 'unknown', 'status': 'failed', 'steps_taken': 0}

    # ------------------------------------------------------------------ #
    # Task management
    # ------------------------------------------------------------------ #

    def cancel_task(self, task_id: str):
        event = self._cancel_events.get(task_id)
        if event:
            event.set()

    def get_active_tasks(self) -> List[Dict]:
        return [{'task_id': t.id, 'description': t.description,
                 'status': t.status.value, 'steps': len(t.steps)}
                for t in self.active_tasks.values()]

    def get_task_history(self, limit: int = 50) -> List[Dict]:
        return self.task_history[-limit:]

    def get_performance_metrics(self):
        return self.performance_metrics

    async def provide_human_input(self, task_id: str, input_text: str) -> bool:
        """Attach a human hint to the running task; the AI reads it on the
        next analyze_page_text call via context['human_inputs']."""
        task = self.active_tasks.get(task_id)
        if not task or not input_text:
            return False
        task.context.setdefault('human_inputs', []).append({
            'text': input_text.strip(),
            'step': len(task.steps),
            'timestamp': datetime.now().isoformat(),
        })
        logger.info(f"Human input attached to {task_id}: {input_text[:80]}")
        return True

    def _update_metrics(self, task: AdvancedTask, exec_time: float):
        m = self.performance_metrics
        m['total_tasks'] += 1
        if task.status == TaskStatus.COMPLETED:
            m['successful_tasks'] += 1
        elif task.status == TaskStatus.FAILED:
            m['failed_tasks'] += 1
        total = m['total_tasks']
        m['average_steps'] = ((m['average_steps'] * (total - 1)) + len(task.steps)) / total
        m['average_execution_time'] = ((m['average_execution_time'] * (total - 1)) + exec_time) / total
        m['total_cost'] = m.get('total_cost', 0) + task.total_cost
