"""
Task Orchestrator - Agentic loop with robust error handling.
plan -> execute -> evaluate -> adapt -> repeat
"""

import asyncio
import json
import re
import time
import logging
import uuid
from typing import Dict, List, Any, Optional, AsyncGenerator, Tuple
from datetime import datetime
from enum import Enum

from .ai_agent import GroqAIAgent
from .browser_engine import AdvancedBrowserEngine, PageState
from . import blockers as blocker_mod
from . import validators as validators_mod
from . import risk as risk_layer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"        # stopped on a precise, often-recoverable blocker
    UNVERIFIED = "unverified"  # did work but could not prove completion


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
        self.max_steps = options.get('max_steps', 8)
        self.context_id = options.get('context_id', 'default')
        self.total_cost = 0.0
        self.result_summary = ""
        # Structured intent (set at task start) and terminal blocker (set when
        # the run cannot complete). Both feed the WebSocket/DB contract.
        self.intent: Optional[Dict] = None
        self.blocker: Optional[Dict] = None
        self.validation: Optional[Dict] = None


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
        # template runs, workflow runs, and scheduled runs. Created lazily on
        # first use so the lock binds to the running event loop (and so the
        # orchestrator can be constructed outside a loop, e.g. in tests).
        self._run_lock: Optional[asyncio.Lock] = None

    def _ensure_run_lock(self) -> asyncio.Lock:
        """Lazily create the shared run lock bound to the active event loop.

        Callers (task stream, templates, workflows) use this so the lock is
        always present once any browser-driving work begins, regardless of
        whether the orchestrator was constructed inside a running loop.
        """
        if self._run_lock is None:
            self._run_lock = asyncio.Lock()
        return self._run_lock

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

        # Bind the run lock to the active event loop on first use.
        self._ensure_run_lock()

        # Wait our turn on the shared browser. If the caller cancels while
        # we're queued, bail out cleanly without ever marking EXECUTING.
        # Track ownership explicitly: Lock.locked() is True whenever *any*
        # coroutine holds the lock, so we must only release one we acquired.
        lock_held = False
        try:
            # Peek lock - if held, surface a 'queued' update so the UI
            # knows this task is waiting rather than silently stalling.
            if self._run_lock.locked():
                yield {'type': 'task_queued', 'task_id': task_id,
                       'description': description}
            await self._run_lock.acquire()
            lock_held = True
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

            # -- Build structured intent up front. Requirement: every task must
            #    have a success condition before execution starts. --
            intent = self._intent_for(task)
            if not intent.get('success_condition'):
                blocker = blocker_mod.Blocker(
                    blocker_type='ambiguous_instruction',
                    blocker_message='Could not derive a success condition for this request.',
                    current_url=page_state.url if page_state else '',
                ).to_dict()
                task.blocker = blocker
                task.status = TaskStatus.FAILED
                task.result_summary = blocker['blocker_message']
                yield self._terminal_event(task, task_id, 0)
                return

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
                    task.result_summary = f"Agent got stuck after {step_num - 1} steps"
                    task.status = TaskStatus.FAILED
                    task.blocker = blocker_mod.Blocker(
                        blocker_type='partial_completion',
                        blocker_message='Detected repeated actions with no progress.',
                        current_url=page_state.url if page_state else '',
                        page_title=page_state.title if page_state else '',
                        failed_step=step_num,
                        last_successful_step=max(len(task.steps) - 1, 0),
                        suggested_next_step='Rephrase the task or take over the browser.',
                    ).to_dict()
                    yield {'type': 'step_executed', 'step': step_num, 'action': 'done',
                           'success': False, 'confidence': 0.2, 'task_id': task_id,
                           'reasoning': 'Detected repeated actions with no progress',
                           'thinking': '', 'screenshot': screenshot,
                           'error': 'Agent got stuck in a repeated action pattern'}
                    yield self._terminal_event(task, task_id, time.time() - task.start_time)
                    return

                # -- Browser liveness check --
                if page_state.is_error and 'closed' in page_state.error.lower():
                    task.status = TaskStatus.FAILED
                    task.blocker = blocker_mod.Blocker(
                        blocker_type='navigation_failed',
                        blocker_message='Browser was closed. Reconnect and try again.',
                        current_url=page_state.url if page_state else '',
                        failed_step=step_num,
                    ).to_dict()
                    yield self._terminal_event(task, task_id, time.time() - task.start_time)
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

                # Provider/site blockers - fail the task with a clear
                # message rather than pretending to be "done".
                if analysis.get('error') in (
                    'ai_unavailable', 'site_requires_sign_in',
                    'site_blocked_by_bot_check', 'ambiguous_instruction'
                ):
                    err = analysis.get('error')
                    btype = {
                        'ai_unavailable': 'navigation_failed',
                        'site_requires_sign_in': 'login_required',
                        'site_blocked_by_bot_check': 'captcha_or_bot_protection',
                        'ambiguous_instruction': 'ambiguous_instruction',
                    }[err]
                    task.status = (TaskStatus.BLOCKED if btype != 'navigation_failed'
                                   else TaskStatus.FAILED)
                    task.result_summary = analysis.get('message', 'AI provider unavailable')
                    task.blocker = blocker_mod.Blocker(
                        blocker_type=btype,
                        blocker_message=task.result_summary,
                        current_url=page_state.url if page_state else '',
                        page_title=page_state.title if page_state else '',
                        failed_step=step_num,
                        last_successful_step=max(len(task.steps) - 1, 0),
                    ).to_dict()
                    yield self._terminal_event(task, task_id, time.time() - task.start_time)
                    return

                # -- Check if AI says done --
                if analysis.get('task_complete') or analysis.get('action') == 'done':
                    vres = self._validation_result(task, page_state, analysis)
                    if not vres.ok:
                        logger.info(
                            "Ignoring unvalidated completion for %s: %s",
                            task_id, vres.reason)
                        analysis = {
                            'action': 'extract',
                            'parameters': {'target': 'evidence for task completion'},
                            'reasoning': (
                                'Completion was proposed but not validated; '
                                'extracting page evidence instead.'
                            ),
                            'thinking': vres.reason,
                            'confidence': min(analysis.get('confidence', 0.5), 0.45),
                            'task_complete': False,
                        }
                    else:
                        summary = (analysis.get('parameters', {}).get('summary', '')
                                   or analysis.get('reasoning', 'Task completed'))
                        task.validation = self._validation_payload(task, vres)
                        screenshot = await self.browser.take_screenshot(
                            task.context_id, task_id=task_id, step=step_num)
                        yield {'type': 'step_executed', 'step': step_num, 'action': 'done',
                               'success': True, 'confidence': analysis.get('confidence', 0.9),
                               'reasoning': summary, 'thinking': analysis.get('thinking', ''),
                               'screenshot': screenshot, 'error': '', 'task_id': task_id,
                               'success_condition': intent.get('success_condition', ''),
                               'validation_method': intent.get('validation_strategy', ''),
                               'validation': task.validation}
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

                # -- Risk / confirmation gate. Only the action that finalizes a
                #    payment/order/submit pauses for confirmation: the model
                #    flags it (requires_confirmation), and a deterministic
                #    commit-signal check is the safety net. Cookie banners,
                #    searches, and option-picking flow through. --
                needs_confirm = (
                    bool(analysis.get('requires_confirmation'))
                    or risk_layer.action_requires_confirmation(
                        action_type, params, intent))
                if not task.options.get('confirmed') and needs_confirm:
                    blocker = blocker_mod.confirmation_blocker(
                        risk_layer.confirmation_message(intent),
                        url=page_state.url if page_state else '',
                        title=page_state.title if page_state else '',
                        last_successful_step=len(task.steps),
                    ).to_dict()
                    task.blocker = blocker
                    task.status = TaskStatus.BLOCKED
                    task.result_summary = blocker['blocker_message']
                    yield self._terminal_event(task, task_id,
                                               time.time() - task.start_time)
                    return

                exec_result = await self.browser.execute_action(
                    task.context_id, action_type, params)

                # -- Unsafe URL: the SSRF guard refused this navigation. Stop
                #    with a precise, user-overridable blocker rather than
                #    burning retries on a target we will never open. --
                if exec_result.get('blocked_url'):
                    task.status = TaskStatus.BLOCKED
                    task.result_summary = exec_result.get('error', 'Blocked unsafe URL')
                    task.blocker = blocker_mod.Blocker(
                        blocker_type='unsafe_action',
                        blocker_message=task.result_summary,
                        current_url=page_state.url if page_state else '',
                        page_title=page_state.title if page_state else '',
                        failed_step=step_num,
                        last_successful_step=max(len(task.steps) - 1, 0),
                        suggested_next_step=(
                            'This URL points at a local or metadata address and is '
                            'blocked for safety. Set HELM_ALLOW_PRIVATE_HOSTS=true to '
                            'allow local targets you trust.'),
                    ).to_dict()
                    yield {'type': 'step_executed', 'step': step_num,
                           'action': action_type, 'parameters': params,
                           'success': False, 'confidence': 0,
                           'reasoning': 'Navigation blocked by URL safety policy',
                           'thinking': '', 'screenshot': None,
                           'error': task.result_summary, 'task_id': task_id}
                    yield self._terminal_event(task, task_id, time.time() - task.start_time)
                    return

                # -- Fatal error: browser gone --
                if exec_result.get('fatal'):
                    task.status = TaskStatus.FAILED
                    task.result_summary = 'Browser disconnected. Please restart and try again.'
                    task.blocker = blocker_mod.Blocker(
                        blocker_type='navigation_failed',
                        blocker_message=task.result_summary,
                        current_url=page_state.url if page_state else '',
                        failed_step=step_num,
                        last_successful_step=max(len(task.steps) - 1, 0),
                        visible_evidence=exec_result.get('error', ''),
                    ).to_dict()
                    yield {'type': 'step_executed', 'step': step_num,
                           'action': action_type, 'success': False,
                           'confidence': 0, 'reasoning': 'Browser disconnected',
                           'thinking': '', 'screenshot': None,
                           'error': exec_result.get('error', ''), 'task_id': task_id}
                    yield self._terminal_event(task, task_id, time.time() - task.start_time)
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
                    'data': exec_result.get('data', {}) or {},
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

                if exec_result.get('task_complete'):
                    vres = self._validation_result(
                        task, new_state, {
                            'action': 'done',
                            'parameters': {'summary': exec_result.get('summary', '')},
                            'intent': analysis.get('intent'),
                        })
                    if vres.ok:
                        task.validation = self._validation_payload(task, vres)
                        task.result_summary = (
                            exec_result.get('summary')
                            or exec_result.get('error')
                            or 'Task completed'
                        )
                        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                        break
                    exec_result['task_complete'] = False
                    logger.info("Ignoring workflow task_complete for %s: %s", task_id, vres.reason)

                # Handle failures
                if not success:
                    consecutive_failures += 1
                    error_text = exec_result.get('error', '')
                    if (
                        'Unsupported browser action' in error_text
                        or 'cannot be executed directly' in error_text
                        or 'Refusing to type' in error_text
                    ):
                        blocker = self._classify_blocker(
                            new_state, error_text,
                            failed_step=step_num,
                            last_successful_step=max(len(task.steps) - 1, 0))
                        blocker['suggested_next_step'] = (
                            'Revise the plan or clean extracted text before retrying.')
                        task.blocker = blocker
                        task.status = (TaskStatus.BLOCKED
                                       if blocker['status'] == 'blocked'
                                       else TaskStatus.FAILED)
                        task.result_summary = blocker['blocker_message']
                        yield self._terminal_event(task, task_id, time.time() - task.start_time)
                        return
                    if consecutive_failures >= 5:
                        blocker = self._classify_blocker(
                            new_state, exec_result.get('error', ''),
                            failed_step=step_num,
                            last_successful_step=max(len(task.steps) - 1, 0))
                        task.blocker = blocker
                        task.status = (TaskStatus.BLOCKED
                                       if blocker['status'] == 'blocked'
                                       else TaskStatus.FAILED)
                        task.result_summary = blocker['blocker_message']
                        yield self._terminal_event(task, task_id, time.time() - task.start_time)
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
                        # The cheap completion model is not trusted on its own.
                        # Gate it through the same evidence validator as the main
                        # done path so this can't become a weaker second route to
                        # COMPLETED.
                        vres = self._validation_result(task, page_state, {})
                        if vres.ok:
                            task.result_summary = completion.get('summary', 'Task completed')
                            task.status = TaskStatus.COMPLETED
                            task.validation = self._validation_payload(task, vres)
                            yield {'type': 'completion_check', 'completed': True,
                                   'confidence': completion['confidence'],
                                   'summary': task.result_summary, 'task_id': task_id}
                            break
                        logger.info(
                            "Ignoring unvalidated periodic completion for %s: %s",
                            task_id, vres.reason)

            # -- Finalize --
            task.end_time = time.time()
            exec_time = task.end_time - task.start_time
            stats = self.ai_agent.get_token_stats()
            task.total_cost = stats.get('total_cost', 0)

            # Ran out of steps while still EXECUTING. Let the validator decide
            # the honest terminal state instead of a blanket "failed".
            if task.status == TaskStatus.EXECUTING:
                vres = self._validation_result(task, page_state, {})
                if vres.ok:
                    task.status = TaskStatus.COMPLETED
                    task.validation = self._validation_payload(task, vres)
                    task.result_summary = task.result_summary or vres.reason
                elif vres.status == validators_mod.UNVERIFIED:
                    task.status = TaskStatus.UNVERIFIED
                    task.result_summary = (
                        f"Reached {step_num} steps; performed work but could not "
                        f"prove completion."
                    )
                    task.blocker = blocker_mod.Blocker(
                        blocker_type='partial_completion',
                        blocker_message=vres.reason or task.result_summary,
                        current_url=page_state.url if page_state else '',
                        page_title=page_state.title if page_state else '',
                        last_successful_step=max(len(task.steps) - 1, 0),
                        visible_evidence=vres.evidence,
                    ).to_dict()
                else:
                    blocker = vres.blocker or self._classify_blocker(
                        page_state, vres.reason,
                        failed_step=len(task.steps),
                        last_successful_step=max(len(task.steps) - 1, 0))
                    task.blocker = task.blocker or blocker
                    task.status = (TaskStatus.BLOCKED
                                   if task.blocker.get('status') == 'blocked'
                                   else TaskStatus.FAILED)
                    task.result_summary = (task.blocker.get('blocker_message')
                                           or vres.reason)

            # Persist first, then count. Metrics should reflect tasks that were
            # actually recorded, not ones whose save threw.
            if self._db:
                try:
                    await self._db.save_task(task)
                except Exception as e:
                    logger.error(f"DB save failed: {e}")

            self._update_metrics(task, exec_time)

            if task.status != TaskStatus.CANCELLED:
                yield self._terminal_event(task, task_id, exec_time)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.end_time = time.time()
            logger.error(f"Task failed: {e}")
            self.performance_metrics['failed_tasks'] += 1
            task.blocker = task.blocker or blocker_mod.Blocker(
                blocker_type='navigation_failed',
                blocker_message=f"Unexpected error during execution: {e}",
                current_url=(task.current_page_state.url
                             if task.current_page_state else ''),
                failed_step=len(task.steps),
            ).to_dict()
            task.result_summary = task.result_summary or str(e)
            yield self._terminal_event(
                task, task_id,
                (task.end_time - (task.start_time or task.end_time)))
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
            # Only release the lock if THIS run acquired it.
            if lock_held:
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

        # Same failed action+params 3x in a row means we are retrying a
        # broken selector or blocked action. Successful repeated actions can
        # be legitimate on dynamic pages, and should be judged by state/next
        # planning rather than killed preemptively.
        if history and len(history) >= 3:
            last3h = history[-3:]
            same_action = len({h.get('action', '') for h in last3h}) == 1
            same_params = len({json.dumps(h.get('parameters', {}), sort_keys=True) for h in last3h}) == 1
            all_failed = all(not h.get('success') for h in last3h)
            if same_action and same_params and all_failed:
                return True

        return False

    def _classify_blocker(self, state: Optional[PageState], last_error: str = "",
                          **kw) -> Dict[str, Any]:
        """Delegate to the structured blocker classifier (core/blockers.py)."""
        state = state or PageState("error", "Unknown", "", [], error="")
        blocker = blocker_mod.classify_blocker(
            url=state.url or "",
            title=state.title or "",
            content=state.content or "",
            last_error=last_error or state.error or "",
            is_error=state.is_error,
            **kw,
        )
        return blocker.to_dict()

    def _intent_for(self, task: AdvancedTask) -> Dict[str, Any]:
        """Parse (once) and cache the structured intent for a task."""
        if task.intent:
            return task.intent
        planner = getattr(self.ai_agent, "intent_planner", None)
        if planner is not None:
            try:
                task.intent = planner.parse_intent(task.description).to_dict()
            except Exception as e:
                logger.error(f"Intent parse failed: {e}")
                task.intent = {}
        else:
            task.intent = {}
        return task.intent

    def _validation_result(self, task: AdvancedTask, state: Optional[PageState],
                           analysis: Dict[str, Any]):
        """Run the composable validator system and return a ValidationResult.

        The success condition comes from the deterministic planner
        (``task.intent``), never from what the model reports back in
        ``analysis`` - otherwise a misbehaving or injected model could supply
        an easy condition it also claims to satisfy. The model-reported intent
        is used only if the planner produced none.
        """
        intent = self._intent_for(task) or (analysis or {}).get("intent")
        history = task.context.get("action_history", [])
        extracted = task.context.get("extracted_data", [])
        return validators_mod.validate_completion(intent, state, history, extracted)

    def _validate_completion(self, task: AdvancedTask, state: Optional[PageState],
                             analysis: Dict[str, Any]) -> Tuple[bool, str]:
        """Back-compat boolean wrapper around the validator system. Completion
        requires the validator to return COMPLETED."""
        res = self._validation_result(task, state, analysis)
        return res.ok, res.reason

    def _validation_payload(self, task: AdvancedTask, vres) -> Dict[str, Any]:
        intent = task.intent or {}
        return {
            "status": vres.status,
            "success_condition": intent.get("success_condition", ""),
            "validation_method": intent.get("validation_strategy", ""),
            "reason": vres.reason,
            "evidence": vres.evidence,
        }

    def _terminal_event(self, task: AdvancedTask, task_id: str,
                        exec_time: float) -> Dict[str, Any]:
        """Build the terminal WebSocket event for a finished run.

        COMPLETED runs carry validation evidence; every other terminal state
        carries a precise structured blocker.
        """
        base = {
            "task_id": task_id,
            "status": task.status.value,
            "steps_taken": len(task.steps),
            "execution_time": exec_time,
            "result_summary": task.result_summary,
        }
        stats = (self.ai_agent.get_token_stats()
                 if hasattr(self.ai_agent, "get_token_stats") else {})
        if task.status == TaskStatus.COMPLETED:
            base.update({
                "type": "task_completed",
                "cost_summary": f"${task.total_cost:.4f}",
                "groq_stats": stats,
                "urls_visited": task.context.get("urls_visited", []),
                "extracted_data": task.context.get("extracted_data", []),
                "validation": task.validation or {},
            })
            return base
        # failed / blocked / unverified share the task_failed envelope so the
        # frontend's terminal handling fires; the precise state is in `status`.
        blocker = task.blocker or {}
        base.update({
            "type": "task_failed",
            "error": (task.result_summary or blocker.get("blocker_message")
                      or "Task did not complete"),
            "blocker": blocker,
            "blocker_type": blocker.get("blocker_type", ""),
            "blocker_message": blocker.get("blocker_message", ""),
            "current_url": blocker.get("current_url", ""),
            "page_title": blocker.get("page_title", ""),
            "failed_step": blocker.get("failed_step", len(task.steps)),
            "last_successful_step": blocker.get("last_successful_step",
                                                max(len(task.steps) - 1, 0)),
            "attempted_recoveries": blocker.get("attempted_recoveries", []),
            "visible_evidence": blocker.get("visible_evidence", ""),
            "suggested_next_step": blocker.get("suggested_next_step", ""),
        })
        return base

    def _same_url(self, a: str, b: str) -> bool:
        def clean(u: str) -> str:
            return (u or "").rstrip("/")
        return bool(a and b and clean(a) == clean(b))

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
