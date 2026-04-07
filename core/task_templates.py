"""
Task Templates
==============
Pre-built automation templates. Templates have deterministic step lists,
so we execute them directly against the browser rather than re-asking the
LLM what to do (which would be slow, expensive, and non-deterministic).
"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict

logger = logging.getLogger(__name__)


class TemplateEngine:
    """Execute task templates by substituting variables and running steps."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.browser = orchestrator.browser

    async def execute_template(self, template: Dict, variables: Dict) -> AsyncGenerator:
        steps_json = template.get('steps_json', '[]')
        steps = json.loads(steps_json) if isinstance(steps_json, str) else steps_json
        resolved = self._resolve_variables(steps, variables)

        task_id = f"tmpl_{uuid.uuid4().hex[:8]}"
        name = template.get('name', 'Template')

        # Share the orchestrator's lock so we never drive the browser
        # concurrently with an AI task or another template.
        lock = getattr(self.orchestrator, '_run_lock', None)
        if lock and lock.locked():
            yield {'type': 'task_queued', 'task_id': task_id,
                   'description': f"Template: {name}"}
        if lock:
            await lock.acquire()

        start = time.time()

        try:
            yield {
                'type': 'task_started',
                'task_id': task_id,
                'description': f"Template: {name}",
                'max_steps': len(resolved),
            }

            executed = 0
            failures = 0
            last_screenshot = None

            for i, step in enumerate(resolved, start=1):
                action = step.get('action', '')
                params = step.get('parameters', {})

                yield {
                    'type': 'step_started', 'step': i,
                    'max_steps': len(resolved), 'task_id': task_id,
                }

                try:
                    result = await self.browser.execute_action('default', action, params)
                except Exception as e:
                    result = {'success': False, 'error': str(e)}

                # Executor already settles internally.

                if action in ('extract', 'done'):
                    last_screenshot = None
                else:
                    last_screenshot = await self.browser.take_screenshot(
                        'default', task_id=task_id, step=i)

                success = bool(result.get('success'))
                executed += 1
                if not success:
                    failures += 1

                yield {
                    'type': 'step_executed', 'step': i,
                    'action': action, 'parameters': params,
                    'success': success, 'confidence': 1.0,
                    'reasoning': f"Template step: {action}",
                    'thinking': '',
                    'screenshot': last_screenshot,
                    'error': result.get('error', ''),
                    'task_id': task_id,
                }

                if not success and failures >= 2:
                    yield {
                        'type': 'task_failed', 'task_id': task_id,
                        'error': f'Template failed at step {i}: {result.get("error", "")}',
                        'steps_taken': executed,
                        'execution_time': time.time() - start,
                    }
                    return

            yield {
                'type': 'task_completed', 'task_id': task_id,
                'status': 'completed', 'steps_taken': executed,
                'execution_time': time.time() - start,
                'cost_summary': '$0.0000',
                'result_summary': f"Template '{name}' executed {executed} steps",
                'urls_visited': [], 'extracted_data': [],
            }
        finally:
            if lock and lock.locked():
                try:
                    lock.release()
                except RuntimeError:
                    pass

    def _resolve_variables(self, obj: Any, variables: Dict) -> Any:
        if isinstance(obj, str):
            for key, value in variables.items():
                obj = obj.replace(f'{{{key}}}', str(value))
            return obj
        if isinstance(obj, dict):
            return {k: self._resolve_variables(v, variables) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_variables(item, variables) for item in obj]
        return obj
