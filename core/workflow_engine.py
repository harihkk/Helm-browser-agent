"""
Workflow Engine
===============
Chain multiple tasks into multi-step workflows with conditions.
"""

import json
import uuid
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from datetime import datetime

logger = logging.getLogger(__name__)


class WorkflowStep:
    def __init__(self, step_data: Dict):
        self.order = step_data.get('order', 0)
        self.name = step_data.get('name', f'Step {self.order}')
        self.task_description = step_data.get('task_description', '')
        self.condition = step_data.get('condition', None)  # {"type": "if_success|if_failed|always", "step": N}
        self.on_failure = step_data.get('on_failure', 'stop')  # stop | skip | retry


class WorkflowEngine:
    """Execute multi-step workflows with conditions and branching."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.active_workflows: Dict[str, Dict] = {}

    async def execute_workflow(self, workflow: Dict) -> AsyncGenerator:
        """Execute a workflow definition with conditional steps."""
        workflow_id = workflow.get('id', str(uuid.uuid4())[:12])
        name = workflow.get('name', 'Workflow')
        steps_json = workflow.get('steps_json', '[]')

        if isinstance(steps_json, str):
            steps_data = json.loads(steps_json)
        else:
            steps_data = steps_json

        steps = [WorkflowStep(s) for s in steps_data]
        steps.sort(key=lambda s: s.order)

        self.active_workflows[workflow_id] = {
            'id': workflow_id,
            'name': name,
            'status': 'running',
            'current_step': 0,
            'results': {}
        }

        yield {
            'type': 'workflow_started',
            'workflow_id': workflow_id,
            'name': name,
            'total_steps': len(steps)
        }

        step_results = {}

        for i, step in enumerate(steps):
            # Check condition
            if step.condition:
                cond_type = step.condition.get('type', 'always')
                ref_step = step.condition.get('step')

                if cond_type == 'if_success' and ref_step is not None:
                    if not step_results.get(ref_step, {}).get('success'):
                        yield {
                            'type': 'workflow_step_skipped',
                            'step': step.order,
                            'name': step.name,
                            'reason': f'Condition not met: step {ref_step} did not succeed'
                        }
                        continue
                elif cond_type == 'if_failed' and ref_step is not None:
                    if step_results.get(ref_step, {}).get('success', True):
                        yield {
                            'type': 'workflow_step_skipped',
                            'step': step.order,
                            'name': step.name,
                            'reason': f'Condition not met: step {ref_step} did not fail'
                        }
                        continue

            yield {
                'type': 'workflow_step_started',
                'step': step.order,
                'name': step.name,
                'task': step.task_description
            }

            # Execute the step's task
            step_success = False
            step_summary = ""
            async for update in self.orchestrator.execute_task_stream(step.task_description):
                update['workflow_id'] = workflow_id
                update['workflow_step'] = step.order
                yield update

                if update.get('type') == 'task_completed':
                    step_success = True
                    step_summary = update.get('result_summary', '')
                elif update.get('type') == 'task_failed':
                    step_success = False
                    step_summary = update.get('error', 'Failed')

            step_results[step.order] = {
                'success': step_success,
                'summary': step_summary
            }

            yield {
                'type': 'workflow_step_completed',
                'step': step.order,
                'name': step.name,
                'success': step_success,
                'summary': step_summary
            }

            # Handle failure
            if not step_success and step.on_failure == 'stop':
                yield {
                    'type': 'workflow_stopped',
                    'reason': f'Step {step.order} "{step.name}" failed and on_failure=stop',
                    'workflow_id': workflow_id
                }
                break

        self.active_workflows[workflow_id]['status'] = 'completed'

        yield {
            'type': 'workflow_completed',
            'workflow_id': workflow_id,
            'name': name,
            'step_results': step_results
        }
