"""
Task Scheduler
==============
Cron-style scheduling for recurring automations.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def parse_simple_interval(cron_expr: str) -> Optional[int]:
    """Parse a simple interval expression into seconds.
    Supports: '5m', '30m', '1h', '6h', '1d', '12h'
    """
    expr = cron_expr.strip().lower()
    try:
        if expr.endswith('m'):
            return int(expr[:-1]) * 60
        elif expr.endswith('h'):
            return int(expr[:-1]) * 3600
        elif expr.endswith('d'):
            return int(expr[:-1]) * 86400
        elif expr.endswith('s'):
            return int(expr[:-1])
        else:
            return int(expr) * 60  # Default to minutes
    except ValueError:
        return None


class TaskScheduler:
    """Simple interval-based task scheduler."""

    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self._tasks: Dict[str, Dict] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._db = None

    def set_database(self, db):
        self._db = db

    async def load_from_db(self):
        """Load scheduled tasks from database on startup."""
        if not self._db:
            return
        try:
            tasks = await self._db.get_scheduled_tasks()
            for t in tasks:
                if t.get('enabled'):
                    self._tasks[t['id']] = t
                    self._start_task_loop(t['id'])
        except Exception as e:
            logger.error(f"Failed to load scheduled tasks: {e}")

    async def add_task(self, name: str, description: str, interval: str) -> Dict:
        """Add a new scheduled task."""
        task_id = str(uuid.uuid4())[:12]
        seconds = parse_simple_interval(interval)
        if not seconds:
            return {"error": f"Invalid interval: {interval}. Use formats like '5m', '1h', '1d'"}

        task = {
            'id': task_id,
            'name': name,
            'description': description,
            'cron_expression': interval,
            'interval_seconds': seconds,
            'enabled': True,
            'last_run': None,
            'next_run': datetime.now().isoformat(),
            'run_count': 0
        }

        self._tasks[task_id] = task

        if self._db:
            await self._db.save_scheduled_task(
                task_id, name, description, interval,
                task['next_run']
            )

        self._start_task_loop(task_id)
        return task

    def _start_task_loop(self, task_id: str):
        if task_id in self._running_tasks:
            return
        self._running_tasks[task_id] = asyncio.create_task(self._run_loop(task_id))

    async def _run_loop(self, task_id: str):
        """Run a task on its interval."""
        while True:
            task = self._tasks.get(task_id)
            if not task or not task.get('enabled'):
                break

            seconds = task.get('interval_seconds', 3600)
            await asyncio.sleep(seconds)

            task = self._tasks.get(task_id)
            if not task or not task.get('enabled'):
                break

            logger.info(f"Scheduler: Running task '{task.get('name')}'")

            try:
                # Execute the task (non-streaming)
                await self.orchestrator.execute_advanced_task(task['description'])
                task['run_count'] = task.get('run_count', 0) + 1
                task['last_run'] = datetime.now().isoformat()
                next_run = (datetime.now() + timedelta(seconds=seconds)).isoformat()
                task['next_run'] = next_run

                if self._db:
                    await self._db.update_scheduled_task_run(task_id, next_run)
            except Exception as e:
                logger.error(f"Scheduled task '{task.get('name')}' failed: {e}")

    async def remove_task(self, task_id: str) -> bool:
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
            del self._running_tasks[task_id]
        self._tasks.pop(task_id, None)

        if self._db:
            await self._db.delete_scheduled_task(task_id)
        return True

    async def toggle_task(self, task_id: str, enabled: bool) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task['enabled'] = enabled

        if enabled:
            self._start_task_loop(task_id)
        else:
            if task_id in self._running_tasks:
                self._running_tasks[task_id].cancel()
                del self._running_tasks[task_id]

        if self._db:
            await self._db.toggle_scheduled_task(task_id, enabled)
        return True

    def get_tasks(self) -> List[Dict]:
        return list(self._tasks.values())

    async def stop_all(self):
        for task in self._running_tasks.values():
            task.cancel()
        self._running_tasks.clear()
