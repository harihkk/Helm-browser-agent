"""
Database Layer (Async SQLite)
=============================
Persistence for tasks, steps, templates, recordings, workflows, and scheduled tasks.
"""

import json
import logging
import aiosqlite
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = "./agent_memory.db"


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info(f"Database initialized: {self.db_path}")

    async def _create_tables(self):
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT (datetime('now')),
                completed_at TEXT,
                steps_taken INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0,
                execution_time REAL DEFAULT 0,
                result_summary TEXT DEFAULT '',
                urls_visited TEXT DEFAULT '[]',
                extracted_data TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS task_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                step_number INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                parameters TEXT DEFAULT '{}',
                success INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0,
                reasoning TEXT DEFAULT '',
                thinking TEXT DEFAULT '',
                error TEXT DEFAULT '',
                screenshot_path TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );

            CREATE TABLE IF NOT EXISTS task_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                category TEXT DEFAULT 'general',
                steps_json TEXT NOT NULL,
                variables TEXT DEFAULT '[]',
                usage_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS recordings (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                task_id TEXT,
                steps_json TEXT NOT NULL,
                duration REAL DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            );

            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                steps_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                last_run TEXT,
                run_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                cron_expression TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                last_run TEXT,
                next_run TEXT,
                run_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_task_steps_task_id ON task_steps(task_id);
            CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
            CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);
        """)
        await self._db.commit()

        # Seed default templates
        count = await self._db.execute("SELECT COUNT(*) FROM task_templates")
        row = await count.fetchone()
        if row[0] == 0:
            await self._seed_templates()

    async def _seed_templates(self):
        templates = [
            ("Google Search", "Search Google for any query", "search",
             json.dumps([
                 {"action": "navigate", "parameters": {"url": "https://www.google.com"}},
                 {"action": "type", "parameters": {"selector": "textarea[name='q']", "text": "{query}"}},
                 {"action": "press_key", "parameters": {"key": "Enter"}},
                 {"action": "wait", "parameters": {"duration": 2}},
                 {"action": "extract", "parameters": {"target": "search results"}}
             ]),
             json.dumps(["query"])),
            ("Scrape Page", "Extract all data from a webpage", "scraping",
             json.dumps([
                 {"action": "navigate", "parameters": {"url": "{url}"}},
                 {"action": "wait", "parameters": {"duration": 3}},
                 {"action": "extract", "parameters": {"target": "all page data"}}
             ]),
             json.dumps(["url"])),
            ("Take Screenshot", "Navigate to URL and take a screenshot", "utility",
             json.dumps([
                 {"action": "navigate", "parameters": {"url": "{url}"}},
                 {"action": "wait", "parameters": {"duration": 3}}
             ]),
             json.dumps(["url"])),
            ("Fill Form", "Fill out a web form with provided data", "automation",
             json.dumps([
                 {"action": "navigate", "parameters": {"url": "{url}"}},
                 {"action": "wait", "parameters": {"duration": 2}}
             ]),
             json.dumps(["url"])),
            ("Wikipedia Search", "Search Wikipedia for information", "search",
             json.dumps([
                 {"action": "navigate", "parameters": {"url": "https://en.wikipedia.org"}},
                 {"action": "type", "parameters": {"selector": "#searchInput", "text": "{query}"}},
                 {"action": "press_key", "parameters": {"key": "Enter"}},
                 {"action": "wait", "parameters": {"duration": 2}},
                 {"action": "extract", "parameters": {"target": "article content"}}
             ]),
             json.dumps(["query"])),
        ]
        for name, desc, cat, steps, vars_ in templates:
            await self._db.execute(
                "INSERT INTO task_templates (name, description, category, steps_json, variables) VALUES (?, ?, ?, ?, ?)",
                (name, desc, cat, steps, vars_)
            )
        await self._db.commit()

    # ------------------------------------------------------------------ #
    # Tasks
    # ------------------------------------------------------------------ #

    async def save_task(self, task) -> None:
        await self._db.execute("""
            INSERT OR REPLACE INTO tasks (id, description, status, completed_at, steps_taken,
                total_cost, execution_time, result_summary, urls_visited, extracted_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.id, task.description, task.status.value,
            datetime.now().isoformat() if task.end_time else None,
            len(task.steps), task.total_cost,
            (task.end_time - task.start_time) if task.end_time and task.start_time else 0,
            task.result_summary,
            json.dumps(task.context.get('urls_visited', [])),
            json.dumps(task.context.get('extracted_data', []))
        ))

        for step in task.steps:
            await self._db.execute("""
                INSERT INTO task_steps (task_id, step_number, action_type, parameters,
                    success, reasoning, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id, step.get('step', 0), step.get('action', ''),
                json.dumps(step.get('parameters', {})),
                1 if step.get('success') else 0,
                step.get('evaluation', ''),
                step.get('error', '')
            ))

        await self._db.commit()

    async def get_task_history(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        cursor = await self._db.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_task_detail(self, task_id: str) -> Optional[Dict]:
        cursor = await self._db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        if not row:
            return None
        task = dict(row)

        steps_cursor = await self._db.execute(
            "SELECT * FROM task_steps WHERE task_id = ? ORDER BY step_number", (task_id,)
        )
        steps = await steps_cursor.fetchall()
        task['steps'] = [dict(s) for s in steps]
        return task

    # ------------------------------------------------------------------ #
    # Analytics
    # ------------------------------------------------------------------ #

    async def get_analytics(self) -> Dict:
        total = await self._fetchval("SELECT COUNT(*) FROM tasks")
        successful = await self._fetchval("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
        failed = await self._fetchval("SELECT COUNT(*) FROM tasks WHERE status = 'failed'")
        avg_steps = await self._fetchval("SELECT AVG(steps_taken) FROM tasks") or 0
        avg_time = await self._fetchval("SELECT AVG(execution_time) FROM tasks") or 0
        total_cost = await self._fetchval("SELECT SUM(total_cost) FROM tasks") or 0

        # Tasks per day (last 7 days)
        cursor = await self._db.execute("""
            SELECT date(created_at) as day, COUNT(*) as count
            FROM tasks
            WHERE created_at >= datetime('now', '-7 days')
            GROUP BY day ORDER BY day
        """)
        daily = [dict(r) for r in await cursor.fetchall()]

        return {
            'total_tasks': total,
            'successful_tasks': successful,
            'failed_tasks': failed,
            'success_rate': round(successful / max(total, 1) * 100, 1),
            'avg_steps': round(avg_steps, 1),
            'avg_execution_time': round(avg_time, 1),
            'total_cost': round(total_cost, 4),
            'daily_tasks': daily
        }

    async def _fetchval(self, query: str):
        cursor = await self._db.execute(query)
        row = await cursor.fetchone()
        return row[0] if row else None

    # ------------------------------------------------------------------ #
    # Templates
    # ------------------------------------------------------------------ #

    async def get_templates(self) -> List[Dict]:
        cursor = await self._db.execute("SELECT * FROM task_templates ORDER BY usage_count DESC")
        return [dict(r) for r in await cursor.fetchall()]

    async def get_template(self, template_id: int) -> Optional[Dict]:
        cursor = await self._db.execute("SELECT * FROM task_templates WHERE id = ?", (template_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def increment_template_usage(self, template_id: int):
        await self._db.execute("UPDATE task_templates SET usage_count = usage_count + 1 WHERE id = ?", (template_id,))
        await self._db.commit()

    async def save_template(self, name: str, description: str, category: str,
                            steps_json: str, variables: str) -> int:
        cursor = await self._db.execute(
            "INSERT INTO task_templates (name, description, category, steps_json, variables) VALUES (?, ?, ?, ?, ?)",
            (name, description, category, steps_json, variables)
        )
        await self._db.commit()
        return cursor.lastrowid

    # ------------------------------------------------------------------ #
    # Recordings
    # ------------------------------------------------------------------ #

    async def save_recording(self, recording_id: str, name: str, task_id: str,
                             steps_json: str, duration: float):
        await self._db.execute(
            "INSERT INTO recordings (id, name, task_id, steps_json, duration) VALUES (?, ?, ?, ?, ?)",
            (recording_id, name, task_id, steps_json, duration)
        )
        await self._db.commit()

    async def get_recordings(self) -> List[Dict]:
        cursor = await self._db.execute("SELECT * FROM recordings ORDER BY created_at DESC")
        return [dict(r) for r in await cursor.fetchall()]

    async def get_recording(self, recording_id: str) -> Optional[Dict]:
        cursor = await self._db.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    # Workflows
    # ------------------------------------------------------------------ #

    async def save_workflow(self, workflow_id: str, name: str, description: str, steps_json: str):
        await self._db.execute(
            "INSERT OR REPLACE INTO workflows (id, name, description, steps_json) VALUES (?, ?, ?, ?)",
            (workflow_id, name, description, steps_json)
        )
        await self._db.commit()

    async def get_workflows(self) -> List[Dict]:
        cursor = await self._db.execute("SELECT * FROM workflows ORDER BY created_at DESC")
        return [dict(r) for r in await cursor.fetchall()]

    async def get_workflow(self, workflow_id: str) -> Optional[Dict]:
        cursor = await self._db.execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------ #
    # Scheduled Tasks
    # ------------------------------------------------------------------ #

    async def save_scheduled_task(self, task_id: str, name: str, description: str,
                                   cron_expression: str, next_run: str = None):
        await self._db.execute(
            "INSERT OR REPLACE INTO scheduled_tasks (id, name, description, cron_expression, next_run) VALUES (?, ?, ?, ?, ?)",
            (task_id, name, description, cron_expression, next_run)
        )
        await self._db.commit()

    async def get_scheduled_tasks(self) -> List[Dict]:
        cursor = await self._db.execute("SELECT * FROM scheduled_tasks ORDER BY created_at DESC")
        return [dict(r) for r in await cursor.fetchall()]

    async def update_scheduled_task_run(self, task_id: str, next_run: str):
        await self._db.execute(
            "UPDATE scheduled_tasks SET last_run = datetime('now'), run_count = run_count + 1, next_run = ? WHERE id = ?",
            (next_run, task_id)
        )
        await self._db.commit()

    async def toggle_scheduled_task(self, task_id: str, enabled: bool):
        await self._db.execute(
            "UPDATE scheduled_tasks SET enabled = ? WHERE id = ?", (1 if enabled else 0, task_id)
        )
        await self._db.commit()

    async def delete_scheduled_task(self, task_id: str):
        await self._db.execute("DELETE FROM scheduled_tasks WHERE id = ?", (task_id,))
        await self._db.commit()

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    async def close(self):
        if self._db:
            await self._db.close()
