"""
Session Recorder
================
Record browser automation sessions and export as Python scripts or JSON workflows.
"""

import json
import uuid
import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Records task steps and exports them in various formats."""

    def __init__(self):
        self.active_recordings: Dict[str, Dict] = {}

    def start_recording(self, task_id: str, name: str = "") -> str:
        recording_id = str(uuid.uuid4())[:12]
        self.active_recordings[recording_id] = {
            'id': recording_id,
            'name': name or f"Recording {recording_id}",
            'task_id': task_id,
            'steps': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None,
        }
        return recording_id

    def record_step(self, recording_id: str, action: str, parameters: Dict,
                     success: bool, url: str = ""):
        rec = self.active_recordings.get(recording_id)
        if rec:
            rec['steps'].append({
                'action': action,
                'parameters': parameters,
                'success': success,
                'url': url,
                'timestamp': datetime.now().isoformat()
            })

    def stop_recording(self, recording_id: str) -> Optional[Dict]:
        rec = self.active_recordings.get(recording_id)
        if rec:
            rec['end_time'] = datetime.now().isoformat()
            return rec
        return None

    def export_as_python(self, recording: Dict) -> str:
        """Generate a standalone Python/Playwright script from a recording."""
        steps = recording.get('steps', [])
        if isinstance(steps, str):
            steps = json.loads(steps)

        lines = [
            '"""',
            f'Auto-generated Playwright script: {recording.get("name", "Recording")}',
            f'Generated at: {datetime.now().isoformat()}',
            '"""',
            '',
            'import asyncio',
            'from playwright.async_api import async_playwright',
            '',
            '',
            'async def run():',
            '    async with async_playwright() as p:',
            '        browser = await p.chromium.launch(headless=False)',
            '        context = await browser.new_context(',
            "            viewport={'width': 1280, 'height': 720}",
            '        )',
            '        page = await context.new_page()',
            '',
        ]

        for i, step in enumerate(steps):
            action = step.get('action', '')
            params = step.get('parameters', {})
            lines.append(f'        # Step {i+1}: {action}')

            # Use repr() so quotes, backslashes, and newlines in recorded
            # values can't produce broken Python.
            if action == 'navigate':
                url = params.get('url', '')
                lines.append(f'        await page.goto({url!r})')
                lines.append('        await page.wait_for_load_state("networkidle")')
            elif action == 'click':
                sel = params.get('selector', '')
                lines.append(f'        await page.click({sel!r})')
            elif action == 'type':
                sel = params.get('selector', '')
                text = params.get('text', '')
                lines.append(f'        await page.fill({sel!r}, {text!r})')
            elif action == 'press_key':
                key = params.get('key', 'Enter')
                lines.append(f'        await page.keyboard.press({key!r})')
            elif action == 'scroll':
                direction = params.get('direction', 'down')
                amount = 600 if direction == 'down' else -600
                lines.append(f'        await page.evaluate("window.scrollBy(0, {amount})")')
            elif action == 'select':
                sel = params.get('selector', '')
                val = params.get('value', '')
                lines.append(f'        await page.select_option({sel!r}, {val!r})')
            elif action == 'wait':
                try:
                    dur = float(params.get('duration', 2))
                except (TypeError, ValueError):
                    dur = 2.0
                lines.append(f'        await asyncio.sleep({dur})')
            elif action == 'extract':
                lines.append('        content = await page.content()')
                lines.append('        print("Extracted content length:", len(content))')

            lines.append('        await asyncio.sleep(1)')
            lines.append('')

        lines.extend([
            '        # Cleanup',
            '        await browser.close()',
            '',
            '',
            'if __name__ == "__main__":',
            '    asyncio.run(run())',
            '',
        ])

        return '\n'.join(lines)

    def export_as_json(self, recording: Dict) -> str:
        """Export recording as a JSON workflow definition."""
        steps = recording.get('steps', [])
        if isinstance(steps, str):
            steps = json.loads(steps)

        workflow = {
            'name': recording.get('name', 'Workflow'),
            'description': f'Auto-generated from recording {recording.get("id", "")}',
            'version': '1.0',
            'steps': [
                {
                    'order': i + 1,
                    'action': step.get('action'),
                    'parameters': step.get('parameters', {}),
                }
                for i, step in enumerate(steps)
            ]
        }
        return json.dumps(workflow, indent=2)
