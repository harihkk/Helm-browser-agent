"""
Browser Engine - Supports built-in Chromium or user's own browser (Brave, Vivaldi, Chrome, etc.)
"""

import asyncio
import base64
import difflib
import json
import os
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from playwright.async_api import TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known browser paths on macOS
BROWSER_PATHS = {
    'brave': '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser',
    'chrome': '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    'edge': '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge',
    'vivaldi': '/Applications/Vivaldi.app/Contents/MacOS/Vivaldi',
    'arc': '/Applications/Arc.app/Contents/MacOS/Arc',
    'opera': '/Applications/Opera.app/Contents/MacOS/Opera',
    'chromium': '/Applications/Chromium.app/Contents/MacOS/Chromium',
}


class PageState:
    def __init__(self, url: str, title: str, content: str, elements: List[Dict],
                 error: str = ""):
        self.url = url
        self.title = title
        self.content = content
        self.elements = elements
        self.error = error
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        return {
            'url': self.url,
            'title': self.title,
            'content': self.content,
            'elements': self.elements,
            'error': self.error,
            'timestamp': self.timestamp.isoformat()
        }

    @property
    def is_error(self) -> bool:
        return bool(self.error) or self.url == 'error'


class AdvancedBrowserEngine:
    def __init__(self, headless: bool = False, screenshots_dir: str = "./screenshots"):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.contexts: Dict[str, BrowserContext] = {}
        self.pages: Dict[str, Page] = {}
        self.headless = headless
        self.screenshots_dir = screenshots_dir
        self._previous_content: Dict[str, str] = {}
        self._alive = False
        self._browser_name = "built-in"  # "built-in", "brave", "vivaldi", etc.

    def get_available_browsers(self) -> List[Dict[str, str]]:
        """Detect which Chromium-based browsers are installed."""
        available = []
        for name, path in BROWSER_PATHS.items():
            if os.path.exists(path):
                available.append({'name': name, 'path': path})
        return available

    @property
    def browser_name(self) -> str:
        return self._browser_name

    @property
    def is_alive(self) -> bool:
        """Check if browser is actually usable."""
        if not self._alive or not self.browser:
            return False
        try:
            return self.browser.is_connected()
        except Exception:
            return False

    async def start(self):
        """Start built-in Chromium."""
        logger.info("Starting browser engine...")
        try:
            if self.playwright:
                try:
                    await self.playwright.stop()
                except Exception:
                    pass

            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
            )
            await self._create_default_context()
            os.makedirs(self.screenshots_dir, exist_ok=True)
            self._alive = True
            logger.info("Browser engine started")
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            self._alive = False

    async def restart(self):
        """Full restart - close everything and start fresh."""
        logger.info("Restarting browser engine...")
        await self.close()
        await asyncio.sleep(0.5)
        await self.start()

    async def launch_browser(self, browser_name: str) -> Dict[str, Any]:
        """Launch a fresh CDP-controlled instance of the user's browser.

        Uses a dedicated temp profile (NOT their real profile) so it works
        even if their normal browser is already open. On any failure, always
        falls back to built-in Chromium so the system stays functional.
        """
        import subprocess
        import tempfile

        browsers = {b['name']: b['path'] for b in self.get_available_browsers()}
        path = browsers.get(browser_name)
        if not path:
            return {"success": False,
                    "error": f"'{browser_name}' not found. Available: {list(browsers.keys())}"}

        logger.info(f"Launching {browser_name} with CDP...")

        # FAIL FAST if the target browser is already running. On macOS
        # LaunchServices routes our subprocess to the existing instance,
        # which silently ignores our --remote-debugging-port flag. Better
        # to tell the user up front than limp along for 15 seconds.
        if await self._is_browser_running(browser_name):
            return {
                "success": False,
                "error": f"{browser_name.title()} is already running. "
                         f"Quit it first (Cmd+Q) and try again, or stay on built-in."
            }

        # Close existing Playwright browser FIRST
        try:
            await self.close()
        except Exception:
            pass
        await asyncio.sleep(0.3)

        cdp_url = "http://localhost:9222"

        async def _fallback(reason: str) -> Dict[str, Any]:
            """Guaranteed recovery - always get a working browser back."""
            logger.warning(f"CDP launch failed ({reason}); restoring built-in")
            # Kill the subprocess we started (if any)
            try:
                if getattr(self, '_browser_process', None):
                    self._browser_process.kill()
                    self._browser_process = None
            except Exception:
                pass
            # Reset Playwright state so start() gets a clean slate
            self._alive = False
            self.browser = None
            self.contexts.clear()
            self.pages.clear()
            try:
                if self.playwright:
                    try:
                        await self.playwright.stop()
                    except Exception:
                        pass
                    self.playwright = None
            except Exception:
                pass
            # Start a fresh built-in
            try:
                await self.start()
            except Exception as e:
                logger.error(f"Built-in restart ALSO failed: {e}")
            self._browser_name = "built-in"
            return {"success": False,
                    "error": f"Could not launch {browser_name}: {reason}. Using built-in instead."}

        try:
            # Free up port 9222 (some other debugger might be squatting it)
            try:
                proc = await asyncio.create_subprocess_shell(
                    "lsof -ti:9222 | xargs kill -9 2>/dev/null",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                await proc.wait()
            except Exception:
                pass
            await asyncio.sleep(0.3)

            # Dedicated temp profile - isolated from the user's real browsing.
            # Their normal browser can stay open without conflict.
            profile_dir = os.path.join(tempfile.gettempdir(), f"agentic-{browser_name}")
            os.makedirs(profile_dir, exist_ok=True)

            try:
                self._browser_process = subprocess.Popen(
                    [path,
                     '--remote-debugging-port=9222',
                     f'--user-data-dir={profile_dir}',
                     '--no-first-run',
                     '--no-default-browser-check',
                     '--new-window',
                     'about:blank'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as e:
                return await _fallback(f"could not spawn binary: {e}")

            # Poll the CDP port. If the user's browser is already running
            # with a different profile, the new process will exit almost
            # immediately - detect that and fall back clearly.
            ready = False
            for i in range(20):  # up to 10s
                await asyncio.sleep(0.5)
                # Did our subprocess die?
                rc = self._browser_process.poll()
                if rc is not None:
                    return await _fallback(
                        f"{browser_name} exited (code {rc}) - likely already "
                        f"running with a different profile. Close it (Cmd+Q) or use built-in.")
                try:
                    import urllib.request
                    req = urllib.request.urlopen(f"{cdp_url}/json/version", timeout=1.5)
                    req.close()
                    ready = True
                    break
                except Exception:
                    continue

            if not ready:
                return await _fallback(f"{browser_name} CDP port didn't come up in 10s")

            # Connect Playwright
            try:
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.connect_over_cdp(cdp_url)
            except Exception as e:
                return await _fallback(f"Playwright CDP connect failed: {e}")

            # Fresh tab in the new window - never drive their real tabs
            try:
                contexts = self.browser.contexts
                ctx = contexts[0] if contexts else await self.browser.new_context(
                    viewport={'width': 1280, 'height': 720})
                page = await ctx.new_page()
                self.contexts["default"] = ctx
                self.pages["default"] = page
            except Exception as e:
                return await _fallback(f"could not open tab: {e}")

            os.makedirs(self.screenshots_dir, exist_ok=True)
            self._alive = True
            self._browser_name = browser_name

            logger.info(f"{browser_name} launched and connected via CDP")
            return {
                "success": True,
                "browser": browser_name,
                "url": page.url,
                "message": f"{browser_name.title()} launched - automation ready",
            }

        except Exception as e:
            return await _fallback(str(e))

    async def switch_to_builtin(self) -> Dict[str, Any]:
        """Switch back to built-in Chromium."""
        if self._browser_name == "built-in":
            return {"success": True, "message": "Already using built-in Chromium"}
        # Kill the CDP browser process
        if hasattr(self, '_browser_process') and self._browser_process:
            try:
                self._browser_process.kill()
                self._browser_process = None
            except Exception:
                pass
        await self.close()
        await asyncio.sleep(0.5)
        await self.start()
        self._browser_name = "built-in"
        return {"success": True, "message": "Switched to built-in Chromium"}

    async def _create_default_context(self):
        """Create the default browsing context."""
        if not self.browser:
            return
        try:
            ctx = await self.browser.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                java_script_enabled=True,
                ignore_https_errors=True
            )
            page = await ctx.new_page()
            self.contexts["default"] = ctx
            self.pages["default"] = page
        except Exception as e:
            logger.error(f"Failed to create context: {e}")

    # ------------------------------------------------------------------ #
    # Page state
    # ------------------------------------------------------------------ #

    async def get_page_state(self, context_id: str = "default") -> PageState:
        page = self.pages.get(context_id)
        if not page:
            return PageState("error", "No Browser", "", [], error="Browser not available")
        try:
            # Quick liveness check
            url = page.url
            title = await page.title()
        except Exception as e:
            return PageState("error", "Browser Closed", "", [], error=str(e))

        try:
            content = await page.content()
            elements = await self._extract_elements(page)
            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup(["script", "style", "noscript", "svg", "path",
                              "header", "footer", "nav", "aside"]):
                tag.decompose()

            # Prefer the dominant content region so nav/footer noise doesn't
            # crowd out the useful text. Fall back to <body> then to the
            # whole document.
            main = (soup.find('main')
                    or soup.find('article')
                    or soup.find(attrs={'role': 'main'})
                    or soup.find('body')
                    or soup)
            clean = ' '.join(main.get_text(separator=' ').split())[:5000]
            return PageState(url=url, title=title, content=clean, elements=elements)
        except Exception as e:
            return PageState(url=url, title=title, content="", elements=[], error=str(e))

    async def _extract_elements(self, page: Page) -> List[Dict]:
        try:
            return await page.evaluate("""
                () => {
                    const elements = [];
                    const selectors = ['button', 'a[href]', 'input', 'select', 'textarea',
                                       '[role="button"]', '[role="link"]', '[role="tab"]',
                                       '[role="menuitem"]', '[onclick]', '[contenteditable="true"]'];
                    const seen = new Set();

                    selectors.forEach(selector => {
                        try {
                            document.querySelectorAll(selector).forEach((el) => {
                                if (seen.has(el)) return;
                                seen.add(el);

                                const rect = el.getBoundingClientRect();
                                if (rect.width <= 0 || rect.height <= 0) return;
                                const style = window.getComputedStyle(el);
                                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return;

                                let bestSelector = '';
                                if (el.id) {
                                    bestSelector = '#' + CSS.escape(el.id);
                                } else if (el.name) {
                                    bestSelector = el.tagName.toLowerCase() + '[name="' + el.name + '"]';
                                } else if (el.getAttribute('aria-label')) {
                                    bestSelector = '[aria-label="' + el.getAttribute('aria-label') + '"]';
                                } else if (el.getAttribute('data-testid')) {
                                    bestSelector = '[data-testid="' + el.getAttribute('data-testid') + '"]';
                                } else if (el.getAttribute('placeholder')) {
                                    bestSelector = el.tagName.toLowerCase() + '[placeholder="' + el.getAttribute('placeholder') + '"]';
                                } else if (el.type && el.tagName.toLowerCase() === 'input') {
                                    bestSelector = 'input[type="' + el.type + '"]';
                                } else {
                                    let path = [];
                                    let current = el;
                                    while (current && current !== document.body && path.length < 3) {
                                        let seg = current.tagName.toLowerCase();
                                        if (current.id) { seg = '#' + CSS.escape(current.id); path.unshift(seg); break; }
                                        if (current.className && typeof current.className === 'string') {
                                            const cls = current.className.trim().split(/\\s+/).slice(0, 2).map(c => '.' + CSS.escape(c)).join('');
                                            if (cls) seg += cls;
                                        }
                                        path.unshift(seg);
                                        current = current.parentElement;
                                    }
                                    bestSelector = path.join(' > ');
                                }

                                elements.push({
                                    primary_selector: bestSelector,
                                    tag_name: el.tagName.toLowerCase(),
                                    text: (el.textContent || '').trim().substring(0, 100),
                                    attributes: {
                                        id: el.id || '',
                                        class: (typeof el.className === 'string' ? el.className : '').substring(0, 100),
                                        type: el.type || '',
                                        href: el.href || '',
                                        name: el.name || '',
                                        value: el.value || '',
                                        placeholder: el.getAttribute('placeholder') || '',
                                        'aria-label': el.getAttribute('aria-label') || '',
                                        role: el.getAttribute('role') || '',
                                        'data-testid': el.getAttribute('data-testid') || ''
                                    },
                                    is_visible: true,
                                    position: {x: Math.round(rect.x), y: Math.round(rect.y)},
                                    size: {width: Math.round(rect.width), height: Math.round(rect.height)}
                                });
                            });
                        } catch (e) {}
                    });
                    return elements.slice(0, 60);
                }
            """)
        except Exception as e:
            logger.error(f"Element extraction failed: {e}")
            return []

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    async def navigate(self, context_id: str, url: str) -> PageState:
        page = self.pages.get(context_id)
        if not page:
            return PageState(url, "No Browser", "", [], error="Browser not available")
        try:
            await page.goto(url, timeout=30000, wait_until='domcontentloaded')
            await self._smart_wait(page)
            return await self.get_page_state(context_id)
        except Exception as e:
            return PageState(url, "Navigation Error", str(e), [], error=str(e))

    # ------------------------------------------------------------------ #
    # Action execution
    # ------------------------------------------------------------------ #

    async def execute_action(self, context_id: str, action_type: str,
                             parameters: Dict) -> Dict[str, Any]:
        page = self.pages.get(context_id)
        if not page:
            return {'success': False, 'error': 'No browser page', 'fatal': True}

        # Quick liveness check
        try:
            _ = page.url
        except Exception:
            return {'success': False, 'error': 'Browser page has been closed', 'fatal': True}

        try:
            result = await self._do_action(page, action_type, parameters)
            return result
        except Exception as e:
            err = str(e)
            fatal = 'has been closed' in err or 'Target closed' in err
            return {'success': False, 'error': err, 'fatal': fatal}

    async def _do_action(self, page: Page, action_type: str, params: Dict) -> Dict:
        if action_type == 'navigate':
            url = params.get('url', '')
            if url:
                # _smart_wait already waits for DOM + networkidle. No extra sleep.
                await page.goto(url, timeout=25000, wait_until='domcontentloaded')
                await self._smart_wait(page)
                return {'success': True, 'action': 'navigate', 'url': url}

        elif action_type == 'click':
            sel = params.get('selector', '')
            if sel:
                try:
                    await page.click(sel, timeout=4000)
                except Exception:
                    text = params.get('text', '')
                    if text:
                        await page.click(f'text="{text}"', timeout=4000)
                    else:
                        raise
                # Let just-triggered JS settle briefly.
                await asyncio.sleep(0.15)
                return {'success': True, 'action': 'click', 'selector': sel}

        elif action_type == 'type':
            sel = params.get('selector', '')
            text = params.get('text', '')
            if sel and text:
                target, resolved_sel = await self._resolve_typeable(page, sel)
                if target is None:
                    return {'success': False, 'action': 'type',
                            'error': f'No typeable element matches: {sel}'}
                try:
                    await target.click(timeout=4000)
                except Exception:
                    pass  # fill() will still focus
                try:
                    await target.fill(text, timeout=5000)
                except Exception as e:
                    return {'success': False, 'action': 'type',
                            'error': f'Fill failed on {resolved_sel}: {e}'}

                # If the target is a search-like input, submit automatically.
                # This avoids the common LLM mistake of typing repeatedly
                # without ever pressing Enter to actually submit the search.
                submitted = False
                if params.get('submit', True):
                    try:
                        info = await target.evaluate(
                            "el => ({type: (el.type||'').toLowerCase(),"
                            " name: (el.name||'').toLowerCase(),"
                            " placeholder: (el.getAttribute('placeholder')||'').toLowerCase(),"
                            " inForm: !!el.form})"
                        )
                        looks_like_search = (
                            info.get('type') in ('search', '')
                            and info.get('inForm')
                            and any(kw in (info.get('name', '') + info.get('placeholder', ''))
                                    for kw in ('search', 'q', 'query', 'find'))
                        )
                        if looks_like_search:
                            await page.keyboard.press('Enter')
                            try:
                                await page.wait_for_load_state(
                                    'domcontentloaded', timeout=3000)
                            except (PlaywrightTimeout, Exception):
                                pass
                            submitted = True
                    except Exception:
                        pass

                return {'success': True, 'action': 'type',
                        'selector': resolved_sel, 'text': text,
                        'submitted': submitted}

        elif action_type == 'select':
            sel = params.get('selector', '')
            value = params.get('value', '')
            if sel:
                await page.select_option(sel, value, timeout=5000)
                return {'success': True, 'action': 'select'}

        elif action_type == 'press_key':
            key = params.get('key', 'Enter')
            await page.keyboard.press(key)
            # Enter often navigates/submits - let the page start loading.
            if key.lower() == 'enter':
                try:
                    await page.wait_for_load_state('domcontentloaded', timeout=3000)
                except (PlaywrightTimeout, Exception):
                    pass
            else:
                await asyncio.sleep(0.15)
            return {'success': True, 'action': 'press_key', 'key': key}

        elif action_type == 'scroll':
            direction = params.get('direction', 'down')
            amount = 600 if direction == 'down' else -600
            await page.evaluate(f'window.scrollBy(0, {amount})')
            await asyncio.sleep(0.15)
            return {'success': True, 'action': 'scroll', 'direction': direction}

        elif action_type == 'wait':
            dur = min(params.get('duration', 0.5), 1)
            await asyncio.sleep(dur)
            return {'success': True, 'action': 'wait', 'duration': dur}

        elif action_type == 'extract':
            state = await self.get_page_state()
            return {'success': True, 'action': 'extract', 'data': {
                'url': state.url, 'title': state.title,
                'content': state.content[:3000], 'element_count': len(state.elements)
            }}

        elif action_type == 'done':
            return {'success': True, 'action': 'done',
                    'summary': params.get('summary', 'Task complete')}

        return {'success': False, 'error': f'Unknown action: {action_type}'}

    # ------------------------------------------------------------------ #
    # Screenshots
    # ------------------------------------------------------------------ #

    async def take_screenshot(self, context_id: str = "default",
                               task_id: str = None, step: int = None,
                               quality: int = 80) -> Optional[str]:
        # When the agent is driving the user's own browser via CDP, taking
        # screenshots brings the tab to focus and disrupts whatever they're
        # doing. They can see the real browser anyway - skip it.
        if self._browser_name != "built-in":
            return None
        page = self.pages.get(context_id)
        if not page:
            return None
        try:
            data = await page.screenshot(type='jpeg', quality=quality)
            if task_id and step is not None:
                d = os.path.join(self.screenshots_dir, task_id)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, f"step_{step:03d}.jpg"), 'wb') as f:
                    f.write(data)
            return base64.b64encode(data).decode('utf-8')
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Page diff
    # ------------------------------------------------------------------ #

    async def get_page_diff(self, context_id: str = "default") -> Optional[Dict]:
        page = self.pages.get(context_id)
        if not page:
            return None
        try:
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            current = soup.get_text().strip()

            prev = self._previous_content.get(context_id, '')
            self._previous_content[context_id] = current

            if not prev:
                return {"changed": True, "diff_summary": "First page load"}

            diff = list(difflib.unified_diff(
                prev.splitlines(), current.splitlines(), lineterm='', n=0
            ))
            added = sum(1 for l in diff if l.startswith('+') and not l.startswith('+++'))
            removed = sum(1 for l in diff if l.startswith('-') and not l.startswith('---'))
            return {
                "changed": added > 0 or removed > 0,
                "diff_summary": f"+{added} -{removed} lines changed"
            }
        except Exception as e:
            return {"changed": True, "diff_summary": f"Diff error: {e}"}

    # ------------------------------------------------------------------ #
    # Data extraction
    # ------------------------------------------------------------------ #

    async def extract_structured_data(self, context_id: str = "default") -> Dict:
        page = self.pages.get(context_id)
        if not page:
            return {"error": "No page"}
        try:
            return await page.evaluate("""
                () => {
                    const result = {tables: [], lists: [], links: [], headings: []};
                    document.querySelectorAll('table').forEach((table, ti) => {
                        const rows = [];
                        table.querySelectorAll('tr').forEach(tr => {
                            const cells = [];
                            tr.querySelectorAll('td, th').forEach(cell => cells.push(cell.textContent.trim()));
                            if (cells.length > 0) rows.push(cells);
                        });
                        if (rows.length > 0) result.tables.push({index: ti, rows: rows.slice(0, 50)});
                    });
                    document.querySelectorAll('ul, ol').forEach((list, li) => {
                        const items = [];
                        list.querySelectorAll('li').forEach(item => items.push(item.textContent.trim().substring(0, 200)));
                        if (items.length > 0) result.lists.push({index: li, items: items.slice(0, 30)});
                    });
                    document.querySelectorAll('a[href]').forEach(a => {
                        if (a.href && a.textContent.trim())
                            result.links.push({text: a.textContent.trim().substring(0, 100), href: a.href});
                    });
                    result.links = result.links.slice(0, 50);
                    document.querySelectorAll('h1, h2, h3').forEach(h => {
                        result.headings.push({level: h.tagName, text: h.textContent.trim().substring(0, 200)});
                    });
                    return result;
                }
            """)
        except Exception as e:
            return {"error": str(e)}

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    async def _is_browser_running(self, browser_name: str) -> bool:
        """Is the target browser already open? If so, CDP launch can't work."""
        # Map to the process name macOS shows in `pgrep`
        process_names = {
            'brave': 'Brave Browser',
            'chrome': 'Google Chrome',
            'vivaldi': 'Vivaldi',
            'edge': 'Microsoft Edge',
            'arc': 'Arc',
            'opera': 'Opera',
            'chromium': 'Chromium',
        }
        pname = process_names.get(browser_name, browser_name)
        try:
            proc = await asyncio.create_subprocess_exec(
                'pgrep', '-x', pname,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=2)
            return bool(out.strip())
        except Exception:
            return False

    async def _resolve_typeable(self, page: Page, sel: str):
        """Return a Locator pointing at an actual typeable element.

        AIs frequently pick a wrapper (a `<form>` or container `<div>`) when
        they meant the input inside it. Try, in order:
          1. The selector itself, if it resolves to <input>/<textarea>/contenteditable
          2. The first matching typeable descendant of the selector
          3. The first visible typeable element on the page
        Returns (locator, selector_string) or (None, '') if nothing fits.
        """
        TYPEABLE = ('input:not([type=hidden]):not([type=submit])'
                    ':not([type=button]):not([type=checkbox]):not([type=radio]),'
                    ' textarea, [contenteditable="true"]')

        # 1) The selector itself
        try:
            base = page.locator(sel).first
            await base.wait_for(state='visible', timeout=2500)
            tag = (await base.evaluate('el => el.tagName.toLowerCase()')) or ''
            editable = await base.evaluate('el => el.isContentEditable === true')
            type_attr = (await base.evaluate('el => (el.type || "").toLowerCase()')) or ''
            if tag in ('input', 'textarea') and type_attr not in (
                    'submit', 'button', 'checkbox', 'radio', 'hidden'):
                return base, sel
            if editable:
                return base, sel
            # 2) Drill into the wrapper for a typeable descendant
            inner = base.locator(TYPEABLE).first
            try:
                await inner.wait_for(state='visible', timeout=1500)
                return inner, f'{sel} >> {TYPEABLE}'
            except Exception:
                pass
        except Exception:
            pass

        # 3) Fallback: any visible typeable element on the page
        try:
            fb = page.locator(TYPEABLE).first
            await fb.wait_for(state='visible', timeout=2500)
            return fb, TYPEABLE
        except Exception:
            return None, ''

    async def _smart_wait(self, page: Page, dom_timeout: float = 2.0,
                           idle_timeout: float = 1.2):
        """DOM ready is the primary signal; networkidle is a short best-effort
        nudge so SPAs have a chance to render. Keep both bounded tightly -
        persistent connections (analytics, websockets) can block idle forever.
        """
        try:
            await page.wait_for_load_state('domcontentloaded',
                                           timeout=int(dom_timeout * 1000))
        except (PlaywrightTimeout, Exception):
            pass
        try:
            await page.wait_for_load_state('networkidle',
                                           timeout=int(idle_timeout * 1000))
        except (PlaywrightTimeout, Exception):
            pass

    async def close(self):
        logger.info("Shutting down browser...")
        self._alive = False
        # Kill CDP browser process if any
        if hasattr(self, '_browser_process') and self._browser_process:
            try:
                self._browser_process.kill()
                self._browser_process = None
            except Exception:
                pass
        try:
            for ctx in self.contexts.values():
                try:
                    await ctx.close()
                except Exception:
                    pass
            self.contexts.clear()
            self.pages.clear()
            if self.browser:
                try:
                    await self.browser.close()
                except Exception:
                    pass
                self.browser = None
            if self.playwright:
                try:
                    await self.playwright.stop()
                except Exception:
                    pass
                self.playwright = None
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
