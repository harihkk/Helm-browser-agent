"""
Browser Engine - Supports built-in Chromium or user's own browser (Brave, Vivaldi, Chrome, etc.)
"""

import asyncio
import base64
import contextlib
import difflib
import json
import os
import re
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from playwright.async_api import async_playwright, Browser, Page, BrowserContext
from playwright.async_api import TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup

from .action_registry import (
    UnsafeTextPayloadError,
    UnsupportedActionError,
    normalize_action_name,
    validate_action,
    validate_text_payload,
)

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
    AMAZON_ACCESSORY_TERMS = (
        "case", "cover", "screen protector", "protector", "keyboard", "pencil",
        "stylus", "charger", "cable", "adapter", "dock", "hub", "stand",
        "mount", "sleeve", "folio", "replacement", "paperlike", "paper",
        "skin", "holder", "tempered glass", "privacy filter",
    )

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
        self._control_mode = "takeover"
        self._browser_process = None
        self._closing = False
        self._closed_by_user = False

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
    def control_mode(self) -> str:
        return self._control_mode

    def set_control_mode(self, mode: str) -> Dict[str, Any]:
        mode = (mode or "").strip().lower().replace("-", "_")
        if mode in ("takeover", "take_over", "control"):
            self._control_mode = "takeover"
        elif mode in ("hands_off", "handoff", "hand_off", "view"):
            self._control_mode = "hands_off"
        else:
            return {"success": False, "error": "Mode must be takeover or hands_off"}
        return {"success": True, "mode": self._control_mode}

    @property
    def is_alive(self) -> bool:
        """Check if browser is actually usable."""
        if self._closed_by_user or not self._alive or not self.browser:
            return False
        try:
            return self.browser.is_connected()
        except Exception:
            return False

    async def start(self):
        """Start built-in Chromium."""
        logger.info("Starting browser engine...")
        try:
            if self.browser or self.playwright or self.contexts or self.pages:
                await self.close()

            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
            )
            self.browser.on("disconnected", self._handle_browser_disconnected)
            self._closed_by_user = False
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

        Uses a dedicated temp profile (NOT their real profile). On macOS we
        ask the user to quit an already-running copy first because some apps
        route new launches into the existing process and ignore our CDP flags.
        On any failure, always falls back to built-in Chromium so the system
        stays functional.
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
                self.browser.on("disconnected", self._handle_browser_disconnected)
                self._closed_by_user = False
            except Exception as e:
                return await _fallback(f"Playwright CDP connect failed: {e}")

            # Fresh tab in the new window - never drive their real tabs
            try:
                contexts = self.browser.contexts
                ctx = contexts[0] if contexts else await self.browser.new_context(
                    viewport={'width': 1280, 'height': 720})
                page = await ctx.new_page()
                self._watch_page_close(context_id="default", page=page)
                self.contexts["default"] = ctx
                self.pages["default"] = page
            except Exception as e:
                return await _fallback(f"could not open tab: {e}")

            os.makedirs(self.screenshots_dir, exist_ok=True)
            self._alive = True
            self._browser_name = browser_name
            self._control_mode = "hands_off"

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
        if self._browser_name == "built-in" and self.is_alive:
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
        self._control_mode = "takeover"
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
            self._watch_page_close(context_id="default", page=page)
            self.contexts["default"] = ctx
            self.pages["default"] = page
        except Exception as e:
            logger.error(f"Failed to create context: {e}")

    async def _ensure_page(self, context_id: str = "default") -> Optional[Page]:
        if self._closed_by_user:
            return None
        if not self.browser or not self.is_alive:
            self._alive = False
            return None

        page = self.pages.get(context_id)
        if page:
            try:
                await page.title()
                return page
            except Exception:
                pass

        old_ctx = self.contexts.pop(context_id, None)
        self.pages.pop(context_id, None)
        if old_ctx:
            try:
                await old_ctx.close()
            except Exception:
                pass

        if context_id == "default":
            await self._create_default_context()
            return self.pages.get("default")

        if not self.browser:
            return None
        try:
            ctx = await self.browser.new_context(
                viewport={'width': 1280, 'height': 720},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                java_script_enabled=True,
                ignore_https_errors=True
            )
            page = await ctx.new_page()
            self._watch_page_close(context_id=context_id, page=page)
            self.contexts[context_id] = ctx
            self.pages[context_id] = page
            return page
        except Exception as e:
            logger.error(f"Failed to recreate page {context_id}: {e}")
            return None

    def _watch_page_close(self, context_id: str, page: Page) -> None:
        """Treat a user-closed default page as a real browser shutdown."""
        page.on("close", lambda _: self._schedule_page_closed(context_id))

    def _schedule_page_closed(self, context_id: str) -> None:
        if self._closing:
            return
        try:
            asyncio.get_running_loop().create_task(
                self._handle_page_closed(context_id))
        except RuntimeError:
            self._alive = False
            self._closed_by_user = context_id == "default"

    async def _handle_page_closed(self, context_id: str) -> None:
        if self._closing:
            return
        self.pages.pop(context_id, None)
        if context_id != "default":
            return

        logger.info("Default browser page was closed; releasing Playwright browser")
        self._closed_by_user = True
        await self.close(mark_manual=True)

    def _handle_browser_disconnected(self, *args) -> None:
        if self._closing:
            return
        logger.info("Browser disconnected")
        try:
            asyncio.get_running_loop().create_task(
                self._cleanup_disconnected_browser())
        except RuntimeError:
            self._alive = False
            self._closed_by_user = True
            self.browser = None
            self.contexts.clear()
            self.pages.clear()

    async def _cleanup_disconnected_browser(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._alive = False
        self._closed_by_user = True
        self.browser = None
        self.contexts.clear()
        self.pages.clear()
        try:
            if self.playwright:
                with contextlib.suppress(Exception):
                    await self.playwright.stop()
                self.playwright = None
            self._terminate_owned_process()
        finally:
            self._closing = False

    def _terminate_owned_process(self) -> None:
        if not self._browser_process:
            return
        proc = self._browser_process
        self._browser_process = None
        with contextlib.suppress(Exception):
            proc.terminate()
            proc.wait(timeout=3)
        if proc.poll() is None:
            with contextlib.suppress(Exception):
                proc.kill()

    # ------------------------------------------------------------------ #
    # Page state
    # ------------------------------------------------------------------ #

    async def get_page_state(self, context_id: str = "default") -> PageState:
        page = await self._ensure_page(context_id)
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
        action_type = normalize_action_name(action_type)
        try:
            validate_action(action_type, parameters or {}, executable_only=True)
        except UnsupportedActionError as e:
            return {'success': False, 'error': str(e), 'fatal': False}

        page = await self._ensure_page(context_id)
        if not page:
            return {'success': False, 'error': 'No browser page', 'fatal': True}

        # Quick liveness check
        try:
            _ = page.url
        except Exception:
            return {'success': False, 'error': 'Browser page has been closed', 'fatal': True}

        for attempt in range(2):
            try:
                result = await self._do_action(page, action_type, parameters)
                return result
            except Exception as e:
                err = str(e)
                fatal = 'has been closed' in err or 'Target closed' in err
                transient_nav = (
                    'Execution context was destroyed' in err
                    or 'navigation' in err.lower()
                )
                if attempt == 0 and transient_nav and not fatal:
                    with contextlib.suppress(Exception):
                        await self._smart_wait(page, dom_timeout=2.0, idle_timeout=1.0)
                    await asyncio.sleep(0.2)
                    continue
                return {'success': False, 'error': err, 'fatal': fatal}

        return {'success': False, 'error': 'Action failed after retry', 'fatal': False}

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
                try:
                    validate_text_payload(
                        text,
                        allow_full_command=bool(params.get('allow_full_command')),
                    )
                except UnsafeTextPayloadError as e:
                    return {'success': False, 'action': 'type', 'error': str(e)}
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

        elif action_type == 'open_top_github_repo':
            user = params.get('user', '')
            repo_url = await page.evaluate("""
                (user) => {
                    const clean = String(user || '').toLowerCase();
                    const hrefs = [...document.querySelectorAll('a[href^="/"]')]
                        .map(a => ({ href: a.getAttribute('href') || '', text: (a.textContent || '').trim() }));
                    const candidates = hrefs
                        .map(item => item.href.split('?')[0].replace(/\\/$/, ''))
                        .filter(href => {
                            const parts = href.split('/').filter(Boolean);
                            return parts.length === 2 && parts[0].toLowerCase() === clean;
                        });
                    const href = candidates[0];
                    return href ? `https://github.com${href}` : '';
                }
            """, user)
            if not repo_url:
                return {
                    'success': False,
                    'action': 'open_top_github_repo',
                    'error': f'Could not find a repository link for {user}',
                }
            await page.goto(repo_url, timeout=25000, wait_until='domcontentloaded')
            await self._smart_wait(page)
            return {'success': True, 'action': 'open_top_github_repo', 'url': repo_url}

        elif action_type == 'open_first_search_result':
            preferred_domain = (params.get('domain') or '').lower().removeprefix('www.')
            result_url = await page.evaluate("""
                (preferredDomain) => {
                    const badHosts = new Set([
                        'google.com', 'www.google.com', 'accounts.google.com',
                        'support.google.com', 'policies.google.com'
                    ]);
                    function normalizeHref(raw) {
                        if (!raw) return '';
                        try {
                            const url = new URL(raw, location.href);
                            if (url.hostname.endsWith('google.com') && url.pathname === '/url') {
                                return url.searchParams.get('q') || '';
                            }
                            return url.href;
                        } catch (_) {
                            return '';
                        }
                    }
                    function allowed(urlText) {
                        try {
                            const url = new URL(urlText);
                            const host = url.hostname.toLowerCase().replace(/^www\\./, '');
                            if (!['http:', 'https:'].includes(url.protocol)) return false;
                            if (badHosts.has(host)) return false;
                            if (preferredDomain && !host.endsWith(preferredDomain)) return false;
                            return true;
                        } catch (_) {
                            return false;
                        }
                    }
                    const anchors = [...document.querySelectorAll('a[href]')];
                    for (const a of anchors) {
                        const text = (a.textContent || '').trim();
                        if (text.length < 2) continue;
                        const href = normalizeHref(a.getAttribute('href'));
                        if (allowed(href)) return href;
                    }
                    return '';
                }
            """, preferred_domain)
            if not result_url:
                return {
                    'success': False,
                    'action': 'open_first_search_result',
                    'error': 'Could not find a usable search result link',
                }
            await page.goto(result_url, timeout=25000, wait_until='domcontentloaded')
            await self._smart_wait(page)
            return {'success': True, 'action': 'open_first_search_result', 'url': result_url}

        elif action_type == 'play_youtube_result':
            return await self._play_youtube_result(page, params)

        elif action_type == 'ensure_youtube_playback':
            return await self._ensure_youtube_playback(page, params)

        elif action_type == 'write_google_keep_note':
            return await self._write_google_keep_note(page, params)

        elif action_type == 'open_first_github_code_result':
            return await self._open_first_github_code_result(page, params)

        elif action_type == 'configure_apple_product':
            return await self._configure_apple_product(page, params)

        elif action_type == 'add_amazon_item_to_cart':
            return await self._add_amazon_item_to_cart(page, params)

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

    async def _write_google_keep_note(self, page: Page, params: Dict) -> Dict[str, Any]:
        text = (params.get('text') or '').strip()
        if not text:
            return {'success': False, 'action': 'write_google_keep_note', 'error': 'No note text provided.'}
        try:
            validate_text_payload(text, allow_full_command=bool(params.get('allow_full_command')))
        except UnsafeTextPayloadError as e:
            return {'success': False, 'action': 'write_google_keep_note', 'error': str(e)}

        page_text = await page.evaluate("() => (document.body?.innerText || '').toLowerCase()")
        if "sign in" in page_text and ("google" in page_text or "keep" in page_text):
            return {
                'success': True,
                'action': 'write_google_keep_note',
                'task_complete': True,
                'summary': 'Google Keep requires sign-in before Helm can create the note. Sign in or use an external browser session, then retry.',
                'data': {'url': page.url},
            }

        result = await page.evaluate("""
            async ({text}) => {
                function visible(el) {
                    const r = el.getBoundingClientRect();
                    const s = getComputedStyle(el);
                    return r.width > 1 && r.height > 1 && s.display !== 'none' && s.visibility !== 'hidden';
                }
                const candidates = [...document.querySelectorAll(
                    '[aria-label*="Take a note" i], [aria-label*="New note" i], [role="textbox"], [contenteditable="true"]'
                )].filter(visible);
                const target = candidates.find(el => /take a note|new note/i.test(el.getAttribute('aria-label') || el.textContent || '')) || candidates[0];
                if (!target) return {ok: false, reason: 'No note editor found'};
                target.scrollIntoView({block: 'center'});
                target.click();
                await new Promise(r => setTimeout(r, 250));
                const editors = [...document.querySelectorAll('[role="textbox"], [contenteditable="true"]')].filter(visible);
                const body = editors.find(el => !/title/i.test(el.getAttribute('aria-label') || '')) || editors[editors.length - 1];
                if (!body) return {ok: false, reason: 'No editable note body found'};
                body.focus();
                document.execCommand('insertText', false, text);
                await new Promise(r => setTimeout(r, 250));
                const done = [...document.querySelectorAll('div[role="button"], button')]
                    .find(el => /done|close/i.test(el.textContent || el.getAttribute('aria-label') || ''));
                if (done) done.click();
                return {ok: true};
            }
        """, {"text": text})
        if not result.get('ok'):
            return {
                'success': False,
                'action': 'write_google_keep_note',
                'error': result.get('reason', 'Could not create Google Keep note.'),
            }
        return {
            'success': True,
            'action': 'write_google_keep_note',
            'task_complete': True,
            'summary': f'Created Google Keep note: {text}',
            'data': {'text': text, 'url': page.url},
        }

    async def _open_first_github_code_result(self, page: Page, params: Dict) -> Dict[str, Any]:
        page_text = await page.evaluate("() => (document.body?.innerText || '').replace(/\\s+/g, ' ').trim()")
        if re.search(r'sign in to search code on github|sign in .* code search', page_text, re.I):
            return {
                'success': True,
                'action': 'open_first_github_code_result',
                'task_complete': True,
                'summary': (
                    'GitHub requires sign-in before Helm can open code search results '
                    'for this repository. Sign in or switch to an external browser session, then retry.'
                ),
                'data': {'url': page.url, 'blocker': 'github_code_search_sign_in_required'},
            }

        result_url = await page.evaluate("""
            () => {
                const anchors = [...document.querySelectorAll('a[href]')];
                for (const a of anchors) {
                    const href = a.getAttribute('href') || '';
                    if (!href) continue;
                    if (/\\/search\\?|\\/issues|\\/pulls|\\/actions|\\/tree\\//.test(href)) continue;
                    if (/\\/blob\\//.test(href) || /\\/commit\\//.test(href)) {
                        return new URL(href, location.href).href;
                    }
                }
                return '';
            }
        """)
        if not result_url:
            return {
                'success': False,
                'action': 'open_first_github_code_result',
                'error': 'Could not find a GitHub code search result to open.',
            }
        await page.goto(result_url, timeout=25000, wait_until='domcontentloaded')
        await self._smart_wait(page)
        return {
            'success': True,
            'action': 'open_first_github_code_result',
            'url': result_url,
            'summary': f'Opened GitHub code result: {result_url}',
        }

    async def _play_youtube_result(self, page: Page, params: Dict) -> Dict[str, Any]:
        query = (params.get('query') or '').strip()
        video_url = await page.evaluate("""
            () => {
                const bad = /shorts|playlist|channel|hashtag|post/i;
                const selectors = [
                    'ytd-video-renderer a#video-title[href^="/watch"]',
                    'ytd-video-renderer a.yt-simple-endpoint[href^="/watch"]',
                    'a#video-title[href^="/watch"]',
                    'a[href^="/watch?v="]'
                ];
                const seen = new Set();
                for (const sel of selectors) {
                    for (const a of document.querySelectorAll(sel)) {
                        const href = a.getAttribute('href') || '';
                        const text = (a.textContent || a.getAttribute('title') || '').trim();
                        if (!href || seen.has(href) || bad.test(href)) continue;
                        seen.add(href);
                        const card = a.closest('ytd-video-renderer, ytd-rich-item-renderer, ytd-compact-video-renderer');
                        const cardText = (card?.innerText || text || '').toLowerCase();
                        if (/sponsored|ad ·|includes paid promotion/.test(cardText)) continue;
                        return new URL(href, location.href).href;
                    }
                }
                return '';
            }
        """)
        if not video_url:
            return {
                'success': False,
                'action': 'play_youtube_result',
                'error': f'Could not find a regular YouTube video result for {query}.',
            }
        await page.goto(video_url, timeout=25000, wait_until='domcontentloaded')
        await self._smart_wait(page, dom_timeout=2.5, idle_timeout=1.0)
        return {
            'success': True,
            'action': 'play_youtube_result',
            'url': video_url,
            'summary': f'Opened YouTube video for {query}.',
        }

    async def _ensure_youtube_playback(self, page: Page, params: Dict) -> Dict[str, Any]:
        query = (params.get('query') or '').strip()
        for _ in range(4):
            state = await page.evaluate("""
                async () => {
                    const video = document.querySelector('video');
                    if (!video) return {found: false, paused: true, currentTime: 0, title: document.title || ''};
                    video.scrollIntoView({block: 'center'});
                    try {
                        if (video.paused) {
                            await video.play();
                        }
                    } catch (_) {}
                    if (video.paused) {
                        const box = video.getBoundingClientRect();
                        video.click();
                    }
                    return {
                        found: true,
                        paused: video.paused,
                        currentTime: video.currentTime || 0,
                        title: document.title || ''
                    };
                }
            """)
            if state.get('found') and not state.get('paused'):
                await asyncio.sleep(0.8)
                progressed = await page.evaluate("""
                    () => {
                        const video = document.querySelector('video');
                        return video ? {paused: video.paused, currentTime: video.currentTime || 0} : {paused: true, currentTime: 0};
                    }
                """)
                if not progressed.get('paused') and progressed.get('currentTime', 0) > state.get('currentTime', 0):
                    return {
                        'success': True,
                        'action': 'ensure_youtube_playback',
                        'task_complete': True,
                        'summary': f'Playing YouTube video for {query}.',
                        'data': {'title': state.get('title', ''), 'url': page.url},
                    }
                return {
                    'success': True,
                    'action': 'ensure_youtube_playback',
                    'task_complete': True,
                    'summary': f'Opened YouTube video for {query}; playback is active or ready in the player.',
                    'data': {'title': state.get('title', ''), 'url': page.url},
                }
            with contextlib.suppress(Exception):
                await page.keyboard.press('k')
            await asyncio.sleep(0.4)

        return {
            'success': True,
            'action': 'ensure_youtube_playback',
            'task_complete': True,
            'summary': f'Opened the YouTube video for {query}, but the player did not start automatically. Click the player once to start playback.',
            'data': {'url': page.url, 'warning': 'YouTube player did not start playback automatically.'},
        }

    def _amazon_query_terms(self, query: str, constraints: Dict[str, Any]) -> List[str]:
        required = [str(t).lower() for t in constraints.get("required_terms", []) if str(t).strip()]
        if required:
            return sorted(set(required))
        stop = {
            "a", "an", "the", "base", "version", "model", "configuration",
            "config", "new", "latest", "official", "amazon", "cart",
        }
        return sorted(set(
            token.lower()
            for token in re.findall(r"[a-z0-9]+", query or "", flags=re.I)
            if len(token) > 1 and token.lower() not in stop
        ))

    def _is_amazon_accessory_candidate(self, title: str, text: str,
                                       constraints: Dict[str, Any]) -> bool:
        if not constraints.get("reject_accessories"):
            return False
        title_low = (title or "").lower()
        text_low = (text or "").lower()
        if any(term in title_low for term in self.AMAZON_ACCESSORY_TERMS):
            return True
        core = str(constraints.get("core_product") or "").lower()
        if core and re.search(rf"\bfor\s+(?:apple\s+)?{re.escape(core)}\b", title_low):
            return True
        accessory_count = sum(1 for term in self.AMAZON_ACCESSORY_TERMS if term in text_low)
        return accessory_count >= 2 and re.search(r"\bfor\s+(?:ipad|iphone|macbook|tablet)\b", text_low)

    def _score_amazon_candidate(self, candidate: Dict[str, Any], query: str,
                                constraints: Dict[str, Any]) -> Dict[str, Any]:
        title = candidate.get("title") or ""
        text = candidate.get("text") or ""
        title_low = title.lower()
        text_low = f"{title} {text}".lower()
        required = self._amazon_query_terms(query, constraints)
        missing = [term for term in required if term not in text_low]

        if self._is_amazon_accessory_candidate(title, text, constraints):
            return {"score": -1000, "missing": missing, "rejected": "accessory"}
        if missing:
            return {"score": -900 + len(required) - len(missing), "missing": missing, "rejected": "missing_terms"}

        score = 20
        for term in required:
            if term in title_low:
                score += 12
            elif term in text_low:
                score += 4

        core = str(constraints.get("core_product") or "").lower()
        if core and core in title_low:
            score += 20
        if constraints.get("brand") and str(constraints["brand"]).lower() in title_low:
            score += 10
        if constraints.get("model_generation") and str(constraints["model_generation"]).lower() in title_low:
            score += 14
        if constraints.get("storage") and str(constraints["storage"]).lower() in text_low.replace(" ", ""):
            score += 5
        if candidate.get("sponsored"):
            score -= 12
        if candidate.get("unavailable"):
            score -= 60

        return {"score": score, "missing": [], "rejected": ""}

    async def _amazon_search_candidates(self, page: Page) -> List[Dict[str, Any]]:
        return await page.evaluate("""
            () => {
                const cards = [...document.querySelectorAll('[data-component-type="s-search-result"]')];
                const fromCard = cards.map((card, index) => {
                    const titleEl = card.querySelector('h2 span, h2 a span, [data-cy="title-recipe"] span');
                    const link = card.querySelector('h2 a[href], a.a-link-normal.s-no-outline[href], a[href*="/dp/"]');
                    const priceEl = card.querySelector('.a-price .a-offscreen');
                    const title = (titleEl?.textContent || '').replace(/\\s+/g, ' ').trim();
                    const text = (card.innerText || '').replace(/\\s+/g, ' ').trim();
                    let url = '';
                    if (link) {
                        const parsed = new URL(link.getAttribute('href'), location.href);
                        if (parsed.pathname.includes('/dp/') || parsed.pathname.includes('/gp/product/')) {
                            url = parsed.href;
                        }
                    }
                    return {
                        index,
                        title,
                        text,
                        url,
                        price: (priceEl?.textContent || '').trim(),
                        sponsored: /sponsored/i.test(text),
                        unavailable: /currently unavailable|out of stock/i.test(text)
                    };
                }).filter(item => item.title && item.url);
                if (fromCard.length) return fromCard.slice(0, 24);
                return [...document.querySelectorAll('a[href*="/dp/"], a[href*="/gp/product/"]')]
                    .slice(0, 20)
                    .map((link, index) => ({
                        index,
                        title: (link.textContent || link.getAttribute('aria-label') || '').replace(/\\s+/g, ' ').trim(),
                        text: (link.closest('div')?.innerText || link.textContent || '').replace(/\\s+/g, ' ').trim(),
                        url: new URL(link.getAttribute('href'), location.href).href,
                        price: '',
                        sponsored: false,
                        unavailable: false
                    }))
                    .filter(item => item.title && item.url);
            }
        """)

    async def _amazon_product_snapshot(self, page: Page) -> Dict[str, str]:
        return await page.evaluate("""
            () => {
                const title = (
                    document.querySelector('#productTitle')?.textContent ||
                    document.querySelector('h1')?.textContent ||
                    document.title ||
                    ''
                ).replace(/\\s+/g, ' ').trim();
                const price = (
                    document.querySelector('#corePriceDisplay_desktop_feature_div .a-offscreen')?.textContent ||
                    document.querySelector('.a-price .a-offscreen')?.textContent ||
                    ''
                ).replace(/\\s+/g, ' ').trim();
                const text = (document.body?.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 8000);
                return {title, price, text, url: location.href};
            }
        """)

    def _amazon_reviews_url(self, product_url: str) -> str:
        parsed = urlparse(product_url or "")
        match = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})", parsed.path, re.I)
        if match:
            return f"{parsed.scheme or 'https'}://{parsed.netloc or 'www.amazon.com'}/product-reviews/{match.group(1)}"
        return product_url.split("#", 1)[0] + "#customerReviews"

    async def _open_amazon_reviews(self, page: Page, product_url: str) -> bool:
        reviews_url = self._amazon_reviews_url(product_url)
        await page.goto(reviews_url, timeout=25000, wait_until='domcontentloaded')
        await self._smart_wait(page, dom_timeout=2.0, idle_timeout=1.0)
        text = await page.evaluate("() => (document.body?.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 5000)")
        return bool(
            re.search(r"customer reviews|top reviews|review this product|global ratings|ratings?", text, re.I)
            or "/product-reviews/" in page.url
            or "#customerReviews" in page.url
        )

    async def _add_amazon_item_to_cart(self, page: Page, params: Dict) -> Dict[str, Any]:
        query = (params.get('query') or '').strip()
        constraints = params.get('constraints') or {}
        if not isinstance(constraints, dict):
            constraints = {}
        open_reviews = bool(constraints.get("open_reviews"))
        current = page.url
        product_url = current
        candidate_score: Dict[str, Any] = {}

        if "/s?" in current or "/s/" in current or "keywords=" in current:
            candidates = await self._amazon_search_candidates(page)
            ranked = []
            for candidate in candidates:
                score = self._score_amazon_candidate(candidate, query, constraints)
                ranked.append((score.get("score", -1000), candidate, score))
            ranked.sort(key=lambda item: item[0], reverse=True)
            best = ranked[0] if ranked else None
            if not best or best[0] < 0:
                rejected = best[2].get("rejected") if best else "no_candidates"
                missing = ", ".join(best[2].get("missing", [])) if best else ""
                detail = f" Missing required terms: {missing}." if missing else ""
                return {
                    'success': False,
                    'action': 'add_amazon_item_to_cart',
                    'error': (
                        f'Could not find a verified Amazon product result for {query}; '
                        f'best candidate was rejected as {rejected}.{detail}'
                    ),
                    'data': {'query': query, 'candidate_count': len(candidates), 'product_match': False},
                }
            product_url = best[1]["url"]
            candidate_score = best[2]
            await page.goto(product_url, timeout=25000, wait_until='domcontentloaded')
            await self._smart_wait(page, dom_timeout=2.0, idle_timeout=1.0)

        snapshot = await self._amazon_product_snapshot(page)
        page_score = self._score_amazon_candidate(snapshot, query, constraints)
        if page_score.get("score", -1000) < 0:
            missing = ", ".join(page_score.get("missing", []))
            detail = f" Missing required terms: {missing}." if missing else ""
            return {
                'success': False,
                'action': 'add_amazon_item_to_cart',
                'error': (
                    f'Amazon product page does not match "{query}" '
                    f'({page_score.get("rejected") or "low_score"}).{detail}'
                ),
                'data': {
                    'query': query,
                    'url': page.url,
                    'title': snapshot.get('title', ''),
                    'product_match': False,
                },
            }
        product_url = snapshot.get("url") or page.url

        for _ in range(5):
            clicked = await page.evaluate("""
                () => {
                    const selectors = [
                        '#add-to-cart-button',
                        'input[name="submit.add-to-cart"]',
                        'input[aria-labelledby*="submit.add-to-cart"]',
                        'button[name="submit.add-to-cart"]'
                    ];
                    for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (!el || el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
                        el.scrollIntoView({block: 'center'});
                        el.click();
                        return true;
                    }
                    const controls = [...document.querySelectorAll('button, input[type="submit"], a[role="button"]')];
                    const add = controls.find(el => /add\\s+to\\s+cart/i.test(el.innerText || el.value || el.getAttribute('aria-label') || ''));
                    if (add && !add.disabled && add.getAttribute('aria-disabled') !== 'true') {
                        add.scrollIntoView({block: 'center'});
                        add.click();
                        return true;
                    }
                    return false;
                }
            """)
            if clicked:
                await self._smart_wait(page, dom_timeout=2.0, idle_timeout=1.2)
                await asyncio.sleep(0.5)
                text = await page.evaluate("() => (document.body?.innerText || '').replace(/\\s+/g, ' ').trim()")
                cart_confirmed = bool(re.search(r'added to cart|cart subtotal|added to basket|proceed to checkout', text, re.I))
                if not cart_confirmed:
                    return {
                        'success': False,
                        'action': 'add_amazon_item_to_cart',
                        'error': f'Clicked Add to Cart for {query}, but Amazon did not show a cart confirmation.',
                        'data': {
                            'query': query,
                            'url': page.url,
                            'title': snapshot.get('title', ''),
                            'product_match': True,
                            'cart_confirmed': False,
                        },
                    }

                reviews_opened = False
                if open_reviews:
                    reviews_opened = await self._open_amazon_reviews(page, product_url)
                    if not reviews_opened:
                        return {
                            'success': False,
                            'action': 'add_amazon_item_to_cart',
                            'error': f'Added {snapshot.get("title") or query} to the cart, but could not open its reviews.',
                            'data': {
                                'query': query,
                                'url': page.url,
                                'title': snapshot.get('title', ''),
                                'product_match': True,
                                'cart_confirmed': True,
                                'reviews_opened': False,
                            },
                        }

                summary = f"Added {snapshot.get('title') or query} to the cart."
                if open_reviews:
                    summary += " Opened the product reviews."
                return {
                    'success': True,
                    'action': 'add_amazon_item_to_cart',
                    'task_complete': True,
                    'summary': summary,
                    'data': {
                        'query': query,
                        'url': page.url,
                        'product_url': product_url,
                        'title': snapshot.get('title', ''),
                        'price': snapshot.get('price', ''),
                        'product_match': True,
                        'cart_confirmed': True,
                        'reviews_opened': reviews_opened,
                        'score': page_score.get('score'),
                        'candidate_score': candidate_score.get('score'),
                    },
                }
            await page.evaluate("window.scrollBy(0, Math.max(420, window.innerHeight * 0.7))")
            await asyncio.sleep(0.2)

        return {
            'success': False,
            'action': 'add_amazon_item_to_cart',
            'error': f'Could not find an Add to Cart button for verified product {query}.',
            'data': {
                'query': query,
                'url': page.url,
                'title': snapshot.get('title', ''),
                'product_match': True,
            },
        }

    async def _configure_apple_product(self, page: Page, params: Dict) -> Dict[str, Any]:
        model = (params.get('model') or '').strip()
        storage = (params.get('storage') or '').strip().replace(" ", "")

        selected = []
        if model:
            clicked = await self._click_visible_text_option(
                page,
                [model],
                reject=[f"{model} Plus"] if "plus" not in model.lower() else [],
                max_scrolls=3,
            )
            if clicked:
                selected.append(model)
                await self._smart_wait(page, dom_timeout=1.0, idle_timeout=0.8)

        if storage:
            storage_patterns = [
                storage,
                storage.replace("GB", " GB").replace("TB", " TB"),
                storage.lower(),
            ]
            clicked = await self._click_visible_text_option(
                page,
                storage_patterns,
                max_scrolls=8,
            )
            if clicked:
                selected.append(storage)
                await self._smart_wait(page, dom_timeout=1.0, idle_timeout=0.8)
            else:
                visible = await self._visible_storage_options(page)
                option_text = f" Visible storage options: {', '.join(visible)}." if visible else ""
                return {
                    'success': True,
                    'action': 'configure_apple_product',
                    'task_complete': True,
                    'summary': (
                        f'Apple does not show a {storage} option for {model} '
                        f'on the current buy page, so there is no direct Apple price for that configuration.'
                        f'{option_text}'
                    ),
                    'data': {'blocker': f'{storage} is not visible on the Apple buy page.'},
                }

        details = await self._read_apple_price_details(page, model, storage)
        if not details.get("price"):
            visible = await self._visible_storage_options(page)
            option_text = f" Visible storage options: {', '.join(visible)}." if visible else ""
            return {
                'success': True,
                'action': 'configure_apple_product',
                'task_complete': True,
                'summary': (
                    f'Configured {", ".join(selected) or model or "Apple product"} '
                    f'but Apple did not show a readable final price.{option_text}'
                ),
                'data': details,
            }

        label = " ".join(part for part in (details.get("model") or model, details.get("storage") or storage) if part)
        summary = f"{label}: {details['price']}"
        if details.get("monthly"):
            summary += f" ({details['monthly']})"
        return {
            'success': True,
            'action': 'configure_apple_product',
            'task_complete': True,
            'summary': summary,
            'data': details,
        }

    async def _click_visible_text_option(self, page: Page, needles: List[str],
                                         reject: List[str] = None,
                                         max_scrolls: int = 5) -> bool:
        reject = reject or []
        script = """
            ({needles, reject}) => {
                const wanted = needles.map(v => String(v || '').toLowerCase()).filter(Boolean);
                const blocked = reject.map(v => String(v || '').toLowerCase()).filter(Boolean);
                const candidates = [...document.querySelectorAll(
                    'button, label, [role="radio"], [role="button"], a, input[type="radio"]'
                )];
                function visible(el) {
                    const r = el.getBoundingClientRect();
                    const s = getComputedStyle(el);
                    return r.width > 1 && r.height > 1 && s.visibility !== 'hidden' && s.display !== 'none';
                }
                function textFor(el) {
                    let text = (el.innerText || el.textContent || '').trim();
                    if (!text && el.id) {
                        const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
                        text = (label?.innerText || label?.textContent || '').trim();
                    }
                    return text.replace(/\\s+/g, ' ');
                }
                for (const el of candidates) {
                    if (!visible(el)) continue;
                    const text = textFor(el);
                    const lower = text.toLowerCase();
                    if (!wanted.some(v => lower.includes(v))) continue;
                    if (blocked.some(v => lower.includes(v))) continue;
                    el.scrollIntoView({block: 'center', inline: 'center'});
                    el.click();
                    return {clicked: true, text};
                }
                return {clicked: false, text: ''};
            }
        """
        for i in range(max_scrolls + 1):
            result = await page.evaluate(script, {"needles": needles, "reject": reject})
            if result.get("clicked"):
                return True
            if i < max_scrolls:
                await page.evaluate("window.scrollBy(0, Math.max(420, window.innerHeight * 0.7))")
                await asyncio.sleep(0.15)
        return False

    async def _read_apple_price_details(self, page: Page, model: str, storage: str) -> Dict[str, str]:
        text = await page.evaluate("""
            () => (document.body?.innerText || '').replace(/\\s+/g, ' ').trim()
        """)
        clean_storage = storage.replace(" ", "")
        price = ""
        monthly = ""

        option_chunks = []
        if clean_storage:
            storage_text = re.escape(clean_storage).replace("GB", r"\s*GB").replace("TB", r"\s*TB")
            option_chunks = re.findall(
                rf'(.{{0,80}}{storage_text}.{{0,180}})',
                text,
                flags=re.I,
            )

        haystacks = option_chunks + [text]
        for chunk in haystacks:
            match = re.search(r'(?:from\s*)?(\$[\d,]+(?:\.\d{2})?)', chunk, flags=re.I)
            if match:
                price = match.group(1)
                monthly_match = re.search(
                    r'(\$[\d,]+(?:\.\d{2})?\s*/mo\.?(?:\s*for\s*\d+\s*mo\.?)?)',
                    chunk,
                    flags=re.I,
                )
                monthly = monthly_match.group(1) if monthly_match else ""
                break

        if not monthly:
            monthly_match = re.search(
                r'(\$[\d,]+(?:\.\d{2})?\s*/mo\.?(?:\s*for\s*\d+\s*mo\.?)?)',
                text,
                flags=re.I,
            )
            monthly = monthly_match.group(1) if monthly_match else ""

        return {
            "model": model,
            "storage": clean_storage,
            "price": price,
            "monthly": monthly,
            "url": page.url,
        }

    async def _visible_storage_options(self, page: Page) -> List[str]:
        try:
            text = await page.evaluate("""
                () => (document.body?.innerText || '').replace(/\\s+/g, ' ').trim()
            """)
            options = []
            for amount, unit in re.findall(r'\\b(\\d+)\\s*(GB|TB)\\b', text, flags=re.I):
                label = f"{amount}{unit.upper()}"
                if label not in options:
                    options.append(label)
            return options[:8]
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    # Screenshots
    # ------------------------------------------------------------------ #

    async def take_screenshot(self, context_id: str = "default",
                               task_id: str = None, step: int = None,
                               quality: int = 80) -> Optional[str]:
        page = await self._ensure_page(context_id)
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

    async def click_viewport(self, x_ratio: float, y_ratio: float,
                             context_id: str = "default") -> Dict[str, Any]:
        """Click the current page using normalized viewport coordinates."""
        if self._control_mode != "takeover":
            return {
                "success": False,
                "error": "Hands off is enabled. Switch to Take over before clicking.",
            }
        page = self.pages.get(context_id)
        if not page:
            return {"success": False, "error": "No browser page"}
        try:
            x_ratio = min(max(float(x_ratio), 0.0), 1.0)
            y_ratio = min(max(float(y_ratio), 0.0), 1.0)
            size = page.viewport_size
            if not size:
                size = await page.evaluate("""
                    () => ({width: window.innerWidth || 1280,
                            height: window.innerHeight || 720})
                """)
            x = round(size["width"] * x_ratio)
            y = round(size["height"] * y_ratio)
            await page.mouse.click(x, y)
            await asyncio.sleep(0.15)
            return {"success": True, "x": x, "y": y}
        except Exception as e:
            return {"success": False, "error": str(e)}

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

    async def close(self, mark_manual: bool = False):
        logger.info("Shutting down browser...")
        if mark_manual:
            self._closed_by_user = True
        self._closing = True
        self._alive = False
        try:
            for ctx in list(self.contexts.values()):
                with contextlib.suppress(Exception):
                    await ctx.close()
            self.contexts.clear()
            self.pages.clear()
            if self.browser:
                with contextlib.suppress(Exception):
                    await self.browser.close()
                self.browser = None
            if self.playwright:
                with contextlib.suppress(Exception):
                    await self.playwright.stop()
                self.playwright = None
            self._terminate_owned_process()
        except Exception as e:
            logger.error(f"Shutdown error: {e}")
        finally:
            self._closing = False
