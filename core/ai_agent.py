"""
AI Agent - Groq-powered with robust JSON parsing and anti-loop logic.
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.error("Groq not available")


class ReasoningType(Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"


class ActionType(Enum):
    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    EXTRACT = "extract"
    SELECT = "select"
    PRESS_KEY = "press_key"
    DONE = "done"


@dataclass
class ReasoningStep:
    type: ReasoningType
    premise: str
    conclusion: str
    confidence: float
    evidence: List[str]
    reasoning_chain: List[str]


@dataclass
class ActionPlan:
    action: ActionType
    parameters: Dict[str, Any]
    reasoning: ReasoningStep
    expected_outcome: str
    success_criteria: List[str]
    fallback_actions: List['ActionPlan'] = field(default_factory=list)
    confidence: float = 0.5
    estimated_duration: float = 2.0


class TokenTracker:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.api_calls = 0

    def track_usage(self, input_tokens: int, output_tokens: int, model: str = ""):
        if "70b" in model:
            cost = (input_tokens / 1e6) * 0.59 + (output_tokens / 1e6) * 0.79
        elif "8b" in model:
            cost = (input_tokens / 1e6) * 0.05 + (output_tokens / 1e6) * 0.08
        else:
            cost = 0
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.api_calls += 1

    def get_session_stats(self):
        return {
            'total_tokens': self.total_input_tokens + self.total_output_tokens,
            'total_cost': self.total_cost,
            'api_calls': self.api_calls,
        }


class GroqAIAgent:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile",
                 eval_model: str = "llama-3.1-8b-instant",
                 gemini_api_key: str = "",
                 gemini_model: str = "gemini-2.0-flash",
                 ollama_url: str = "",
                 ollama_model: str = ""):
        if not api_key or api_key == "your-groq-api-key-here":
            raise ValueError("Please set a valid Groq API key in your .env file")
        if not GROQ_AVAILABLE:
            raise ValueError("Groq package not available")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.eval_model = eval_model
        self.token_tracker = TokenTracker()

        # Provider cascade: Groq -> Gemini -> Ollama (local). Each is a
        # best-effort fallback for the layer above. Ollama is last because
        # it's slowest but has no quota.
        self._gemini_key = gemini_api_key if (
            gemini_api_key and gemini_api_key != "your-gemini-api-key-here") else ""
        self._gemini_model = gemini_model

        self._ollama_url = (ollama_url or "").rstrip('/')
        self._ollama_model = ollama_model

        self._gemini_fallback_count = 0
        self._ollama_fallback_count = 0

        providers = [f"Groq:{model}"]
        if self._gemini_key:
            providers.append(f"Gemini:{gemini_model}")
        if self._ollama_url and self._ollama_model:
            providers.append(f"Ollama:{ollama_model}")
        logger.info(f"AI cascade ready - {' -> '.join(providers)}")

    # ------------------------------------------------------------------ #
    # Analyze page and decide next action
    # ------------------------------------------------------------------ #

    async def analyze_page_text(self, page_state: Dict, task_goal: str,
                                context: Dict) -> Dict[str, Any]:
        elements = self._format_elements(page_state.get('elements', []))
        history = self._format_history(context.get('action_history', []))
        repeat_warn = self._repeat_warning(context.get('action_history', []))
        hints = self._format_human_hints(context.get('human_inputs', []))
        last_failure = self._format_last_failure(context.get('action_history', []))

        prompt = f"""You are a web automation agent. Decide the SINGLE best next action.

GOAL: {task_goal}
{hints}
PAGE:
- URL: {page_state.get('url', 'unknown')}
- Title: {page_state.get('title', 'unknown')}
- Content: {page_state.get('content', '')[:900]}

ELEMENTS:
{elements}

HISTORY:
{history or 'None - first step.'}
{last_failure}{repeat_warn}

ACTIONS:
- navigate: {{"url": "https://..."}}
- click: {{"selector": "css"}}
- type: {{"selector": "css", "text": "..."}}
- scroll: {{"direction": "down|up"}}
- press_key: {{"key": "Enter|Tab"}}
- extract: {{"target": "what"}}
- select: {{"selector": "css", "value": "..."}}
- done: {{"summary": "what was accomplished"}}

RULES:
1. Use EXACT selectors from the elements list. Prefer #id, then [name=...], then others.
2. After typing in a search box, ALWAYS use press_key Enter next (or click the submit button).
3. If the page shows the results you need (title/content match the goal), use "done" with a detailed summary of what you found.
4. NEVER scroll more than 3 times total. If scrolling isn't revealing new information, use "done" or "extract".
5. If the page has loaded and shows content matching the goal, use "done" immediately with a full summary.
6. NEVER repeat the same action with the same parameters. Try something different.
7. If you're on Google/Bing and need to search, find the search input and TYPE the query, then press Enter.
8. Be efficient - complete the task in as few steps as possible. Don't hesitate.
9. If you navigated to a page and it loaded (you can see content), proceed to the next logical step immediately.
10. When the page content contains what the user asked for, use "done" with a comprehensive summary including the actual data found.
11. "extract" returns the full page content. Use it when you need to capture data, then use "done" on the next step.
12. Only one action per response. Pick the single most useful next step.

JSON only:
{{"thinking": "...", "action": "...", "parameters": {{}}, "reasoning": "...", "confidence": 0.8, "task_complete": false}}"""

        try:
            resp = await self._call_groq(prompt, model=self.model)
            result = self._parse_json(resp)
            if 'error' in result and 'action' not in result:
                return self._fallback_analysis(task_goal, page_state, context)
            return result
        except RuntimeError as e:
            # Quota-exhaustion: surface a real error so the task fails
            # cleanly instead of pretending to be "done".
            logger.error(f"AI unavailable: {e}")
            return {'error': 'ai_unavailable', 'message': str(e)}
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return self._fallback_analysis(task_goal, page_state, context)

    async def generate_action_plan(self, analysis: Dict, task_goal: str,
                                   context: Dict) -> List[ActionPlan]:
        try:
            action_str = analysis.get('action', 'extract')
            params = analysis.get('parameters', {})
            try:
                action_enum = ActionType(action_str)
            except ValueError:
                action_enum = ActionType.EXTRACT

            reasoning = ReasoningStep(
                type=ReasoningType.DEDUCTIVE,
                premise=analysis.get('thinking', ''),
                conclusion=analysis.get('reasoning', f'Execute {action_str}'),
                confidence=analysis.get('confidence', 0.5),
                evidence=[], reasoning_chain=[]
            )
            plan = ActionPlan(
                action=action_enum, parameters=params, reasoning=reasoning,
                expected_outcome=analysis.get('reasoning', ''),
                success_criteria=["No errors"],
                confidence=analysis.get('confidence', 0.5)
            )
            return [plan]
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return self._fallback_plan(task_goal)

    async def evaluate_action_success(self, plan: ActionPlan,
                                      exec_result: Dict, new_state: Dict) -> Dict:
        success = exec_result.get('success', False)
        if not success:
            return {"success": False, "confidence": 0.1,
                    "reasoning": f"Action failed: {exec_result.get('error', '')}"}

        prompt = f"""Did this browser action succeed?
ACTION: {plan.action.value} {json.dumps(plan.parameters)}
RESULT: {json.dumps(exec_result)}
NEW PAGE: {new_state.get('url', '?')} - {new_state.get('title', '?')}
Content: {new_state.get('content', '')[:500]}

JSON: {{"success": true/false, "confidence": 0.0-1.0, "reasoning": "..."}}"""

        try:
            resp = await self._call_groq(prompt, model=self.eval_model)
            result = self._parse_json(resp)
            if 'success' in result:
                return result
        except Exception:
            pass
        return {"success": success, "confidence": 0.7 if success else 0.2,
                "reasoning": "Evaluated from execution result"}

    async def check_completion(self, task: str, actions: List[Dict],
                               state: Dict) -> Dict:
        history = "\n".join(
            f"  {i+1}. {a.get('action','?')}: {a.get('summary','?')}"
            for i, a in enumerate(actions[-8:]))

        prompt = f"""Has this task been completed?
TASK: {task}
ACTIONS (last 8):
{history}
PAGE: {state.get('url', '?')} - {state.get('title', '?')}
Content: {state.get('content', '')[:800]}

JSON: {{"completed": true/false, "confidence": 0.0-1.0, "summary": "..."}}"""

        try:
            resp = await self._call_groq(prompt, model=self.eval_model)
            result = self._parse_json(resp)
            if 'completed' in result:
                return result
        except Exception:
            pass
        return {"completed": False, "confidence": 0.3, "summary": f"Ran {len(actions)} actions"}

    async def get_error_recovery_plan(self, error: str, plan: ActionPlan,
                                       state: Dict, goal: str) -> Dict:
        elements = self._format_elements(state.get('elements', []))
        prompt = f"""Action FAILED. Suggest ONE alternative.
GOAL: {goal}
FAILED: {plan.action.value} {json.dumps(plan.parameters)}
ERROR: {error}
PAGE: {state.get('url', '?')}
ELEMENTS:
{elements}

JSON: {{"action": "...", "parameters": {{}}, "reasoning": "..."}}"""

        try:
            resp = await self._call_groq(prompt, model=self.model)
            return self._parse_json(resp)
        except Exception:
            return {"action": "done", "parameters": {"summary": f"Could not recover from error for: {goal}"}, "reasoning": "Recovery failed"}

    # ------------------------------------------------------------------ #
    # Groq API
    # ------------------------------------------------------------------ #

    _RETRY_AFTER_RE = re.compile(r'(?:retry[- ]after|try again in)[^0-9]*([0-9]*\.?[0-9]+)\s*(ms|s)?', re.I)

    def _parse_retry_after(self, err: Exception) -> Optional[float]:
        """Extract a retry delay (seconds) from a Groq rate-limit error.
        Groq embeds hints like 'try again in 6.2s' or returns a Retry-After
        header on the underlying HTTP response.
        """
        # Header on the APIError's response
        resp = getattr(err, 'response', None)
        if resp is not None:
            headers = getattr(resp, 'headers', None) or {}
            ra = headers.get('retry-after') or headers.get('Retry-After')
            if ra:
                try:
                    return float(ra)
                except ValueError:
                    pass
        # Fallback: parse the message text
        m = self._RETRY_AFTER_RE.search(str(err))
        if m:
            try:
                val = float(m.group(1))
                if (m.group(2) or '').lower() == 'ms':
                    val /= 1000.0
                return val
            except ValueError:
                return None
        return None

    async def _call_groq(self, prompt: str, model: str = None, retries: int = 3) -> str:
        """Try Groq with retry. On exhausted rate-limits, fall back to Gemini
        if configured. On daily quota exhaustion, short-circuit immediately -
        no point burning 45 seconds of retries for an error that won't clear
        for hours."""
        use_model = model or self.model
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                def sync_call():
                    resp = self.client.chat.completions.create(
                        model=use_model,
                        messages=[
                            {"role": "system", "content": "You are a precise web automation agent. Respond with valid JSON only."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=600,
                    )
                    usage = resp.usage
                    if usage:
                        self.token_tracker.track_usage(usage.prompt_tokens,
                                                      usage.completion_tokens, use_model)
                    return resp.choices[0].message.content

                return await asyncio.get_event_loop().run_in_executor(None, sync_call)
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                is_rate_limit = '429' in msg or 'rate' in msg or 'too many' in msg
                is_daily = 'tokens per day' in msg or 'tpd' in msg or 'requests per day' in msg or 'rpd' in msg
                is_5xx = any(c in msg for c in ('500', '502', '503', '504'))
                if is_daily:
                    logger.warning(f"Groq daily quota exhausted - skipping retries on {use_model}")
                    break  # try Gemini below; don't sleep
                if is_rate_limit:
                    if attempt == retries - 1 and (self._gemini_key or self._ollama_model):
                        # Try Gemini first, then Ollama, before giving up.
                        if self._gemini_key:
                            try:
                                logger.warning("Groq rate-limited; falling back to Gemini")
                                return await self._call_gemini(prompt)
                            except Exception as ge:
                                logger.warning(f"Gemini failed: {str(ge)[:120]}")
                        if self._ollama_url and self._ollama_model:
                            try:
                                logger.warning("Gemini also failed; trying local Ollama")
                                return await self._call_ollama(prompt)
                            except Exception as oe:
                                logger.warning(f"Ollama failed: {str(oe)[:120]}")
                        break
                    hint = self._parse_retry_after(e)
                    wait = min(max(hint if hint else (2 ** attempt), 1.0), 15.0)
                    logger.warning(f"Groq rate-limited ({use_model}); retry in {wait:.1f}s "
                                   f"(attempt {attempt + 1}/{retries})")
                    await asyncio.sleep(wait)
                    continue
                if is_5xx and attempt < retries - 1:
                    await asyncio.sleep(1.0 + attempt)
                    continue
                raise
        # Try Gemini if we exited the loop without returning
        gemini_err = None
        if self._gemini_key:
            try:
                logger.warning("Groq failed; using Gemini fallback")
                return await self._call_gemini(prompt)
            except Exception as ge:
                gemini_err = ge
                logger.warning(f"Gemini fallback failed: {str(ge)[:120]}")

        # Last-resort: Ollama (local, no quota)
        if self._ollama_url and self._ollama_model:
            try:
                logger.warning("Upstream AI unavailable; using local Ollama fallback")
                return await self._call_ollama(prompt)
            except Exception as oe:
                logger.warning(f"Ollama fallback failed: {str(oe)[:120]}")
                last_err = self._classify_quota_error(last_err, gemini_err or oe)

        if gemini_err:
            last_err = self._classify_quota_error(last_err, gemini_err)
        raise last_err if last_err else Exception("All AI providers exhausted")

    def _classify_quota_error(self, groq_err, gemini_err) -> Exception:
        """Both providers failed - produce a single clear message."""
        groq_daily = any(s in str(groq_err).lower()
                         for s in ('tokens per day', 'tpd', 'requests per day', 'rpd'))
        gemini_429 = '429' in str(gemini_err)
        if groq_daily and gemini_429:
            return RuntimeError(
                "AI quota exhausted: both Groq daily limit AND Gemini rate limit hit. "
                "Wait until tomorrow or use Ollama locally (ollama serve)."
            )
        if groq_daily:
            return RuntimeError(
                "Groq daily token limit exhausted (resets in ~24h). "
                "Add a working Gemini key, or run Ollama locally."
            )
        if gemini_429:
            return RuntimeError(
                "Gemini rate limit hit. Wait a minute and retry."
            )
        return RuntimeError(f"AI providers failed. Groq: {groq_err}. Gemini: {gemini_err}")

    async def _call_ollama(self, prompt: str) -> str:
        """Send the prompt to a local Ollama instance. Last-resort fallback
        - no quota, but slower and usually a smaller model."""
        import httpx
        if not (self._ollama_url and self._ollama_model):
            raise RuntimeError("Ollama fallback not configured")

        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "system": "You are a precise web automation agent. Respond with valid JSON only.",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 600},
        }
        # Local calls can be slow on first token; give them headroom.
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(f"{self._ollama_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            self._ollama_fallback_count += 1
            return data.get('response', '')

    async def _call_gemini(self, prompt: str) -> str:
        """Send the prompt to Gemini. Used as a fallback when Groq 429s."""
        import httpx
        if not self._gemini_key:
            raise RuntimeError("Gemini fallback not configured")

        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self._gemini_model}:generateContent?key={self._gemini_key}")
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text":
                "You are a precise web automation agent. Respond with valid JSON only."}]},
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600},
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # Defensive extraction - Gemini's response shape can vary.
            try:
                text = data['candidates'][0]['content']['parts'][0]['text']
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(f"Unexpected Gemini response: {str(data)[:200]}")
            self._gemini_fallback_count += 1
            return text

    # ------------------------------------------------------------------ #
    # JSON parsing
    # ------------------------------------------------------------------ #

    def _parse_json(self, response: str) -> Dict:
        if not response:
            return {"error": "Empty response"}
        # Direct
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        # Markdown block
        for pat in [r'```json\s*(.*?)\s*```', r'```\s*(.*?)\s*```']:
            m = re.search(pat, response, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(1).strip())
                except json.JSONDecodeError:
                    continue
        # Brace extraction
        start = response.find('{')
        if start >= 0:
            depth = 0
            for i in range(start, len(response)):
                if response[i] == '{': depth += 1
                elif response[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(response[start:i+1])
                        except json.JSONDecodeError:
                            break
        return {"error": "Parse failed", "raw": response[:300]}

    # ------------------------------------------------------------------ #
    # Formatting
    # ------------------------------------------------------------------ #

    def _format_elements(self, elements: List[Dict]) -> str:
        if not elements:
            return "No interactive elements found."
        lines = []
        for i, el in enumerate(elements[:25]):
            tag = el.get('tag_name', '?')
            text = el.get('text', '')[:40]
            attrs = el.get('attributes', {})
            sel = el.get('primary_selector', '')
            eid = attrs.get('id', '')
            best = (f"#{eid}" if eid
                    else f"{tag}[name='{attrs.get('name', '')}']" if attrs.get('name')
                    else sel)

            parts = [f"[{i}] <{tag}>"]
            if text:
                parts.append(f'"{text}"')
            if attrs.get('type'):
                parts.append(f'type={attrs["type"]}')
            if attrs.get('placeholder'):
                parts.append(f'ph="{attrs["placeholder"][:30]}"')
            if attrs.get('href'):
                parts.append(f'href={attrs["href"][:50]}')
            parts.append(f'sel={best}')
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def _format_history(self, history: List[Dict]) -> str:
        if not history: return ""
        return "\n".join(
            f"  {i+1}. {h.get('action','?')} -> {'OK' if h.get('success') else 'FAILED'}: {h.get('summary','')}"
            for i, h in enumerate(history[-6:]))

    def _format_last_failure(self, history: List[Dict]) -> str:
        """If the most recent step failed, surface its error and tell the AI
        not to retry the same selector. This is the single biggest source of
        wasted steps - the AI keeps trying the same failed action."""
        if not history:
            return ""
        last = history[-1]
        if last.get('success'):
            return ""
        action = last.get('action', '?')
        params = last.get('parameters', {}) or {}
        sel = params.get('selector') or params.get('url') or params.get('text', '')
        err = (last.get('result', '') or last.get('summary', ''))[:200]
        return (f"\nLAST STEP FAILED: {action}({sel!s}) - {err}\n"
                f"DO NOT retry that exact selector. Pick a different element from ELEMENTS, "
                f"or try a different action type.\n")

    def _format_human_hints(self, hints: List[Dict]) -> str:
        if not hints:
            return ""
        recent = hints[-3:]
        lines = [f"  - {h.get('text', '')}" for h in recent if h.get('text')]
        if not lines:
            return ""
        return "\nHUMAN HINTS (follow these - the user just provided them):\n" + "\n".join(lines) + "\n"

    def _repeat_warning(self, history: List[Dict]) -> str:
        if len(history) < 2: return ""
        recent = [h.get('action', '') for h in history[-4:]]
        from collections import Counter
        warnings = []
        for act, cnt in Counter(recent).items():
            if cnt >= 2:
                warnings.append(
                    f"\nWARNING: '{act}' used {cnt} times recently. "
                    f"Do NOT use '{act}' again unless absolutely necessary. "
                    f"Use 'done' if the task looks complete, or try a completely different action.")
        return "".join(warnings)

    # ------------------------------------------------------------------ #
    # Fallbacks
    # ------------------------------------------------------------------ #

    def _extract_url_from_goal(self, goal: str) -> str:
        """Try to extract a target URL from the task goal."""
        goal_lower = goal.lower()
        # Common site mappings
        sites = {
            'wikipedia': 'https://en.wikipedia.org',
            'amazon': 'https://www.amazon.com',
            'google': 'https://www.google.com',
            'github': 'https://github.com',
            'youtube': 'https://www.youtube.com',
            'twitter': 'https://twitter.com',
            'reddit': 'https://www.reddit.com',
            'hacker news': 'https://news.ycombinator.com',
            'ycombinator': 'https://news.ycombinator.com',
            'stackoverflow': 'https://stackoverflow.com',
            'stack overflow': 'https://stackoverflow.com',
        }
        for name, url in sites.items():
            if name in goal_lower:
                return url
        # Check for explicit URLs
        import re
        url_match = re.search(r'https?://[^\s]+', goal)
        if url_match:
            return url_match.group(0)
        # Check for domain-like patterns
        domain_match = re.search(r'(?:go to |navigate to |open |visit )([a-zA-Z0-9.-]+\.[a-z]{2,})', goal_lower)
        if domain_match:
            return 'https://' + domain_match.group(1)
        return 'https://www.google.com'

    def _fallback_analysis(self, goal: str, state: Dict = None, ctx: Dict = None) -> Dict:
        url = state.get('url', '') if state else ''
        history = ctx.get('action_history', []) if ctx else []

        if not history or url in ('', 'about:blank', 'error'):
            target_url = self._extract_url_from_goal(goal)
            return {"thinking": "Starting fresh, navigating to target", "action": "navigate",
                    "parameters": {"url": target_url},
                    "reasoning": f"Navigate to {target_url} for: {goal}",
                    "confidence": 0.8, "task_complete": False}

        # Only bail out when history is genuinely exhausted. The main loop
        # already enforces max_steps, so this fallback should only 'done'
        # when we're near that limit AND the recent actions look idle.
        scroll_count = sum(1 for h in history[-5:] if h.get('action') == 'scroll')
        recent_idle = scroll_count >= 4 or (
            len(history) >= 15
            and all(h.get('action') in ('scroll', 'wait', 'extract') for h in history[-5:])
        )
        if recent_idle:
            return {"thinking": "Idle pattern detected, marking done",
                    "action": "done",
                    "parameters": {"summary": f"Completed: {goal}. Page: {url}"},
                    "reasoning": "Recent actions are idle; no further progress expected",
                    "confidence": 0.6, "task_complete": True}

        # If on a search page, try to type the goal as a search query
        content = state.get('content', '') if state else ''
        elements = state.get('elements', []) if state else []
        for el in elements:
            attrs = el.get('attributes', {})
            if el.get('tag_name') == 'input' and attrs.get('type', 'text') in ('text', 'search', ''):
                sel = f"#{attrs['id']}" if attrs.get('id') else el.get('primary_selector', 'input[type=text]')
                return {"thinking": "Found search input, typing query",
                        "action": "type", "parameters": {"selector": sel, "text": goal[:80]},
                        "reasoning": f"Typing search query into {sel}",
                        "confidence": 0.6, "task_complete": False}

        return {"thinking": "Fallback: marking done", "action": "done",
                "parameters": {"summary": f"Attempted: {goal}. Page: {url}"},
                "reasoning": "Could not determine next action",
                "confidence": 0.5, "task_complete": True}

    def _fallback_plan(self, goal: str) -> List[ActionPlan]:
        r = ReasoningStep(type=ReasoningType.DEDUCTIVE, premise="Fallback",
                          conclusion="Navigate to Google", confidence=0.6,
                          evidence=[], reasoning_chain=[])
        return [ActionPlan(action=ActionType.NAVIGATE,
                           parameters={"url": "https://www.google.com"},
                           reasoning=r, expected_outcome="Google loaded",
                           success_criteria=["Page loaded"], confidence=0.8)]

    def get_token_stats(self) -> Dict:
        return self.token_tracker.get_session_stats()
