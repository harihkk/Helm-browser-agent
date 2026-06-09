"""
Intent planner for natural-language browser tasks.

The browser agent should not ask an LLM to rediscover obvious routes like
GitHub profiles, repository URLs, direct URLs, or plain web searches. This
module converts high-confidence user intents into deterministic browser
actions before the page-level LLM loop runs.
"""

import re
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
from urllib.parse import parse_qs, quote_plus, unquote_plus, urlparse

from .action_registry import validate_action
from .prompt_normalizer import PromptNormalizer
from . import risk as risk_layer


# Canonical blocker vocabulary (see core/blockers.py). Kept as a per-intent
# hint of what is most likely to go wrong for this task family.
DEFAULT_BLOCKERS = [
    "bad_url",
    "page_not_found",
    "wrong_domain",
    "login_required",
    "captcha_or_bot_protection",
    "no_results",
    "unavailable_option",
    "timeout",
    "unsupported_action",
]


class MissingSuccessConditionError(ValueError):
    """Raised when planning is attempted without a success condition."""


_NORMALIZER = PromptNormalizer()


@dataclass
class BrowserIntent:
    original_prompt: str
    normalized_prompt: str
    task_type: str
    target_site: str = ""
    target_url: str = ""
    target_app: str = ""
    search_query: str = ""
    content_to_type: str = ""
    entity_or_object: str = ""
    constraints: Dict = field(default_factory=dict)
    user_goal: str = ""
    success_condition: str = ""
    risk_level: str = "low"
    requires_confirmation: bool = False
    allowed_actions: List[str] = field(default_factory=list)
    blocker_conditions: List[str] = field(default_factory=lambda: list(DEFAULT_BLOCKERS))
    # ---- Spec pipeline fields (additive; aliases of the above where noted) --
    run_id: str = ""
    primary_action: str = ""
    target_domain: str = ""          # alias of target_site (host only)
    context_scope: str = "web"       # web | current_site | provided_url | page
    expected_output: str = ""
    validation_strategy: str = ""    # name of the validator that gates completion

    def to_dict(self) -> Dict:
        data = asdict(self)
        # Stable public aliases requested by the pipeline contract.
        data["task_family"] = self.task_type
        data["needs_user_confirmation"] = self.requires_confirmation
        data["allowed_action_types"] = list(self.allowed_actions)
        data["likely_blockers"] = list(self.blocker_conditions)
        return data


@dataclass
class IntentAction:
    action: str
    parameters: Dict
    reasoning: str
    thinking: str
    confidence: float = 0.9
    task_complete: bool = False
    success_condition: str = ""
    validation_method: str = ""
    intent: Optional[BrowserIntent] = None

    def to_analysis(self) -> Dict:
        validate_action(self.action, self.parameters)
        return {
            "thinking": self.thinking,
            "action": self.action,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "task_complete": self.task_complete,
            "success_condition": self.success_condition,
            "validation_method": self.validation_method,
            "intent": self.intent.to_dict() if self.intent else None,
        }


class IntentPlanner:
    """High-confidence prompt router for common browser workflows."""

    SITE_ALIASES = {
        "youtube": "youtube.com",
        "reddit": "reddit.com",
        "stackoverflow": "stackoverflow.com",
        "stack overflow": "stackoverflow.com",
        "amazon": "amazon.com",
        "apple": "apple.com",
        "hacker news": "news.ycombinator.com",
        "ycombinator": "news.ycombinator.com",
        "news.ycombinator": "news.ycombinator.com",
        "medium": "medium.com",
        "x": "x.com",
        "twitter": "x.com",
        "docs": "",
    }

    DIRECT_SITE_SEARCHES = {
        "reddit.com": "https://www.reddit.com/search/?q={query}",
        "youtube.com": "https://www.youtube.com/results?search_query={query}",
        "stackoverflow.com": "https://stackoverflow.com/search?q={query}",
        "amazon.com": "https://www.amazon.com/s?k={query}",
        "medium.com": "https://medium.com/search?q={query}",
    }

    GITHUB_STOP_WORDS = {
        "github", "profile", "user", "account", "look", "for", "find",
        "open", "go", "to", "and", "search", "visit", "the", "a", "an",
        "page", "on", "of", "show", "me", "get", "about", "his", "her",
        "their", "repo", "repository", "repositories", "code",
    }

    def plan(self, goal: str, state: Dict = None, history: List[Dict] = None) -> Optional[Dict]:
        state = state or {}
        history = history or []
        goal_text = (goal or "").strip()
        if not goal_text:
            return None
        intent = self.parse_intent(goal_text)
        # Requirement: every task must have a success condition before any
        # execution starts. Fail loudly rather than plan blindly.
        if not intent.success_condition:
            raise MissingSuccessConditionError(
                "Refusing to plan a task without a success condition.")

        for planner in (
            self._plan_google_keep,
            self._plan_amazon_cart,
            self._plan_youtube_media,
            self._plan_github_repo_search,
            self._plan_explicit_url,
            self._plan_github,
            self._plan_linkedin,
            self._plan_wikipedia,
            self._plan_web_search,
            self._plan_direct_site_search,
            self._plan_generic_site_search,
            self._plan_site_home,
        ):
            action = planner(goal_text, state, history)
            if action:
                if not action.intent:
                    action.intent = intent
                if not action.success_condition:
                    action.success_condition = intent.success_condition
                if not action.validation_method:
                    action.validation_method = self._validation_method_for(intent)
                return action.to_analysis()

        fallback = self._plan_generic_fallback(intent, state, history)
        return fallback.to_analysis() if fallback else None

    def parse_intent(self, goal: str) -> BrowserIntent:
        original = goal or ""
        normalized = self._normalize_prompt(original)
        target_url = self._extract_url(original)
        target_site = self._extract_target_domain(normalized)
        target_app = ""
        task_type = "generic_browser_task"
        search_query = ""
        content_to_type = ""
        entity_or_object = ""
        constraints: Dict = {}
        allowed_actions = [
            "navigate", "search_web", "site_search", "click", "type",
            "press_key", "select", "extract", "done", "report_blocker",
        ]
        risk_level = "low"
        requires_confirmation = False

        low = normalized.lower()
        if "keep" in low and any(w in low for w in ("write", "note", "create", "add", "save")):
            task_type = "note_creation"
            target_app = "google_keep"
            target_site = "keep.google.com"
            content_to_type = self._extract_keep_note_text(original)
            allowed_actions += ["write_google_keep_note", "validate_note_created"]
        elif "youtube" in low and any(w in low for w in ("play", "watch", "listen", "search", "find")):
            task_type = "media_playback"
            target_site = "youtube.com"
            search_query = self._extract_youtube_media_query(original)
            allowed_actions += ["play_youtube_result", "ensure_youtube_playback", "validate_media_playing"]
        elif "github" in low and self._extract_github_repo_search(original):
            task_type = "repo_search"
            gh = self._extract_github_repo_search(original)
            target_site = "github.com"
            target_url = target_url or f"https://github.com/{gh['owner']}/{gh['repo']}"
            search_query = gh["query"]
            entity_or_object = f"{gh['owner']}/{gh['repo']}"
            allowed_actions += ["open_first_github_code_result"]
        elif "amazon" in low and any(w in low for w in ("add", "cart", "basket", "buy", "price", "find", "search")):
            task_type = "cart_update" if any(w in low for w in ("add", "cart", "basket")) else "product_search"
            target_site = "amazon.com"
            search_query = self._extract_cart_query(original) or self._extract_site_query(original, "amazon.com")
            constraints.update(self._extract_amazon_constraints(original, search_query))
            search_query = constraints.get("search_query") or search_query
            entity_or_object = search_query
            risk_level = "medium" if task_type == "cart_update" else "low"
            requires_confirmation = task_type == "cart_update"
            allowed_actions += ["add_amazon_item_to_cart", "validate_cart_updated"]
        elif "iphone" in low and ("apple" in low or "apply" in low):
            # No hardcoded Apple product-page slugs - they 404 and rot every
            # product cycle. Treat this as a site search and let the agent open
            # the real page and extract the price.
            task_type = "site_search"
            target_site = "apple.com"
            search_query = self._extract_site_query(original, "apple.com")
        elif target_url:
            task_type = "navigation" if not any(w in low for w in ("search", "find", "look")) else "site_search"
            search_query = self._extract_query_around_url(original, target_url)
        elif any(w in low for w in ("search", "look up", "google", "find")):
            task_type = "site_search" if target_site else "web_search"
            # Use the same robust cleaners the planners use so the intent's
            # search_query never carries command scaffolding (go to, can you,
            # the destination site, an embedded URL, ...).
            if target_site:
                search_query = self._extract_site_query(original, target_site)
            else:
                search_query = self._clean_command_text(self._normalize_prompt(original))
        elif any(w in low for w in ("fill", "submit", "form", "enter")):
            task_type = "form_filling"
            risk_level = "medium"
            requires_confirmation = "submit" in low
        elif any(w in low for w in ("extract", "read", "summarize", "scrape")):
            task_type = "information_extraction"

        if not search_query and task_type in ("web_search", "site_search", "generic_browser_task"):
            search_query = self._clean_command_text(normalized)

        success_condition = self._success_condition_for(
            task_type, target_site, search_query, content_to_type, entity_or_object, target_url)

        # Risk layer is the single source of truth for confirmation. Merge the
        # family-specific inline signal with the verb-based classifier.
        risk = risk_layer.classify_risk(original, task_type)
        if risk["requires_confirmation"]:
            requires_confirmation = True
        if requires_confirmation:
            risk_level = "high"

        target_domain = (target_site or "").removeprefix("www.")
        if target_url:
            context_scope = "provided_url"
        elif target_site and task_type in (
            "site_search", "repo_search", "information_extraction", "form_filling"
        ):
            context_scope = "current_site"
        else:
            context_scope = "web"

        return BrowserIntent(
            original_prompt=original,
            normalized_prompt=normalized,
            task_type=task_type,
            target_site=target_site or "",
            target_url=target_url or "",
            target_app=target_app,
            search_query=search_query,
            content_to_type=content_to_type,
            entity_or_object=entity_or_object,
            constraints=constraints,
            user_goal=normalized,
            success_condition=success_condition,
            risk_level=risk_level,
            requires_confirmation=requires_confirmation,
            allowed_actions=sorted(set(allowed_actions)),
            run_id=uuid.uuid4().hex[:12],
            primary_action=_NORMALIZER.primary_action(normalized),
            target_domain=target_domain,
            context_scope=context_scope,
            expected_output=self._expected_output_for(task_type),
            validation_strategy=self._validation_method_for_type(task_type),
        )

    def _normalize_prompt(self, goal: str) -> str:
        # Delegates to the first-class PromptNormalizer (pipeline stage 1).
        return _NORMALIZER.normalize(goal)

    def _extract_url(self, goal: str) -> str:
        match = re.search(r'https?://[^\s,)]+', goal or "", flags=re.I)
        return match.group(0).rstrip(").],!?;:\"'") if match else ""

    def _extract_query_around_url(self, goal: str, url: str) -> str:
        query = re.sub(re.escape(url), " ", goal, flags=re.I)
        query = re.sub(
            r"\b(?:can|you|please|go|to|open|inside|within|in|search|look|find|for|repo|repository|site|page|url|and|the|a|an)\b",
            " ",
            query,
            flags=re.I,
        )
        return re.sub(r"\s+", " ", query).strip(" .,:;")

    def _clean_command_text(self, goal: str) -> str:
        text = goal or ""
        text = re.sub(r'https?://[^\s,)]+', " ", text, flags=re.I)
        text = re.sub(
            r"\b(?:can|cna|you|u|please|plz|go|got|o|to|up|over|open|navigate|visit|search|look|find|for|me|the|web|google|inside|within|in|and|actually|show|read|extract|from|site|website)\b",
            " ",
            text,
            flags=re.I,
        )
        return re.sub(r"\s+", " ", text).strip(" .,:;")

    def _success_condition_for(self, task_type: str, target_site: str, query: str,
                               content: str, entity: str, target_url: str) -> str:
        if task_type == "media_playback":
            return "A normal YouTube video page is open and playback has started or a clear playback blocker is reported."
        if task_type == "note_creation":
            return f'A note containing "{content}" is visible/saved, or an authentication blocker is reported.'
        if task_type == "repo_search":
            return f"Repo-scoped search results for {query} in {entity} are visible, or a no-results/blocker state is reported."
        if task_type == "cart_update":
            return f"A cart confirmation for the exact requested product ({query or entity}) is visible; optional review/navigation requirements are also satisfied."
        if task_type == "site_search":
            return f"Results scoped to {target_site or target_url} for {query} are visible, or no-results/blocker is reported."
        if task_type == "web_search":
            return f"Search results for {query} are visible, or a no-results/blocker state is reported."
        if task_type == "navigation":
            return f"The target URL {target_url} is loaded and not an error page."
        if task_type == "information_extraction":
            return "Requested visible page information is extracted or a blocker is reported."
        return "The requested browser outcome is verified from page state or a specific blocker is reported."

    def _validation_method_for(self, intent: BrowserIntent) -> str:
        return self._validation_method_for_type(intent.task_type)

    def _validation_method_for_type(self, task_type: str) -> str:
        return {
            "media_playback": "validate_media_playing",
            "note_creation": "validate_note_created",
            "repo_search": "validate_text_visible",
            "cart_update": "validate_cart_updated",
            "site_search": "validate_text_visible",
            "web_search": "validate_text_visible",
            "navigation": "validate_url",
            "information_extraction": "extract_text",
        }.get(task_type, "observe_page")

    def _expected_output_for(self, task_type: str) -> str:
        return {
            "media_playback": "A playing media player on the target video.",
            "note_creation": "A saved note containing the requested text.",
            "repo_search": "Repository-scoped code search results.",
            "cart_update": "A cart confirmation for the verified product.",
            "site_search": "Search results scoped to the target site.",
            "web_search": "Web search results for the cleaned query.",
            "navigation": "The requested page loaded without error.",
            "information_extraction": "Extracted, grounded page content.",
            "form_filling": "A filled (and, if confirmed, submitted) form.",
        }.get(task_type, "The requested browser outcome, verified from page state.")

    def _plan_generic_fallback(self, intent: BrowserIntent, state: Dict,
                               history: List[Dict]) -> Optional[IntentAction]:
        current_url = state.get("url", "")
        last = history[-1] if history else {}

        if intent.target_url and not self._same_url(current_url, intent.target_url):
            return IntentAction(
                action="navigate",
                parameters={"url": intent.target_url},
                reasoning=f"Navigate to the requested URL before acting: {intent.target_url}",
                thinking="Generic fallback found an explicit target URL.",
                confidence=0.82,
                intent=intent,
                success_condition=intent.success_condition,
                validation_method=self._validation_method_for(intent),
            )

        if last.get("action") == "extract" and last.get("success"):
            return IntentAction(
                action="done",
                parameters={"summary": "Extracted the visible page content for the requested task."},
                reasoning="The generic workflow extracted page state after navigation/search.",
                thinking="Generic fallback can complete after extraction.",
                confidence=0.7,
                task_complete=True,
                intent=intent,
                success_condition=intent.success_condition,
                validation_method=self._validation_method_for(intent),
            )

        if intent.target_site and intent.search_query:
            domain = intent.target_site.removeprefix("www.")
            search_url = f"https://www.google.com/search?q={quote_plus('site:' + domain + ' ' + intent.search_query)}"
            if "google.com/search" not in current_url.lower():
                return IntentAction(
                    action="navigate",
                    parameters={"url": search_url},
                    reasoning=f"Use a site-scoped web search for {intent.search_query} on {domain}.",
                    thinking="Generic site-search fallback.",
                    confidence=0.78,
                    intent=intent,
                    success_condition=intent.success_condition,
                    validation_method="validate_text_visible",
                )

        if intent.search_query:
            search_url = f"https://www.google.com/search?q={quote_plus(intent.search_query)}"
            if "google.com/search" not in current_url.lower():
                return IntentAction(
                    action="navigate",
                    parameters={"url": search_url},
                    reasoning=f"Use web search for the cleaned query: {intent.search_query}",
                    thinking="Generic web-search fallback.",
                    confidence=0.76,
                    intent=intent,
                    success_condition=intent.success_condition,
                    validation_method="validate_text_visible",
                )

        return IntentAction(
            action="extract",
            parameters={"target": intent.user_goal or "current page"},
            reasoning="Observe the current page before deciding whether the generic task is complete.",
            thinking="Generic fallback observation step.",
            confidence=0.62,
            intent=intent,
            success_condition=intent.success_condition,
            validation_method="observe_page",
        )

    # ------------------------------------------------------------------ #
    # Planners
    # ------------------------------------------------------------------ #

    def _plan_google_keep(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        goal_lower = goal.lower()
        if "keep" not in goal_lower:
            return None
        if not any(word in goal_lower for word in ("write", "note", "create", "add", "save")):
            return None

        note = self._extract_keep_note_text(goal)
        if not note:
            return None

        current = state.get("url", "")
        host = urlparse(current).netloc.lower()
        last = history[-1] if history else {}

        if last.get("action") == "write_google_keep_note" and last.get("success"):
            return IntentAction(
                action="done",
                parameters={"summary": f"Created a Google Keep note: {note}"},
                reasoning="The Google Keep note action completed successfully.",
                thinking="Google Keep note task is complete.",
                confidence=0.9,
                task_complete=True,
            )

        keep_auth_redirect = (
            host.endswith("accounts.google.com")
            and ("keep.google.com" in current.lower() or "service" in current.lower())
        )
        if host.endswith("keep.google.com") or keep_auth_redirect:
            return IntentAction(
                action="write_google_keep_note",
                parameters={"text": note},
                reasoning="Write the requested text into Google Keep, or report the sign-in wall if Google redirected to authentication.",
                thinking="Google Keep is open or redirected to sign-in; handling the note action.",
                confidence=0.88,
            )

        return IntentAction(
            action="navigate",
            parameters={"url": "https://keep.google.com/"},
            reasoning="Open Google Keep before creating the requested note.",
            thinking="Google Keep note intent detected.",
            confidence=0.9,
        )

    def _plan_amazon_cart(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        goal_lower = goal.lower()
        if "amazon" not in goal_lower:
            return None
        if not any(phrase in goal_lower for phrase in ("add", "cart", "basket")):
            return None

        intent = self.parse_intent(goal)
        query = intent.search_query or self._extract_cart_query(goal)
        if not query:
            return None

        current = state.get("url", "")
        parsed = urlparse(current)
        host = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.lower()
        last = history[-1] if history else {}

        if last.get("action") == "add_amazon_item_to_cart" and last.get("success"):
            return IntentAction(
                action="done",
                parameters={"summary": f"Added the Amazon item matching {query} to the cart."},
                reasoning="The Amazon add-to-cart action completed successfully.",
                thinking="Amazon cart task is complete.",
                confidence=0.9,
                task_complete=True,
            )

        if host.endswith("amazon.com") and ("/s" in path or "/dp/" in path or "/gp/product" in path):
            return IntentAction(
                action="add_amazon_item_to_cart",
                parameters={"query": query, "constraints": intent.constraints},
                reasoning=f"Verify an Amazon product result matches {query}, then add only that product to the cart.",
                thinking="Amazon search/product page is open; select a verified product candidate before cart action.",
                confidence=0.88,
            )

        target = f"https://www.amazon.com/s?k={quote_plus(query)}"
        return IntentAction(
            action="navigate",
            parameters={"url": target},
            reasoning=f"Search Amazon directly for {query} before adding an item to the cart.",
            thinking=f"Amazon add-to-cart intent detected: {query}.",
            confidence=0.9,
        )

    def _plan_youtube_media(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        goal_lower = goal.lower()
        if "youtube" not in goal_lower:
            return None
        wants_play = any(word in goal_lower for word in ("play", "watch", "listen", "open"))
        searchish = any(phrase in goal_lower for phrase in ("search for", "look for", "find", "play"))
        if not (wants_play or searchish):
            return None

        query = self._extract_youtube_media_query(goal)
        if not query:
            return None

        current = state.get("url", "")
        parsed = urlparse(current)
        host = parsed.netloc.lower().removeprefix("www.")
        path = parsed.path.lower()
        last = history[-1] if history else {}

        if last.get("action") == "ensure_youtube_playback" and last.get("success"):
            return IntentAction(
                action="done",
                parameters={"summary": f"Opened and started a YouTube video for {query}."},
                reasoning="The YouTube player was verified after opening the video.",
                thinking="YouTube playback is complete.",
                confidence=0.92,
                task_complete=True,
            )

        if last.get("action") == "play_youtube_result" and last.get("success"):
            return IntentAction(
                action="ensure_youtube_playback",
                parameters={"query": query},
                reasoning="Verify the opened YouTube watch page is playing.",
                thinking="YouTube video opened; ensuring playback.",
                confidence=0.9,
            )

        if host.endswith("youtube.com") and path == "/watch":
            return IntentAction(
                action="ensure_youtube_playback",
                parameters={"query": query},
                reasoning="A YouTube watch page is open; ensure the player starts.",
                thinking="YouTube playback target is open.",
                confidence=0.9,
            )

        if host.endswith("youtube.com") and path == "/results":
            return IntentAction(
                action="play_youtube_result",
                parameters={"query": query},
                reasoning=f"Open the first regular YouTube video result for {query} and start playback.",
                thinking="YouTube search results are open; selecting a video result.",
                confidence=0.9,
            )

        target = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
        return IntentAction(
            action="navigate",
            parameters={"url": target},
            reasoning=f"Search YouTube directly for the media query: {query}",
            thinking=f"YouTube playback intent detected: {query}.",
            confidence=0.93,
        )

    def _plan_github_repo_search(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        goal_lower = goal.lower()
        if "github" not in goal_lower:
            return None
        if not any(word in goal_lower for word in ("search", "look", "find", "open")):
            return None

        intent = self._extract_github_repo_search(goal)
        if not intent:
            return None

        owner = intent["owner"]
        repo = intent["repo"]
        query = intent["query"]
        current = state.get("url", "")
        parsed = urlparse(current)
        path = parsed.path.lower()
        last = history[-1] if history else {}
        target = f"https://github.com/{owner}/{repo}/search?q={quote_plus(query)}&type=code"

        if last.get("action") == "open_first_github_code_result" and last.get("success"):
            return IntentAction(
                action="done",
                parameters={"summary": f"Opened the first GitHub code result for {query} in {owner}/{repo}."},
                reasoning="The first code result from the repo search is open.",
                thinking="GitHub repo search result is open.",
                confidence=0.88,
                task_complete=True,
            )

        github_repo_search_open = (
            parsed.netloc.lower().endswith("github.com")
            and (
                f"/{owner.lower()}/{repo.lower()}/search" in path
                or self._github_global_repo_search_open(parsed, owner, repo)
            )
        )

        if github_repo_search_open:
            if self._wants_to_open_result(goal):
                return IntentAction(
                    action="open_first_github_code_result",
                    parameters={"owner": owner, "repo": repo},
                    reasoning=f"Open the first code result for {query} in {owner}/{repo}.",
                    thinking="GitHub repo search results are open; opening first code result.",
                    confidence=0.86,
                )
            if self._last_successful_extract(history):
                return IntentAction(
                    action="done",
                    parameters={"summary": f"Extracted GitHub code search results for {query} in {owner}/{repo}."},
                    reasoning="The requested GitHub repo search results were extracted.",
                    thinking="GitHub repo search extraction is complete.",
                    confidence=0.84,
                    task_complete=True,
                )
            return IntentAction(
                action="extract",
                parameters={"target": f"code search results for {query} in {owner}/{repo}"},
                reasoning=f"Extract GitHub code search results for {query} in {owner}/{repo}.",
                thinking="GitHub repo search results are open.",
                confidence=0.84,
            )

        return IntentAction(
            action="navigate",
            parameters={"url": target},
            reasoning=f"Use GitHub's repository code search for {query} inside {owner}/{repo}.",
            thinking=f"GitHub repo-search intent detected: {owner}/{repo} / {query}.",
            confidence=0.92,
        )

    def _plan_explicit_url(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        match = re.search(r'https?://[^\s,]+', goal)
        if not match:
            return None
        target = match.group(0).rstrip(").],!?;:\"'")
        current = state.get("url", "")
        if self._same_url(current, target):
            return IntentAction(
                action="done",
                parameters={"summary": f"Opened {target}."},
                reasoning="The requested URL is already open.",
                thinking="Explicit URL is already loaded.",
                confidence=0.95,
                task_complete=True,
            )
        return IntentAction(
            action="navigate",
            parameters={"url": target},
            reasoning=f"Navigate directly to the requested URL: {target}",
            thinking="Explicit URL detected.",
            confidence=0.98,
        )

    def _plan_github(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        if "github" not in goal.lower():
            return None

        intent = self.extract_github_intent(goal)
        user = intent.get("user")
        repo = intent.get("repo")

        if user and self._mentions_user_top_repo(goal):
            current = state.get("url", "")
            if self._github_repo_open_for_user(current, user):
                parsed = urlparse(current)
                parts = [part for part in parsed.path.split("/") if part]
                label = f"{parts[0]}/{parts[1]}"
                return IntentAction(
                    action="done",
                    parameters={"summary": f"Opened top GitHub repository {label}."},
                    reasoning="A repository for the requested GitHub user is open.",
                    thinking=f"Requested top repository for {user} is open.",
                    confidence=0.9,
                    task_complete=True,
                )

            if self._github_user_repos_open(current, user):
                return IntentAction(
                    action="open_top_github_repo",
                    parameters={"user": user},
                    reasoning=(
                        "Open the first repository on the user's repositories page "
                        "after sorting by stars."
                    ),
                    thinking=f"Opening top repository from {user}'s sorted repositories list.",
                    confidence=0.92,
                )

            target = f"https://github.com/{user}?tab=repositories&sort=stargazers"
            return IntentAction(
                action="navigate",
                parameters={"url": target},
                reasoning=(
                    f"Navigate to {user}'s repositories sorted by stars before "
                    "opening the top repository."
                ),
                thinking=f"GitHub top repository intent detected for user {user}.",
                confidence=0.94,
            )

        if self._mentions_trending_repos(goal):
            target = "https://github.com/trending"
            if self._same_url(state.get("url", ""), target):
                return IntentAction(
                    action="extract",
                    parameters={"target": "trending repositories"},
                    reasoning="Extract trending repositories from the current GitHub Trending page.",
                    thinking="GitHub trending page is already open.",
                    confidence=0.88,
                )
            return IntentAction(
                action="navigate",
                parameters={"url": target},
                reasoning="GitHub Trending is the direct destination for top repositories.",
                thinking="GitHub trending repository intent detected.",
                confidence=0.92,
            )

        if not user:
            return None

        if repo:
            target = f"https://github.com/{user}/{repo}"
            label = f"{user}/{repo}"
            if self._github_target_open(state.get("url", ""), user, repo):
                return IntentAction(
                    action="done",
                    parameters={"summary": f"Opened GitHub repository {label}."},
                    reasoning="The current URL matches the requested GitHub repository.",
                    thinking="Requested GitHub repository is already open.",
                    confidence=0.97,
                    task_complete=True,
                )
            return IntentAction(
                action="navigate",
                parameters={"url": target},
                reasoning=(
                    f"Navigate directly to {target}; this prompt names both a "
                    "GitHub profile owner and repository."
                ),
                thinking=f"GitHub repository intent detected: {label}.",
                confidence=0.97,
            )

        target = f"https://github.com/{user}"
        if self._github_target_open(state.get("url", ""), user):
            return IntentAction(
                action="done",
                parameters={"summary": f"Opened GitHub profile {user}."},
                reasoning="The current URL matches the requested GitHub profile.",
                thinking="Requested GitHub profile is already open.",
                confidence=0.95,
                task_complete=True,
            )
        return IntentAction(
            action="navigate",
            parameters={"url": target},
            reasoning=(
                f"Navigate directly to {target}; profile lookup does not need "
                "GitHub advanced search."
            ),
            thinking=f"GitHub profile intent detected: {user}.",
            confidence=0.95,
        )

    def _plan_linkedin(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        goal_lower = goal.lower()
        if "linkedin" not in goal_lower:
            return None

        wants_people = any(
            word in goal_lower
            for word in ("people", "person", "profile", "recruiter", "recruiters", "hiring", "talent")
        )
        searchish = any(
            phrase in goal_lower
            for phrase in ("look for", "search for", "find", "look up")
        )

        if wants_people or searchish:
            query = self._extract_linkedin_query(goal)
            if query:
                target = f"https://www.linkedin.com/search/results/people/?keywords={quote_plus(query)}"
                current = state.get("url", "")
                if "linkedin.com/search/results/people" in current.lower():
                    if self._last_successful_extract(history):
                        return IntentAction(
                            action="done",
                            parameters={"summary": f"Extracted LinkedIn people results for {query}."},
                            reasoning="The requested LinkedIn people results were already extracted.",
                            thinking="LinkedIn people search extraction is complete.",
                            confidence=0.86,
                            task_complete=True,
                        )
                    return IntentAction(
                        action="extract",
                        parameters={"target": f"LinkedIn people results for {query}"},
                        reasoning="Extract visible LinkedIn people results from the current page.",
                        thinking=f"LinkedIn people search is open for: {query}.",
                        confidence=0.82,
                    )
                return IntentAction(
                    action="navigate",
                    parameters={"url": target},
                    reasoning=f"Use LinkedIn's people search URL directly for: {query}",
                    thinking=f"LinkedIn people-search intent detected: {query}.",
                    confidence=0.9,
                )

        target = "https://www.linkedin.com"
        if self._same_url(state.get("url", ""), target):
            return IntentAction(
                action="done",
                parameters={"summary": "Opened LinkedIn."},
                reasoning="The requested LinkedIn page is already open.",
                thinking="LinkedIn home is already loaded.",
                confidence=0.8,
                task_complete=True,
            )
        return IntentAction(
            action="navigate",
            parameters={"url": target},
            reasoning="Navigate directly to LinkedIn.",
            thinking="LinkedIn site intent detected.",
            confidence=0.84,
        )

    def _plan_wikipedia(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        goal_lower = goal.lower()
        if "wikipedia" not in goal_lower:
            return None
        query = self._extract_search_query(goal, ("wikipedia",))
        if not query:
            return None
        target = f"https://en.wikipedia.org/w/index.php?search={quote_plus(query)}"
        if "wikipedia.org" in (state.get("url", "").lower()) and quote_plus(query).lower() in state.get("url", "").lower():
            if self._last_successful_extract(history):
                return IntentAction(
                    action="done",
                    parameters={"summary": f"Extracted Wikipedia search/article results for {query}."},
                    reasoning="The requested Wikipedia results were already extracted.",
                    thinking="Wikipedia extraction is complete.",
                    confidence=0.85,
                    task_complete=True,
                )
            return IntentAction(
                action="done",
                parameters={"summary": f"Opened Wikipedia search for {query}."},
                reasoning="The requested Wikipedia search is already open.",
                thinking="Wikipedia query already loaded.",
                confidence=0.85,
                task_complete=True,
            )
        return IntentAction(
            action="navigate",
            parameters={"url": target},
            reasoning=f"Use Wikipedia's search URL directly for: {query}",
            thinking="Wikipedia search intent detected.",
            confidence=0.9,
        )

    def _plan_web_search(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        goal_lower = goal.lower()
        searchish = (
            goal_lower.startswith("search ")
            or "google search" in goal_lower
            or "search google" in goal_lower
            or "look up" in goal_lower
            or "search the web" in goal_lower
        )
        if not searchish:
            return None
        query = self._extract_search_query(goal, ("google", "search", "the web", "web", "look up"))
        if not query:
            return None
        target = f"https://www.google.com/search?q={quote_plus(query)}"
        if "google.com/search" in state.get("url", "").lower() and quote_plus(query).lower() in state.get("url", "").lower():
            if self._last_successful_extract(history):
                return IntentAction(
                    action="done",
                    parameters={"summary": f"Extracted search results for {query}."},
                    reasoning="The requested search results were already extracted.",
                    thinking="Web search extraction is complete.",
                    confidence=0.84,
                    task_complete=True,
                )
            return IntentAction(
                action="extract",
                parameters={"target": f"search results for {query}"},
                reasoning="Extract results from the current Google search page.",
                thinking="Requested web search is already open.",
                confidence=0.82,
            )
        return IntentAction(
            action="navigate",
            parameters={"url": target},
            reasoning=f"Use Google's search URL directly for: {query}",
            thinking="Plain web search intent detected.",
            confidence=0.88,
        )

    def _plan_generic_site_search(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        """Generic fallback for "go to <site> and look/search/find <thing>".

        This is deliberately after site-specific planners. It prevents the
        agent from landing on a site's homepage and improvising when a focused
        search route is the reliable starting point.
        """
        goal_lower = goal.lower()
        if not any(phrase in goal_lower for phrase in ("look for", "search for", "find", "look up", "open", "visit")):
            return None

        domain = self._extract_target_domain(goal)
        if not domain:
            return None
        if domain in ("github.com", "linkedin.com", "wikipedia.org", "en.wikipedia.org"):
            return None

        query = self._extract_site_query(goal, domain)
        if not query:
            return None

        last = history[-1] if history else {}
        current = state.get("url", "")
        current_host = urlparse(current).netloc.lower().removeprefix("www.")
        if last.get("action") == "open_first_search_result" and last.get("success"):
            if current_host and current_host.endswith(domain.removeprefix("www.")):
                return IntentAction(
                    action="done",
                    parameters={"summary": f"Opened the best result for {query} on {domain}."},
                    reasoning="The first focused search result is already open.",
                    thinking="Generic site search target is open.",
                    confidence=0.86,
                    task_complete=True,
                )

        search_url = f"https://www.google.com/search?q={quote_plus('site:' + domain + ' ' + query)}"
        if "google.com/search" in current.lower():
            if self._last_successful_extract(history):
                return IntentAction(
                    action="done",
                    parameters={"summary": f"Extracted search results for {query} on {domain}."},
                    reasoning="The focused search results were already extracted.",
                    thinking="Generic site-search extraction is complete.",
                    confidence=0.84,
                    task_complete=True,
                )
            if self._wants_to_open_result(goal):
                return IntentAction(
                    action="open_first_search_result",
                    parameters={"domain": domain},
                    reasoning=f"Open the strongest visible result for {query} on {domain}.",
                    thinking="Focused site search results are open; opening first result.",
                    confidence=0.84,
                )
            return IntentAction(
                action="extract",
                parameters={"target": f"search results for {query} on {domain}"},
                reasoning="Extract the visible focused search results.",
                thinking="Focused site search results are open.",
                confidence=0.82,
            )

        return IntentAction(
            action="navigate",
            parameters={"url": search_url},
            reasoning=f"Use focused web search for {query} restricted to {domain}.",
            thinking=f"Generic site-search intent detected: {domain} / {query}.",
            confidence=0.86,
        )

    def _plan_direct_site_search(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        goal_lower = goal.lower()
        if not any(phrase in goal_lower for phrase in ("look for", "search for", "find", "look up", "open", "visit", "dsearch")):
            return None

        domain = self._extract_target_domain(goal)
        if domain not in self.DIRECT_SITE_SEARCHES:
            return None

        query = self._extract_site_query(goal, domain)
        if not query:
            return None

        current = state.get("url", "")
        current_host = urlparse(current).netloc.lower().removeprefix("www.")
        direct_search_open = current_host.endswith(domain) and self._site_search_open(domain, current)
        last = history[-1] if history else {}

        if last.get("action") == "open_first_search_result" and last.get("success"):
            if current_host and current_host.endswith(domain):
                return IntentAction(
                    action="done",
                    parameters={"summary": f"Opened result for {query} on {domain}."},
                    reasoning="A result on the requested site is open.",
                    thinking="Direct site-search target is open.",
                    confidence=0.86,
                    task_complete=True,
                )

        if direct_search_open:
            if self._last_successful_extract(history):
                return IntentAction(
                    action="done",
                    parameters={"summary": f"Extracted {domain} search results for {query}."},
                    reasoning="The requested direct site-search results were already extracted.",
                    thinking=f"{domain} search extraction is complete.",
                    confidence=0.86,
                    task_complete=True,
                )
            if self._wants_to_open_result(goal):
                return IntentAction(
                    action="open_first_search_result",
                    parameters={"domain": domain},
                    reasoning=f"Open the strongest visible result for {query} on {domain}.",
                    thinking=f"{domain} search results are open; opening first result.",
                    confidence=0.84,
                )
            return IntentAction(
                action="extract",
                parameters={"target": f"{domain} search results for {query}"},
                reasoning=f"Extract visible {domain} search results.",
                thinking=f"{domain} search results are open.",
                confidence=0.82,
            )

        target = self.DIRECT_SITE_SEARCHES[domain].format(query=quote_plus(query))
        return IntentAction(
            action="navigate",
            parameters={"url": target},
            reasoning=f"Use {domain}'s own search route for: {query}",
            thinking=f"Direct site-search intent detected: {domain} / {query}.",
            confidence=0.9,
        )

    def _plan_site_home(self, goal: str, state: Dict, history: List[Dict]) -> Optional[IntentAction]:
        sites = {
            "github": "https://github.com",
            "youtube": "https://www.youtube.com",
            "reddit": "https://www.reddit.com",
            "amazon": "https://www.amazon.com",
            "hacker news": "https://news.ycombinator.com",
            "ycombinator": "https://news.ycombinator.com",
            "stackoverflow": "https://stackoverflow.com",
            "stack overflow": "https://stackoverflow.com",
        }
        goal_lower = goal.lower()
        if not any(goal_lower.startswith(prefix) or f" {prefix} " in f" {goal_lower} "
                   for prefix in ("go to", "open", "visit", "navigate to")):
            return None
        for name, target in sites.items():
            if name in goal_lower:
                if self._same_url(state.get("url", ""), target):
                    return IntentAction(
                        action="done",
                        parameters={"summary": f"Opened {name}."},
                        reasoning=f"The requested site {name} is already open.",
                        thinking="Requested site home is already loaded.",
                        confidence=0.82,
                        task_complete=True,
                    )
                return IntentAction(
                    action="navigate",
                    parameters={"url": target},
                    reasoning=f"Navigate directly to {name}.",
                    thinking=f"Known site intent detected: {name}.",
                    confidence=0.84,
                )
        return None

    def _extract_cart_query(self, goal: str) -> str:
        text = goal.strip()
        patterns = [
            r'\badd\s+(?:some|a|an|the)?\s*(.+?)\s+(?:to|into)\s+(?:my\s+)?(?:cart|basket)\b',
            r'\b(?:buy|find|search\s+for)\s+(?:some|a|an|the)?\s*(.+?)(?:\s+on\s+amazon|\s*$)',
            r'\bamazon\b.*?\badd\s+(?:some|a|an|the)?\s*(.+?)(?:\s+(?:and\s+)?(?:open|show|read)\s+reviews?\b|\s+reviews?\b|$)',
        ]
        query = ""
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                query = match.group(1)
                break
        if not query:
            query = self._extract_site_query(text, "amazon.com")
        query = re.sub(
            r'\s+(?:and\s+)?(?:open|show|read)\s+(?:the\s+)?reviews?\b.*$',
            ' ',
            query,
            flags=re.I,
        )
        query = re.sub(r'\s+reviews?\b.*$', ' ', query, flags=re.I)
        query = re.sub(
            r'\b(?:go|to|amazon|and|add|cart|basket|my|please|plz|some|soem)\b',
            ' ',
            query,
            flags=re.I,
        )
        return re.sub(r'\s+', ' ', query).strip()

    def _extract_amazon_constraints(self, goal: str, query: str) -> Dict:
        low = self._normalize_prompt(goal).lower()
        q_low = (query or "").lower()
        constraints: Dict[str, object] = {}

        if re.search(r'\b(?:open|show|read)\s+(?:the\s+)?reviews?\b|\breviews?\b', low):
            constraints["open_reviews"] = True

        core_products = [
            "ipad pro", "ipad air", "ipad mini", "ipad",
            "iphone", "macbook pro", "macbook air", "macbook",
            "airpods", "apple watch",
        ]
        for product in core_products:
            if product in q_low or product in low:
                constraints["core_product"] = product
                if product.startswith(("ipad", "iphone", "macbook", "airpods", "apple watch")):
                    constraints["brand"] = "apple"
                break

        model = re.search(r'\b(m[1-9])\b', q_low or low, re.I)
        if model:
            constraints["model_generation"] = model.group(1).upper()

        if re.search(r'\bbase\s+(?:version|model|configuration|config)\b', low):
            constraints["base_version"] = True
            if constraints.get("core_product", "").startswith("ipad"):
                constraints["storage"] = "256GB"
                constraints["connectivity"] = "Wi-Fi"

        accessory_terms = (
            "case", "cover", "screen protector", "protector", "keyboard", "pencil",
            "stylus", "charger", "cable", "adapter", "dock", "hub", "stand",
            "mount", "sleeve", "folio", "replacement", "paper", "skin",
        )
        asks_accessory = any(term in q_low for term in accessory_terms)
        if constraints.get("core_product") and not asks_accessory:
            constraints["reject_accessories"] = True

        required_terms = []
        if constraints.get("brand"):
            required_terms.append(str(constraints["brand"]))
        core_product = str(constraints.get("core_product") or "")
        if core_product:
            required_terms.extend(core_product.split())
        if constraints.get("model_generation"):
            required_terms.append(str(constraints["model_generation"]).lower())
        constraints["required_terms"] = sorted(set(required_terms))

        search_query = query
        if constraints.get("core_product") == "ipad pro" and constraints.get("model_generation"):
            parts = ["Apple", "iPad Pro", str(constraints["model_generation"])]
            if constraints.get("base_version"):
                parts.extend(["256GB", "Wi-Fi"])
            search_query = " ".join(parts)
        constraints["search_query"] = search_query
        return constraints

    def _extract_youtube_media_query(self, goal: str) -> str:
        text = goal.strip()
        patterns = [
            r'(?:search\s+for|look\s+for|find|play|watch|listen\s+to)\s+(.+?)(?:\s+(?:and\s+)?(?:actually\s+)?(?:play|watch|open)\b|$)',
            r'youtube\s+(?:and\s+)?(.+?)(?:\s+(?:and\s+)?(?:actually\s+)?(?:play|watch|open)\b|$)',
        ]
        query = ""
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                query = match.group(1)
                break
        if not query:
            query = self._extract_site_query(text, "youtube.com")

        query = re.sub(
            r'\b(?:can|cna|cn|u|you|please|plz|go|got|o|to|youtube|and|search|for|look|find|actually|really|please|play|watch|listen|that|this|song|video|music|open)\b',
            ' ',
            query,
            flags=re.I,
        )
        query = re.sub(r'\s+', ' ', query).strip(" -:,")
        return query

    def _extract_keep_note_text(self, goal: str) -> str:
        text = goal.strip()
        patterns = [
            r'(?:write|create|add|save)\s+(?:a\s+)?(?:note\s+)?(?:saying\s+|that\s+says\s+)?["“]?(.+?)["”]?\s+(?:in|to|on)\s+(?:my\s+|the\s+)?(?:google\s+)?keep\b',
            r'(?:in|on)\s+(?:my\s+|the\s+)?(?:google\s+)?keep\s+(?:write|create|add|save)\s+(?:a\s+)?(?:note\s+)?(?:saying\s+|that\s+says\s+)?["“]?(.+?)["”]?$',
            r'(?:google\s+)?keep\s+(?:note\s+)?["“]?(.+?)["”]?$',
        ]
        note = ""
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I)
            if match:
                note = match.group(1)
                break
        # The capture group already isolates the body. Only trim scaffolding at
        # the EDGES - never strip interior words like "to"/"and", which are
        # part of the note the user asked to type.
        note = note.strip(" .,:;\"'“”")
        note = re.sub(r'^(?:please\s+|plz\s+|can\s+(?:you|u)\s+|go\s+to\s+)+', '', note, flags=re.I)
        note = re.sub(
            r'^(?:write|create|add|save)\s+(?:a\s+)?(?:note\s+)?(?:saying\s+|that\s+says\s+)?',
            '', note, flags=re.I)
        note = re.sub(
            r'\s+(?:in|to|on)\s+(?:my\s+|the\s+)?(?:google\s+)?keep\b.*$', '', note, flags=re.I)
        return re.sub(r'\s+', ' ', note).strip(" .,:;\"'“”")

    def _extract_github_repo_search(self, goal: str) -> Dict[str, str]:
        match = re.search(
            r'https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)(?:[/?#][^\s]*)?',
            goal,
            flags=re.I,
        )
        if not match:
            match = re.search(
                r'github(?:\.com)?\s+([A-Za-z0-9_.-]+)\s*/\s*([A-Za-z0-9_.-]+)',
                goal,
                flags=re.I,
            )
        if not match:
            return {}
        owner, repo = match.group(1), match.group(2).rstrip(".;,")
        query = goal
        query = re.sub(r'https?://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:[/?#][^\s]*)?', ' ', query, flags=re.I)
        query = re.sub(rf'\b{re.escape(owner)}\b|\b{re.escape(repo)}\b', ' ', query, flags=re.I)
        query = re.sub(
            r'\b(?:can|cna|u|you|please|plz|go|to|github|repo|repository|inside|within|in|and|search|look|find|for|code|file|files|open|first|result)\b',
            ' ',
            query,
            flags=re.I,
        )
        query = re.sub(r'\s+', ' ', query).strip(" .,:;")
        if not query:
            return {}
        return {"owner": owner, "repo": repo, "query": query}

    def extract_github_intent(self, goal: str) -> Dict[str, str]:
        goal_lower = goal.lower()
        if "github" not in goal_lower:
            return {}

        explicit = re.search(
            r'github\.com/([A-Za-z0-9][A-Za-z0-9-]{0,38})'
            r'(?:/([A-Za-z0-9._-]+))?',
            goal,
            re.I,
        )
        user = explicit.group(1) if explicit else ""
        repo = explicit.group(2) if explicit and explicit.group(2) else ""

        if not user:
            before_profile = re.search(
                r'([A-Za-z0-9][A-Za-z0-9-]{0,38})\s+'
                r'(?:github\s+)?(?:profile|account|user)\b',
                goal,
                re.I,
            )
            if before_profile and before_profile.group(1).lower() not in self.GITHUB_STOP_WORDS:
                user = before_profile.group(1)

        if not user:
            after_lookup = re.search(
                r'(?:look\s+for|search\s+for|find|open|go\s+to)\s+'
                r'([A-Za-z0-9][A-Za-z0-9-]{0,38})'
                r'(?=.*\b(?:profile|account|user)\b)',
                goal,
                re.I,
            )
            if after_lookup and after_lookup.group(1).lower() not in self.GITHUB_STOP_WORDS:
                user = after_lookup.group(1)

        candidates = re.findall(
            r'(?<![A-Za-z0-9-])([A-Za-z0-9][A-Za-z0-9-]{0,38})(?![A-Za-z0-9-])',
            goal,
        )
        usable = [c for c in candidates if c.lower() not in self.GITHUB_STOP_WORDS and not c.isdigit()]
        if not user and usable:
            user = usable[0]

        if not repo:
            repo = self.extract_github_repo_slug(goal, user)

        result = {}
        if user:
            result["user"] = user
        if repo:
            result["repo"] = repo
        return result

    def extract_github_repo_slug(self, goal: str, user: str = "") -> str:
        goal_lower = goal.lower()
        if not any(word in goal_lower for word in ("repo", "repository", "repositories")):
            return ""

        patterns = [
            r'(?:open|find|search\s+for|go\s+to)\s+'
            r'(?:his|her|their|the)?\s*'
            r'(.+?)\s+(?:repository|repo)\b',
            r'(?:repository|repo)\s+(?:named|called)?\s*'
            r'(.+?)(?:$|[,.])',
        ]
        phrase = ""
        for pattern in patterns:
            match = re.search(pattern, goal, re.I)
            if match:
                phrase = match.group(1)
                break
        if not phrase:
            return ""

        phrase = re.sub(
            r'\b(?:github|profile|account|user|open|find|search|for|go|to|and|his|her|their|the|of|on|repo|repository|repositories)\b',
            ' ',
            phrase,
            flags=re.I,
        )
        if user:
            phrase = re.sub(rf'\b{re.escape(user)}\b', ' ', phrase, flags=re.I)
        words = re.findall(r'[A-Za-z0-9._-]+', phrase)
        words = [word.strip('._-') for word in words if word.strip('._-')]
        if not words:
            return ""
        return "-".join(words).replace("_", "-").lower()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _mentions_trending_repos(self, goal: str) -> bool:
        goal_lower = goal.lower()
        return "github" in goal_lower and any(
            phrase in goal_lower
            for phrase in ("top repos", "top repositories", "trending repos", "trending repositories")
        )

    def _mentions_user_top_repo(self, goal: str) -> bool:
        goal_lower = goal.lower()
        return any(
            phrase in goal_lower
            for phrase in (
                "top repo", "top repository", "best repo", "best repository",
                "most starred repo", "most starred repository", "popular repo",
                "popular repository"
            )
        )

    def _extract_search_query(self, goal: str, remove_phrases: tuple) -> str:
        query = goal
        for phrase in remove_phrases:
            # Skip empty phrases: an empty re.escape() yields ``\b\b`` which
            # matches every word boundary and shreds the string.
            if not phrase:
                continue
            query = re.sub(rf'\b{re.escape(phrase)}\b', ' ', query, flags=re.I)
        query = re.sub(r'\b(?:for|about|please|can you|can u|look up|search|go to)\b', ' ', query, flags=re.I)
        query = " ".join(query.split())
        return query.strip(" .,:;")

    def _extract_target_domain(self, goal: str) -> str:
        explicit = re.search(
            r'(?:https?://)?(?:www\.)?([A-Za-z0-9][A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:/[^\s]*)?',
            goal,
            re.I,
        )
        if explicit:
            return explicit.group(1).lower().removeprefix("www.")

        padded = f" {goal.lower()} "
        matches = []
        for alias, domain in self.SITE_ALIASES.items():
            if domain and f" {alias} " in padded:
                matches.append((len(alias), domain))
        if matches:
            return sorted(matches, reverse=True)[0][1]
        return ""

    def _extract_site_query(self, goal: str, domain: str) -> str:
        query = goal
        query = re.sub(r'https?://[^\s]+', ' ', query, flags=re.I)
        query = re.sub(rf'\b(?:www\.)?{re.escape(domain)}\b', ' ', query, flags=re.I)
        root = domain.split('.')[0]
        if root and root not in ("com", "org", "net"):
            query = re.sub(rf'\b{re.escape(root)}\b', ' ', query, flags=re.I)
        for alias, alias_domain in self.SITE_ALIASES.items():
            if alias_domain == domain:
                query = re.sub(rf'\b{re.escape(alias)}\b', ' ', query, flags=re.I)
        query = re.sub(
            r'\b(?:can you|can u|please|go|got|o|to|on|over|an|and|d|dsearch|look|for|search|find|open|visit|site|website|page|result|results)\b',
            ' ',
            query,
            flags=re.I,
        )
        query = " ".join(query.split())
        return query.strip(" .,:;")

    def _wants_to_open_result(self, goal: str) -> bool:
        goal_lower = goal.lower()
        return any(
            phrase in goal_lower
            for phrase in (
                "open", "go to", "visit", "top", "best", "first", "profile",
                "repository", "repo", "page"
            )
        )

    def _last_successful_extract(self, history: List[Dict]) -> bool:
        if not history:
            return False
        last = history[-1]
        return last.get("action") == "extract" and bool(last.get("success"))

    def _github_global_repo_search_open(self, parsed, owner: str, repo: str) -> bool:
        if parsed.path.lower().rstrip("/") != "/search":
            return False
        params = parse_qs(parsed.query or "")
        raw_query = " ".join(params.get("q", []))
        query = unquote_plus(raw_query).lower()
        repo_tokens = {
            f"repo:{owner.lower()}/{repo.lower()}",
            f"{owner.lower()}/{repo.lower()}",
        }
        return any(token in query for token in repo_tokens)

    def _site_search_open(self, domain: str, current: str) -> bool:
        parsed = urlparse(current)
        path = parsed.path.lower()
        if domain == "reddit.com":
            return path.startswith("/search")
        if domain == "youtube.com":
            return path.startswith("/results")
        if domain in ("stackoverflow.com", "amazon.com", "medium.com"):
            return path.startswith("/search") or path.startswith("/s")
        return False

    def _extract_linkedin_query(self, goal: str) -> str:
        query = goal
        query = re.sub(r'\blinked\s*in\b', ' ', query, flags=re.I)
        query = re.sub(r'\blinkedin\b', ' ', query, flags=re.I)
        query = re.sub(
            r'\b(?:can you|can u|please|go|over|to|on|and|look|for|search|find|open|profiles?|people|person|results?)\b',
            ' ',
            query,
            flags=re.I,
        )
        query = " ".join(query.split())
        return query.strip(" .,:;")

    def _same_url(self, current: str, target: str) -> bool:
        if not current or not target:
            return False
        cur = urlparse(current)
        tgt = urlparse(target)
        return cur.netloc.lower().removeprefix("www.") == tgt.netloc.lower().removeprefix("www.") and cur.path.rstrip("/") == tgt.path.rstrip("/")

    def _github_target_open(self, current: str, user: str, repo: str = "") -> bool:
        parsed = urlparse(current)
        if parsed.netloc.lower() not in ("github.com", "www.github.com"):
            return False
        parts = [part for part in parsed.path.split("/") if part]
        if not parts or parts[0].lower() != user.lower():
            return False
        if repo:
            return len(parts) >= 2 and parts[1].lower() == repo.lower()
        return True

    def _github_user_repos_open(self, current: str, user: str) -> bool:
        parsed = urlparse(current)
        if parsed.netloc.lower() not in ("github.com", "www.github.com"):
            return False
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) != 1 or parts[0].lower() != user.lower():
            return False
        return "tab=repositories" in parsed.query.lower()

    def _github_repo_open_for_user(self, current: str, user: str) -> bool:
        parsed = urlparse(current)
        if parsed.netloc.lower() not in ("github.com", "www.github.com"):
            return False
        parts = [part for part in parsed.path.split("/") if part]
        return len(parts) >= 2 and parts[0].lower() == user.lower()
