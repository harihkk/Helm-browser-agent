"""
Validator System - pipeline gate before a run may be called "completed".

Validators are small, composable, page-evidence-based predicates. A run is
COMPLETED only when its task-family validator returns success backed by real
evidence. When the evidence is insufficient to prove success but there is no
hard blocker, the result is UNVERIFIED (never silently "completed").

This module is deliberately free of browser/model imports. ``state`` is
duck-typed: any object exposing ``url``/``title``/``content``/``is_error`` (a
PageState) or a mapping with those keys works.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from .blockers import Blocker, classify_blocker


# Terminal validation statuses.
COMPLETED = "completed"
UNVERIFIED = "unverified"
BLOCKED = "blocked"
FAILED = "failed"


@dataclass
class ValidationResult:
    status: str
    reason: str
    evidence: str = ""
    blocker: Optional[Dict] = field(default=None)

    @property
    def ok(self) -> bool:
        return self.status == COMPLETED


# ------------------------------------------------------------------ #
# state accessors (duck-typed PageState or dict)
# ------------------------------------------------------------------ #

def _get(state: Any, key: str, default: str = "") -> Any:
    if state is None:
        return default
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def _is_error(state: Any) -> bool:
    if state is None:
        return False
    if isinstance(state, dict):
        return bool(state.get("error")) or state.get("url") == "error"
    return bool(getattr(state, "is_error", False))


def _haystack(state: Any) -> str:
    return f"{_get(state, 'url')} {_get(state, 'title')} {_get(state, 'content')}".lower()


# ------------------------------------------------------------------ #
# atomic, composable validators (the public vocabulary)
# ------------------------------------------------------------------ #

def same_url(a: str, b: str) -> bool:
    if not a or not b:
        return False
    pa, pb = urlparse(a), urlparse(b)
    host_a = pa.netloc.lower().removeprefix("www.")
    host_b = pb.netloc.lower().removeprefix("www.")
    return host_a == host_b and pa.path.rstrip("/") == pb.path.rstrip("/")


def validate_url(state: Any, target: str) -> bool:
    return bool(target) and same_url(_get(state, "url"), target)


def validate_domain(state: Any, domain: str) -> bool:
    if not domain:
        return False
    host = urlparse(_get(state, "url")).netloc.lower().removeprefix("www.")
    return host.endswith(domain.removeprefix("www."))


def validate_page_not_error(state: Any) -> bool:
    return not _is_error(state)


def validate_text_visible(state: Any, text: str) -> bool:
    if not text:
        return False
    hay = _haystack(state)
    parts = [p for p in text.lower().split() if p][:4]
    return bool(parts) and all(p in hay for p in parts)


def validate_search_results_visible(state: Any, query: str) -> bool:
    url = _get(state, "url").lower()
    return "search" in url or "/s?" in url or "/results" in url or validate_text_visible(state, query)


def validate_data_extracted(extracted: List) -> bool:
    return bool(extracted)


# A registry so callers/tests can enumerate the supported validators and so
# the planner's ``validation_method`` strings map to real callables.
VALIDATORS: Dict[str, Callable] = {
    "validate_url": validate_url,
    "validate_domain": validate_domain,
    "validate_page_not_error": validate_page_not_error,
    "validate_text_visible": validate_text_visible,
    "validate_search_results_visible": validate_search_results_visible,
    "validate_data_extracted": validate_data_extracted,
}


def _step_succeeded(history: List[Dict], action: str) -> bool:
    return any(s.get("action") == action and s.get("success") for s in (history or []))


# ------------------------------------------------------------------ #
# task-family completion validator
# ------------------------------------------------------------------ #

def validate_completion(intent: Dict, state: Any, history: List[Dict],
                        extracted: List) -> ValidationResult:
    """Decide the terminal status for a run that claims completion.

    Returns COMPLETED only with real evidence, UNVERIFIED when the agent did
    work but could not prove the outcome, and BLOCKED when a hard blocker is
    visible on the page.
    """
    intent = intent or {}
    history = history or []
    extracted = extracted or []
    task_type = intent.get("task_type", "generic_browser_task")
    url = _get(state, "url")
    haystack = _haystack(state)

    # A hard error page or a strong login/captcha/404 wall always wins over any
    # claimed completion - even for a "navigation" task whose URL technically
    # matches (a 404 page still "loaded").
    if _is_error(state):
        blocker = classify_blocker(
            url=url, title=_get(state, "title"), content=_get(state, "content"),
            last_error=_get(state, "error"), is_error=True)
        status = BLOCKED if blocker.status == "blocked" else FAILED
        return ValidationResult(status, blocker.blocker_message,
                                evidence=blocker.visible_evidence,
                                blocker=blocker.to_dict())
    hard = _hard_page_blocker(state)
    if hard is not None:
        status = BLOCKED if hard.status == "blocked" else FAILED
        return ValidationResult(status, hard.blocker_message,
                                evidence=hard.visible_evidence, blocker=hard.to_dict())

    if task_type == "navigation":
        target = intent.get("target_url", "")
        if validate_url(state, target):
            return ValidationResult(COMPLETED, f"Loaded {target} (not an error page).",
                                    evidence=_get(state, "title"))
        return ValidationResult(FAILED, f"Expected current URL to match {target}.")

    if task_type == "media_playback":
        if _step_succeeded(history, "ensure_youtube_playback"):
            return ValidationResult(COMPLETED, "Player verified by ensure_youtube_playback.",
                                    evidence=url)
        return ValidationResult(UNVERIFIED,
                                "Playback was not confirmed by the media validator.")

    if task_type == "note_creation":
        note = (intent.get("content_to_type") or "").lower()
        if _step_succeeded(history, "write_google_keep_note") or (note and note in haystack):
            return ValidationResult(COMPLETED, "Note workflow succeeded or note text is visible.",
                                    evidence=note)
        return ValidationResult(UNVERIFIED, "Could not confirm the note was saved.")

    if task_type == "repo_search":
        entity = (intent.get("entity_or_object") or "").lower()
        query = (intent.get("search_query") or "").lower()
        ok = ("github.com" in haystack
              and ("search" in url.lower() or query in haystack)
              and entity.replace("/", " ") in haystack.replace("/", " "))
        if ok:
            return ValidationResult(COMPLETED, "Repo-scoped GitHub results are visible.",
                                    evidence=entity)
        return ValidationResult(UNVERIFIED, "Repo-scoped GitHub results were not confirmed.")

    if task_type in ("web_search", "site_search"):
        query = (intent.get("search_query") or "").lower()
        target_site = (intent.get("target_site") or "").lower()
        # Completion needs results actually visible (query terms on the page or
        # a recognized search-results URL). Having extracted *something*
        # unrelated is NOT proof - that is the "claimed success without proof"
        # failure mode this validator exists to prevent.
        results = validate_search_results_visible(state, query)
        if target_site:
            results = results and (target_site.replace("www.", "") in haystack
                                   or "google.com/search" in url.lower())
        if results:
            return ValidationResult(COMPLETED, "Search results for the query are visible.",
                                    evidence=query)
        if extracted or _step_succeeded(history, "navigate"):
            return ValidationResult(UNVERIFIED, "Reached a page but could not confirm results.")
        return ValidationResult(FAILED, "No visible search results for the cleaned query.")

    if task_type == "cart_update":
        return _validate_cart(intent, state, history)

    if task_type == "product_configuration":
        if _step_succeeded(history, "configure_apple_product"):
            return ValidationResult(COMPLETED, "Product configuration workflow validated the outcome.")
        return ValidationResult(UNVERIFIED, "Product configuration was not confirmed.")

    if task_type == "information_extraction":
        if extracted or (history and history[-1].get("action") == "extract"):
            return ValidationResult(COMPLETED, "Page content was extracted.",
                                    evidence=url)
        return ValidationResult(UNVERIFIED, "No extracted content to ground the answer.")

    # Generic fallback family.
    if extracted:
        return ValidationResult(COMPLETED, "Observed and extracted the target page.", evidence=url)
    if history:
        return ValidationResult(UNVERIFIED, "Performed actions but did not prove the outcome.")
    return ValidationResult(FAILED, "No observed page action before completion.")


def _validate_cart(intent: Dict, state: Any, history: List[Dict]) -> ValidationResult:
    constraints = intent.get("constraints") or {}
    open_reviews = bool(constraints.get("open_reviews"))
    url = _get(state, "url").lower()
    haystack = _haystack(state)
    cart_steps = [s for s in history
                  if s.get("action") == "add_amazon_item_to_cart" and s.get("success")]
    matching = [s for s in cart_steps
                if (s.get("data") or {}).get("product_match") is True
                and (s.get("data") or {}).get("cart_confirmed") is True]
    if not matching:
        return ValidationResult(UNVERIFIED,
                                "Expected a verified Amazon product match and cart confirmation.")
    if open_reviews:
        reviews_open = (
            any((s.get("data") or {}).get("reviews_opened") is True for s in matching)
            or "/product-reviews/" in url
            or "#customerreviews" in url
            or any(t in haystack for t in ("customer reviews", "top reviews", "global ratings"))
        )
        if reviews_open:
            return ValidationResult(COMPLETED, "Verified product match, cart confirmation, and reviews.")
        return ValidationResult(UNVERIFIED, "Expected the requested Amazon product reviews to be open.")
    return ValidationResult(COMPLETED, "Verified Amazon product match and cart confirmation.")


def _hard_page_blocker(state: Any) -> Optional[Blocker]:
    """Conservatively detect a 404 / captcha / login wall on the *current*
    page. Deliberately strict so a normal page that merely has a "Sign in"
    button in its chrome is not mistaken for an auth wall.
    """
    title = _get(state, "title").lower()
    content = _get(state, "content").lower()
    url = _get(state, "url")
    blob = f"{title} {content}"

    if "page not found" in blob or "404 not found" in blob or " 404 " in f" {blob} ":
        btype, msg = "page_not_found", "The requested page is missing (404)."
    elif any(t in blob for t in ("recaptcha", "unusual traffic", "i'm not a robot",
                                 "are you a robot", "verify you are human")):
        btype, msg = "captcha_or_bot_protection", "A captcha/bot challenge is blocking automation."
    elif (("sign in" in blob or "log in" in blob or "login" in blob)
          and any(t in blob for t in ("password", "continue with google",
                                      "create account", "log in to continue",
                                      "sign in to continue", "join now"))):
        btype, msg = "login_required", "The site is showing a sign-in wall."
    else:
        return None
    return Blocker(
        blocker_type=btype, blocker_message=msg, current_url=url,
        page_title=_get(state, "title"),
        visible_evidence=(_get(state, "title") or _get(state, "content"))[:200])
