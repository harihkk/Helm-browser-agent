"""
Blocker System - pipeline terminal state for failed / blocked / unverified runs.

Every run that does not complete must surface a precise, structured blocker
instead of vague text. This module owns the blocker vocabulary, the schema,
and a deterministic classifier that maps page evidence to a blocker type.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Canonical blocker vocabulary. Planners, validators, and the executor must
# only emit these types so the frontend and tests can rely on them.
BLOCKER_TYPES = (
    "bad_url",
    "page_not_found",
    "navigation_failed",
    "wrong_domain",
    "login_required",
    "captcha_or_bot_protection",
    "permission_denied",
    "popup_blocking",
    "missing_element",
    "disabled_element",
    "no_results",
    "unavailable_option",
    "unsupported_action",
    "timeout",
    "validation_failed",
    "ambiguous_instruction",
    "confirmation_required",
    "unsafe_action",
    "stale_run_event",
    "partial_completion",
)

# Statuses a non-completed run can end in.
BLOCKER_STATUSES = ("blocked", "failed", "unverified")

# Default next-step guidance per blocker type. Kept human and actionable.
_SUGGESTIONS = {
    "bad_url": "Check the URL is well-formed and reachable, then retry.",
    "page_not_found": "Try the site's home page or a corrected URL.",
    "navigation_failed": "Retry navigation or check connectivity.",
    "wrong_domain": "Re-issue the task with the correct destination site.",
    "login_required": "Sign in (or hand off the browser), then retry.",
    "captcha_or_bot_protection": "Solve the challenge manually or use the site's own search, then retry.",
    "permission_denied": "Grant the required permission or use an authenticated session.",
    "popup_blocking": "Dismiss the popup/cookie banner and retry.",
    "missing_element": "The expected control was not found; rephrase what to click or read.",
    "disabled_element": "A prerequisite is unmet; satisfy it before retrying.",
    "no_results": "Broaden or correct the query and retry.",
    "unavailable_option": "Pick an available option; the requested one is not offered.",
    "unsupported_action": "Rephrase the task using a supported browser action.",
    "timeout": "The page was slow; retry, possibly with fewer steps.",
    "validation_failed": "The outcome could not be proven; refine the task or success criteria.",
    "ambiguous_instruction": "Clarify the instruction (target, query, or content).",
    "confirmation_required": "Confirm the high-impact action to proceed.",
    "unsafe_action": "This action is blocked for safety; confirm explicitly to override.",
    "stale_run_event": "Ignore - this event belonged to a previous run.",
    "partial_completion": "Some steps succeeded; review and continue or refine the task.",
}

# Which blocker types are "blocked" (recoverable / awaiting something) vs a
# hard "failed". Anything not listed defaults to failed.
_BLOCKED_TYPES = frozenset({
    "login_required", "captcha_or_bot_protection", "permission_denied",
    "popup_blocking", "confirmation_required", "unsafe_action",
    "disabled_element", "unavailable_option", "no_results", "page_not_found",
    "wrong_domain", "ambiguous_instruction",
})


@dataclass
class Blocker:
    """Structured terminal state. ``to_dict`` is the WebSocket/DB contract."""
    blocker_type: str
    blocker_message: str
    status: str = ""
    current_url: str = ""
    page_title: str = ""
    failed_step: int = 0
    last_successful_step: int = 0
    attempted_recoveries: List[str] = field(default_factory=list)
    visible_evidence: str = ""
    suggested_next_step: str = ""

    def __post_init__(self):
        if self.blocker_type not in BLOCKER_TYPES:
            # Never silently accept an unknown blocker type - fail loudly so a
            # planner/executor bug is caught in tests rather than shipped.
            raise ValueError(f"Unknown blocker_type: {self.blocker_type!r}")
        if not self.status:
            self.status = "blocked" if self.blocker_type in _BLOCKED_TYPES else "failed"
        if self.status not in BLOCKER_STATUSES:
            raise ValueError(f"Unknown blocker status: {self.status!r}")
        if not self.suggested_next_step:
            self.suggested_next_step = _SUGGESTIONS.get(self.blocker_type, "Review and retry.")

    def to_dict(self) -> Dict:
        return {
            "status": self.status,
            "blocker_type": self.blocker_type,
            "blocker_message": self.blocker_message,
            "current_url": self.current_url,
            "page_title": self.page_title,
            "failed_step": self.failed_step,
            "last_successful_step": self.last_successful_step,
            "attempted_recoveries": list(self.attempted_recoveries),
            "visible_evidence": self.visible_evidence,
            "suggested_next_step": self.suggested_next_step,
        }


def classify_blocker(
    url: str = "",
    title: str = "",
    content: str = "",
    last_error: str = "",
    *,
    is_error: bool = False,
    failed_step: int = 0,
    last_successful_step: int = 0,
    attempted_recoveries: Optional[List[str]] = None,
) -> Blocker:
    """Map page evidence + last error onto a structured Blocker.

    The classifier is ordered most-specific first. ``unsupported_action`` and
    bot/login/404 detection win over generic failure.
    """
    url = url or ""
    title_low = (title or "").lower()
    content_low = (content or "").lower()
    error_low = (last_error or "").lower()
    text = f"{title_low} {content_low} {error_low}"
    recoveries = list(attempted_recoveries or [])

    def make(btype: str, message: str) -> Blocker:
        evidence = (title or content or last_error or "")[:200]
        return Blocker(
            blocker_type=btype,
            blocker_message=message,
            current_url=url,
            page_title=title or "",
            failed_step=failed_step,
            last_successful_step=last_successful_step,
            attempted_recoveries=recoveries,
            visible_evidence=evidence,
        )

    if "cannot be executed directly" in error_low or "unsupported browser action" in error_low or "unsupported" in text:
        return make("unsupported_action",
                    "The planner requested an action the browser engine does not support.")
    if "refusing to type" in error_low:
        return make("unsafe_action",
                    "Refused to type text that still looked like a full command rather than content.")
    if any(t in text for t in ("captcha", "recaptcha", "i'm not a robot", "unusual traffic", "are you a robot")):
        return make("captcha_or_bot_protection",
                    "A captcha or bot challenge is blocking automation.")
    if any(t in text for t in ("sign in", "log in", "login required", "authentication required", "please sign in")):
        return make("login_required",
                    "The site requires authentication before the agent can continue.")
    if "404" in text or "page not found" in text or "page isn't available" in text:
        return make("page_not_found",
                    "The requested page appears to be missing or unavailable.")
    if "no results" in text or "did not match any" in text or "no matches" in text:
        return make("no_results", "The site returned no results for the requested query.")
    if "timeout" in text or "timed out" in text:
        return make("timeout", "The browser timed out waiting for the page or element.")
    if "out of stock" in text or "currently unavailable" in text or "unavailable" in text:
        return make("unavailable_option", "The requested option or item is unavailable.")
    if "no typeable element" in error_low or "no note editor" in error_low or re.search(r"could not find .* (button|element|link|result)", error_low):
        return make("missing_element", last_error or "An expected element could not be found.")
    if "disabled" in error_low:
        return make("disabled_element", "A required control is disabled; a prerequisite is unmet.")
    if is_error:
        return make("bad_url", last_error or "The current page is in an error state.")
    return make("validation_failed", last_error or "The task outcome could not be validated.")


def confirmation_blocker(message: str, url: str = "", title: str = "",
                         last_successful_step: int = 0) -> Blocker:
    """Build the structured blocker used when a high-impact action needs the
    user's go-ahead before the agent will perform it."""
    return Blocker(
        blocker_type="confirmation_required",
        blocker_message=message,
        current_url=url,
        page_title=title,
        last_successful_step=last_successful_step,
    )
