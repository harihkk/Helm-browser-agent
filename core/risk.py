"""
Risk & Confirmation Layer - pipeline stage that separates safe tasks from
high-impact ones that must be confirmed before the agent acts.

Safe (no confirmation): open a page, search, read, summarize, play media,
type into a draft/note the user explicitly asked for.

Requires confirmation (irreversible / outward-facing): purchases, checkout,
sending messages/emails, submitting forms with personal/payment/legal data,
deleting content, changing account settings, posting publicly, committing
code, financial transactions.
"""

import re
from typing import Dict


# Surface phrases that signal a high-impact intent. Categories, not sites.
_CONFIRM_PATTERNS = [
    r"\bbuy\b", r"\bpurchase\b", r"\bcheckout\b", r"\bcheck\s*out\b",
    r"\bplace (?:an? )?order\b", r"\border\b", r"\bpay\b", r"\bpayment\b",
    r"\bsubmit\b", r"\bsend\b(?!\s+me)", r"\bemail\b", r"\bmessage\b",
    r"\bpost\b", r"\bpublish\b", r"\btweet\b", r"\bcomment\b",
    r"\bdelete\b", r"\bremove\b", r"\bcancel (?:my|the) (?:order|subscription|account)\b",
    r"\bunsubscribe\b", r"\btransfer\b", r"\bwithdraw\b", r"\bdeposit\b",
    r"\bcommit\b", r"\bpush\b", r"\bmerge\b",
    r"\bchange (?:my )?(?:password|email|settings|account)\b",
    r"\bbook\b", r"\breserve\b", r"\bapply for\b",
]
_CONFIRM_RE = re.compile("|".join(_CONFIRM_PATTERNS), re.I)

# Executable workflow actions that mutate external state and therefore need a
# go-ahead even when the verb itself was implicit (e.g. "add ... to cart").
HIGH_IMPACT_ACTIONS = frozenset({
    "add_amazon_item_to_cart",
})

# Task families whose very nature is high-impact.
HIGH_IMPACT_FAMILIES = frozenset({
    "cart_update", "checkout", "purchase", "form_submission",
})

# Task families that are inherently read-only / safe. For these, a high-impact
# *word* inside the prompt is content (e.g. "buy milk" in a note, "delete row"
# as a search term), not an action the agent will perform - so they never
# require confirmation regardless of verb keywords.
SAFE_FAMILIES = frozenset({
    "navigation", "web_search", "site_search", "media_playback",
    "note_creation", "information_extraction",
    "repo_search",
})


def classify_risk(text: str, task_type: str = "") -> Dict:
    """Return {'risk_level', 'requires_confirmation'} for a prompt/task family.

    This is the single source of truth used both when building the intent and
    when the executor decides whether to pause for confirmation.
    """
    if task_type in SAFE_FAMILIES:
        return {"risk_level": "low", "requires_confirmation": False}
    low = (text or "").lower()
    high = bool(_CONFIRM_RE.search(low)) or task_type in HIGH_IMPACT_FAMILIES
    # "add ... to cart" / "add ... to basket" is high-impact even though "add"
    # alone is not.
    if re.search(r"\badd\b.*\b(cart|basket)\b", low):
        high = True
    if high:
        return {"risk_level": "high", "requires_confirmation": True}
    return {"risk_level": "low", "requires_confirmation": False}


def action_requires_confirmation(action: str, params: Dict, intent: Dict) -> bool:
    """Should the executor pause before running ``action`` for this intent?

    True only for genuinely high-impact actions. Navigation, search, extract,
    scroll, typing into a requested note, etc. never trip this.
    """
    intent = intent or {}
    if action in HIGH_IMPACT_ACTIONS:
        return True
    if not intent.get("requires_confirmation"):
        return False
    # The intent is flagged high-impact; gate the action that actually performs
    # the mutation (submit/select on a form, the cart action), not the
    # navigation/search/extract steps that lead up to it.
    if action in ("navigate", "extract", "scroll", "wait", "press_key", "done"):
        return False
    if action in ("click", "type", "select", "submit"):
        return True
    return action not in ("observe_page",)


def confirmation_message(intent: Dict) -> str:
    """Human-readable description of what needs confirming."""
    intent = intent or {}
    family = intent.get("task_type", "")
    goal = intent.get("user_goal") or intent.get("original_prompt") or "this action"
    if family == "cart_update":
        item = intent.get("search_query") or intent.get("entity_or_object") or "the item"
        return (f"This will add \"{item}\" to a real cart, which is a high-impact "
                f"action. Confirm to proceed.")
    return (f"\"{goal}\" is a high-impact action (it changes state, sends, "
            f"submits, or spends). Confirm to proceed.")
