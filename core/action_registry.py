"""Central action registry and browser action guardrails.

The agent should never emit or execute an action that the browser layer does
not understand. This module is intentionally deterministic and free of model
calls so tests can catch unsupported planner output early.
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class ActionSpec:
    name: str
    required_params: Tuple[str, ...] = ()
    description: str = ""
    validates: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    executable: bool = True


class UnsupportedActionError(ValueError):
    pass


class UnsafeTextPayloadError(ValueError):
    pass


ACTION_SPECS: Dict[str, ActionSpec] = {
    # Core executable browser actions.
    "navigate": ActionSpec("navigate", ("url",), "Open a URL", ("validate_url",)),
    "click": ActionSpec("click", ("selector",), "Click an element"),
    "type": ActionSpec("type", ("selector", "text"), "Type safe extracted text"),
    "press_key": ActionSpec("press_key", ("key",), "Press a keyboard key"),
    "select": ActionSpec("select", ("selector", "value"), "Select an option",
                         aliases=("select_option",)),
    "scroll": ActionSpec("scroll", ("direction",), "Scroll the page"),
    "wait": ActionSpec("wait", (), "Wait briefly"),
    "extract": ActionSpec("extract", (), "Extract visible page content",
                          aliases=("observe_page", "extract_text",
                                   "extract_visible_text")),
    "done": ActionSpec("done", ("summary",), "Validated task completion"),

    # Higher-level executable workflows implemented in browser_engine.py.
    "open_top_github_repo": ActionSpec("open_top_github_repo", ("user",), "Open top repository for a GitHub user"),
    "open_first_search_result": ActionSpec("open_first_search_result", (), "Open strongest visible search result"),
    "play_youtube_result": ActionSpec("play_youtube_result", ("query",), "Open a regular YouTube video result"),
    "ensure_youtube_playback": ActionSpec("ensure_youtube_playback", ("query",), "Validate/start YouTube playback", ("validate_media_playing",)),
    "write_google_keep_note": ActionSpec("write_google_keep_note", ("text",), "Create a Google Keep note", ("validate_note_created",)),
    "open_first_github_code_result": ActionSpec("open_first_github_code_result", (), "Open first GitHub code search result"),
    "add_amazon_item_to_cart": ActionSpec("add_amazon_item_to_cart", ("query",), "Add Amazon product result to cart", ("validate_cart_updated",)),

    # Canonical planner concepts. These are registry-visible so planners can
    # be tested against the requested public vocabulary, but they must compile
    # down to executable actions before reaching browser_engine.execute_action.
    "search_web": ActionSpec("search_web", ("query",), "Search the web", ("validate_text_visible",), executable=False),
    "site_search": ActionSpec("site_search", ("target_site", "query"), "Search inside a site", ("validate_text_visible",), executable=False),
    "wait_for_selector": ActionSpec("wait_for_selector", ("selector",), "Wait for a selector", executable=False),
    "extract_text": ActionSpec("extract_text", (), "Extract text", executable=False),
    "select_option": ActionSpec("select_option", ("selector", "value"), "Select an option", executable=False),
    "observe_page": ActionSpec("observe_page", (), "Observe current page", executable=False),
    "validate_url": ActionSpec("validate_url", ("url",), "Validate URL", executable=False),
    "validate_text_visible": ActionSpec("validate_text_visible", ("text",), "Validate text is visible", executable=False),
    "validate_media_playing": ActionSpec("validate_media_playing", (), "Validate media playback", executable=False),
    "validate_note_created": ActionSpec("validate_note_created", ("text",), "Validate note was created", executable=False),
    "validate_cart_updated": ActionSpec("validate_cart_updated", (), "Validate cart state", executable=False),
    "recover_from_error_page": ActionSpec("recover_from_error_page", (), "Recover from page error", executable=False),
    "report_blocker": ActionSpec("report_blocker", ("blocker_type", "blocker_message"), "Report blocker", executable=False),
}


ALIASES = {
    alias: spec.name
    for spec in ACTION_SPECS.values()
    for alias in spec.aliases
}


def _audit_registry() -> None:
    """Fail at import time if the alias table is inconsistent.

    A canonical action name must never appear as an alias of a *different*
    action, and two actions must never claim the same alias. Either mistake
    would silently remap an executable action to a planner concept (the kind
    of bug that makes ``extract`` un-executable).
    """
    seen = {}
    for spec in ACTION_SPECS.values():
        for alias in spec.aliases:
            if alias in seen and seen[alias] != spec.name:
                raise RuntimeError(
                    f"Alias collision: {alias!r} claimed by both "
                    f"{seen[alias]!r} and {spec.name!r}")
            seen[alias] = spec.name
            target = ACTION_SPECS.get(alias)
            if target is not None and target.executable and alias != spec.name:
                raise RuntimeError(
                    f"Alias {alias!r} shadows executable action {alias!r}")


_audit_registry()


SUSPICIOUS_COMMAND_RE = re.compile(
    r"\b("
    r"can\s+you|cna\s+u|please|plz|go\s+to|got\s+o|search\s+for|look\s+for|"
    r"inside\s+https?://|open|actually\s+play|write\s+this\s+in|"
    r"create\s+a\s+note|find\s+me|add\s+to\s+cart|navigate\s+to"
    r")\b",
    re.I,
)


def normalize_action_name(name: str) -> str:
    name = (name or "").strip()
    # An executable canonical action is authoritative - never remap it to a
    # planner concept via an accidental alias collision.
    spec = ACTION_SPECS.get(name)
    if spec is not None and spec.executable:
        return name
    return ALIASES.get(name, name)


def get_action_spec(name: str) -> ActionSpec:
    normalized = normalize_action_name(name)
    try:
        return ACTION_SPECS[normalized]
    except KeyError:
        raise UnsupportedActionError(f"Unsupported browser action: {name}") from None


def is_supported_action(name: str) -> bool:
    return normalize_action_name(name) in ACTION_SPECS


def is_executable_action(name: str) -> bool:
    return get_action_spec(name).executable


def validate_action(name: str, params: Dict = None, executable_only: bool = False) -> None:
    params = params or {}
    spec = get_action_spec(name)
    if executable_only and not spec.executable:
        raise UnsupportedActionError(
            f"Action '{name}' is a planner concept and cannot be executed directly"
        )
    missing = [key for key in spec.required_params if not params.get(key)]
    if missing:
        raise UnsupportedActionError(
            f"Action '{name}' missing required parameter(s): {', '.join(missing)}"
        )


def registry_names(executable_only: bool = False) -> List[str]:
    return sorted(
        name for name, spec in ACTION_SPECS.items()
        if not executable_only or spec.executable
    )


def validate_text_payload(text: str, *, allow_full_command: bool = False) -> None:
    """Reject text that looks like an instruction rather than extracted text.

    This guard applies to search fields, note bodies, and form fields. A caller
    may opt out only when the product explicitly asks to type the exact sentence.
    """
    candidate = re.sub(r"\s+", " ", (text or "")).strip()
    if not candidate:
        raise UnsafeTextPayloadError("Refusing to type empty text")
    if allow_full_command:
        return
    if SUSPICIOUS_COMMAND_RE.search(candidate):
        raise UnsafeTextPayloadError(
            "Refusing to type text that still looks like a full browser command"
        )


def validate_plan_actions(actions: Iterable[Dict]) -> None:
    for action in actions:
        validate_action(action.get("action", ""), action.get("parameters", {}))
