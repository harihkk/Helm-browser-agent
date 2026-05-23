"""
Prompt Normalizer - pipeline stage 1.

Takes a raw, messy, possibly angry/typo-ridden user prompt and produces a
cleaned form plus a set of cheap structural signals (URLs, quoted content,
action verbs) that downstream intent extraction relies on.

Two rules govern this module:

1. ``normalize()`` is conservative. It collapses whitespace and fixes a small
   set of high-confidence typos. It must never destroy content the user might
   want typed verbatim (note bodies, search queries). The aggressive helpers
   (``strip_filler``) are only used to build *search queries*, never to build
   ``content_to_type``.
2. Everything here is deterministic and free of model calls so it can be
   tested exhaustively.
"""

import re
from typing import List


# High-confidence typo/shorthand fixes. These are applied during normalize()
# and are intentionally narrow - each one is a real shorthand we have seen,
# not a fuzzy spell-corrector that might mangle a product name or note body.
TYPO_FIXES = {
    r"\bcna\b": "can",
    r"\bu\b": "you",
    r"\bplz\b": "please",
    r"\bpls\b": "please",
    r"\bgot\s+o\b": "go to",
    r"\ban\s+dsearch\b": "and search",
    r"\bdsearch\b": "search",
    r"\bapply website\b": "apple website",
    r"\bsow em\b": "show me",
    r"\bsoem\b": "some",
}

# Filler / politeness / command scaffolding. Stripped ONLY when building a
# search query. Never stripped from content the user asked to type.
FILLER_WORDS = {
    "can", "could", "would", "will", "you", "u", "please", "plz", "pls",
    "kindly", "hey", "hi", "ok", "okay", "just", "really", "actually",
    "now", "asap", "the", "a", "an", "some", "me", "my", "i", "want",
    "wanna", "gonna", "need", "to", "for", "and", "then",
}

# Command verb families. The key is the canonical primary_action; the values
# are surface forms (categories, not site names) we map onto it.
ACTION_VERBS = {
    "navigate": ("go to", "goto", "open", "visit", "navigate", "launch", "bring up"),
    "search": ("search", "look up", "lookup", "look for", "find", "google", "search for"),
    "type": ("type", "write", "put", "enter", "compose", "draft", "note down"),
    "add": ("add", "append", "save"),
    "click": ("click", "press", "tap", "select", "choose", "hit"),
    "play": ("play", "watch", "listen", "stream"),
    "read": ("read", "summarize", "summarise", "tell me about", "explain"),
    "extract": ("extract", "scrape", "pull", "grab", "get me", "collect"),
    "compare": ("compare", "versus", "vs", "difference between"),
    "fill": ("fill", "fill in", "fill out", "complete"),
    "submit": ("submit", "send", "post", "publish", "checkout", "buy", "order", "purchase"),
}

_URL_RE = re.compile(r"https?://[^\s,)\]]+", re.I)
_QUOTED_RE = re.compile(r'["“‘’”\']([^"“‘’”\']{2,})["“‘’”\']')


class PromptNormalizer:
    """Stage-1 normalization and structural signal extraction."""

    def normalize(self, text: str) -> str:
        """Collapse whitespace and apply high-confidence typo fixes.

        This is intentionally identical in behaviour to the historic
        IntentPlanner normalization so routing stays stable.
        """
        out = re.sub(r"\s+", " ", (text or "").strip())
        for pattern, repl in TYPO_FIXES.items():
            out = re.sub(pattern, repl, out, flags=re.I)
        return re.sub(r"\s+", " ", out).strip()

    def detect_urls(self, text: str) -> List[str]:
        """Return any explicit http(s) URLs, with trailing punctuation trimmed."""
        return [m.rstrip(").],!?;:\"'") for m in _URL_RE.findall(text or "")]

    def detect_quoted(self, text: str) -> List[str]:
        """Return the contents of any quoted spans (straight or smart quotes)."""
        return [m.strip() for m in _QUOTED_RE.findall(text or "") if m.strip()]

    def detect_action_verbs(self, text: str) -> List[str]:
        """Return the canonical action families present in the prompt, in a
        stable priority order. The priority favours the *outer* intent: a
        prompt that both navigates and plays is primarily a play task."""
        low = f" {(text or '').lower()} "
        found = []
        for action, surfaces in ACTION_VERBS.items():
            if any(f" {s} " in low or low.strip().startswith(s + " ") for s in surfaces):
                found.append(action)
        # Outer-intent priority: a "search ... and play" prompt is a play task.
        priority = ["submit", "play", "add", "type", "fill", "extract",
                    "compare", "read", "search", "click", "navigate"]
        return sorted(set(found), key=lambda a: priority.index(a) if a in priority else 99)

    def primary_action(self, text: str) -> str:
        """Best single guess at what the user wants done. Defaults to navigate
        when an explicit URL is present, otherwise search."""
        verbs = self.detect_action_verbs(text)
        if verbs:
            return verbs[0]
        if self.detect_urls(text):
            return "navigate"
        return "search"

    def strip_filler(self, text: str) -> str:
        """Remove politeness/command scaffolding to leave the payload.

        Used ONLY to derive search queries. URLs are removed. The result is a
        best-effort cleaned phrase, never used as note/form content.
        """
        out = _URL_RE.sub(" ", text or "")
        # Strip multi-word action surfaces first so "go to" doesn't leave "to".
        for surfaces in ACTION_VERBS.values():
            for s in sorted(surfaces, key=len, reverse=True):
                if " " in s:
                    out = re.sub(rf"\b{re.escape(s)}\b", " ", out, flags=re.I)
        words = [w for w in re.split(r"\s+", out) if w]
        kept = [w for w in words if w.lower().strip(".,:;!?") not in FILLER_WORDS]
        # Drop leftover single-word action verbs (search/find/open/etc.).
        single_verbs = {s for surfaces in ACTION_VERBS.values() for s in surfaces if " " not in s}
        kept = [w for w in kept if w.lower().strip(".,:;!?") not in single_verbs]
        return re.sub(r"\s+", " ", " ".join(kept)).strip(" .,:;!?")
