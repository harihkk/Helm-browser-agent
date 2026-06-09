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

# Product/brand signals that imply a destination domain even when the user
# never names a site. Ordered list, first match wins, so multi-word entries
# precede the shorter ones they overlap. Matched case-insensitively as a plain
# substring against the whole prompt.
PRODUCT_DOMAIN_SIGNALS = [
    ("apple watch", "apple.com"),         # Apple Watch wearables
    ("macbook air", "apple.com"),         # MacBook Air laptops
    ("macbook pro", "apple.com"),         # MacBook Pro laptops
    ("macbook", "apple.com"),             # any other MacBook
    ("mac mini", "apple.com"),            # Mac mini desktop
    ("mac studio", "apple.com"),          # Mac Studio desktop
    ("mac pro", "apple.com"),             # Mac Pro desktop
    ("imac", "apple.com"),                # iMac all-in-one
    ("iphone", "apple.com"),              # iPhone phones
    ("ipad", "apple.com"),                # iPad tablets
    ("airpods", "apple.com"),             # AirPods earbuds
    ("pixel watch", "store.google.com"),  # Google Pixel Watch wearables
    ("pixel buds", "store.google.com"),   # Google Pixel Buds earbuds
    ("pixel tablet", "store.google.com"), # Google Pixel Tablet
    ("pixel", "store.google.com"),        # Google Pixel phones
    ("galaxy watch", "samsung.com"),      # Samsung Galaxy Watch wearables
    ("galaxy buds", "samsung.com"),       # Samsung Galaxy Buds earbuds
    ("galaxy tab", "samsung.com"),        # Samsung Galaxy Tab tablets
    ("galaxy", "samsung.com"),            # Samsung Galaxy phones
]

# Purchase / price intent verbs. These imply "go shopping" even when no brand
# or site is named. Single source of truth: the verbs that also lived in
# ACTION_VERBS["submit"] (buy/order/purchase) are removed from there.
PURCHASE_VERBS = frozenset({
    "buy", "order", "purchase", "get me", "find me", "add to cart",
    "price of", "cost of", "how much is", "how much does", "cheapest",
})

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
    "submit": ("submit", "send", "post", "publish", "checkout"),
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

    def infer_target_domain(self, text: str) -> str:
        """Domain implied by a product/brand signal, even when the user never
        names a site. First match in PRODUCT_DOMAIN_SIGNALS wins; '' if none."""
        low = (text or "").lower()
        for signal, domain in PRODUCT_DOMAIN_SIGNALS:
            if signal in low:
                return domain
        return ""

    def has_purchase_intent(self, text: str) -> bool:
        """True when a purchase/price verb is present as a genuine purchase.

        Returns False when the only purchase verbs sit inside a note/reminder
        clause (e.g. "write a note to buy milk"), detected by a
        note/write/reminder/remember word within the 4 tokens before the verb.
        """
        tokens = [t.strip(".,:;!?") for t in (text or "").lower().split()]
        note_words = {"note", "write", "reminder", "remember"}
        for verb in PURCHASE_VERBS:
            parts = verb.split()
            n = len(parts)
            for i in range(len(tokens) - n + 1):
                if tokens[i:i + n] == parts:
                    window = tokens[max(0, i - 4):i]
                    if not any(w in note_words for w in window):
                        return True
        return False

    def has_media_intent(self, text: str) -> bool:
        """True for play/watch/listen/stream prompts. Reuses detect_action_verbs
        so the media-verb list lives in exactly one place."""
        return "play" in self.detect_action_verbs(text)

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
