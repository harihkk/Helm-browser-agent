"""
URL safety policy (SSRF guard)
==============================
The agent opens URLs an LLM chooses, and that choice can be influenced by
untrusted page content (prompt injection). This module is the single
deterministic gate every navigation passes through before ``page.goto()``.

Safe-by-default policy:
  * Only http / https schemes (blocks file:, data:, javascript:, about:, ftp:).
  * Blocks loopback, private, link-local, reserved, and unspecified IPs.
  * Blocks cloud metadata endpoints (169.254.169.254, metadata.google.internal).
  * Blocks integer/hex-packed IPv4 that decode into those ranges
    (the classic http://2130706433 bypass).
  * When a resolver is supplied, also blocks hostnames that resolve into any of
    those ranges.

Self-hosters automating a local app can opt back in, since this is meant to be
run on your own machine:
    HELM_ALLOW_PRIVATE_HOSTS=true
or by passing ``allow_private=True``. Host suffix allow/deny lists can be
supplied via HELM_URL_ALLOWLIST / HELM_URL_DENYLIST (comma-separated), or as
arguments.

DNS-rebinding note: without a resolver this validates the literal host only. A
host that passes here can still resolve to a private address at connect time.
Pass ``resolver=`` (or the bundled :func:`system_resolver`) to also reject names
that resolve into blocked ranges.
"""

import ipaddress
import os
import socket
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple
from urllib.parse import urlsplit

ALLOWED_SCHEMES = frozenset({"http", "https"})

# Names that always point at the local machine or a metadata service.
_BLOCKED_HOST_SUFFIXES: Tuple[str, ...] = (
    "localhost",
    "ip6-localhost",
    "ip6-loopback",
    "metadata.google.internal",
)

# Well-known cloud metadata addresses (AWS/GCP/Azure share the v4; AWS also v6).
_METADATA_IPS = frozenset({"169.254.169.254", "fd00:ec2::254"})


@dataclass(frozen=True)
class UrlVerdict:
    """Result of a policy check. Truthy iff the URL may be opened."""
    allowed: bool
    reason: str = ""
    category: str = "ok"   # ok | malformed | bad_scheme | denylisted |
                           # not_allowlisted | private_host | metadata

    def __bool__(self) -> bool:
        return self.allowed


# ------------------------------------------------------------------ #
# Config helpers
# ------------------------------------------------------------------ #

def _env_flag(name: str) -> bool:
    return (os.getenv(name, "") or "").strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str) -> Tuple[str, ...]:
    raw = os.getenv(name, "") or ""
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


def _host_matches(host: str, suffixes: Iterable[str]) -> bool:
    """True if ``host`` equals or is a subdomain of any suffix."""
    host = (host or "").lower().rstrip(".")
    for suf in suffixes:
        suf = (suf or "").lower().strip().lstrip(".")
        if not suf:
            continue
        if host == suf or host.endswith("." + suf):
            return True
    return False


def _ip_is_blocked(ip) -> bool:
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


def _coerce_ip(host: str):
    """Return an ip_address for ``host`` if it is an IP literal, else None.

    Also decodes integer-packed and hex-packed IPv4 (e.g. ``2130706433`` and
    ``0x7f000001``) so they cannot be used to smuggle a loopback address past
    the literal-IP check.
    """
    host = (host or "").strip()
    if not host:
        return None
    # Bracketless IPv6 / dotted IPv4 / plain IPv6.
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    # Integer or hex-packed IPv4 with no dots and no colons.
    if ":" not in host and "." not in host:
        try:
            value = int(host, 16) if host.lower().startswith("0x") else int(host)
        except ValueError:
            return None
        if 0 <= value <= 0xFFFFFFFF:
            try:
                return ipaddress.IPv4Address(value)
            except ValueError:
                return None
    return None


# ------------------------------------------------------------------ #
# Resolver (optional, opt-in)
# ------------------------------------------------------------------ #

def system_resolver(host: str) -> List[str]:
    """Resolve ``host`` to a list of IP strings using the system resolver.

    Pass this as ``resolver=`` to :func:`check_url` to also reject hostnames
    that resolve into a blocked range. Network failures yield no addresses,
    which leaves the literal-host decision untouched.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, OSError, UnicodeError):
        return []
    return list({info[4][0] for info in infos})


# ------------------------------------------------------------------ #
# The gate
# ------------------------------------------------------------------ #

def check_url(
    url: str,
    *,
    allow_private: Optional[bool] = None,
    allowlist: Optional[Iterable[str]] = None,
    denylist: Optional[Iterable[str]] = None,
    resolver: Optional[Callable[[str], List[str]]] = None,
) -> UrlVerdict:
    """Decide whether ``url`` is safe for the agent to open.

    ``allow_private`` defaults to the HELM_ALLOW_PRIVATE_HOSTS env flag.
    ``allowlist`` / ``denylist`` default to HELM_URL_ALLOWLIST /
    HELM_URL_DENYLIST (comma-separated host suffixes). A non-empty allowlist
    switches to allowlist-only mode: anything not matching is rejected.
    """
    if allow_private is None:
        allow_private = _env_flag("HELM_ALLOW_PRIVATE_HOSTS")
    allowlist = tuple(allowlist) if allowlist is not None else _env_list("HELM_URL_ALLOWLIST")
    denylist = tuple(denylist) if denylist is not None else _env_list("HELM_URL_DENYLIST")

    if not url or not str(url).strip():
        return UrlVerdict(False, "Empty URL", "malformed")

    try:
        parts = urlsplit(str(url).strip())
    except ValueError as e:
        return UrlVerdict(False, f"Malformed URL: {e}", "malformed")

    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        shown = scheme or "(relative/no scheme)"
        return UrlVerdict(
            False, f"Scheme '{shown}' is not allowed; only http/https.", "bad_scheme")

    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        return UrlVerdict(False, "URL has no host.", "malformed")

    # Explicit deny always wins.
    if denylist and _host_matches(host, denylist):
        return UrlVerdict(False, f"Host '{host}' is denylisted.", "denylisted")

    # Allowlist mode: only listed hosts may pass (private checks still apply).
    if allowlist and not _host_matches(host, allowlist):
        return UrlVerdict(
            False, f"Host '{host}' is not in the allowlist.", "not_allowlisted")

    # Metadata endpoints are never reachable for a good reason, even when
    # private hosts are allowed: they exist only to hand out credentials.
    if host in _METADATA_IPS or _host_matches(host, ("metadata.google.internal",)):
        return UrlVerdict(False, f"Host '{host}' is a cloud metadata endpoint.", "metadata")

    ip = _coerce_ip(host)
    if ip is not None:
        if str(ip) in _METADATA_IPS:
            return UrlVerdict(False, f"Host '{ip}' is a cloud metadata endpoint.", "metadata")
        if _ip_is_blocked(ip) and not allow_private:
            return UrlVerdict(
                False,
                f"Host '{ip}' is a private/loopback/reserved address.",
                "private_host")
        return UrlVerdict(True)

    # Named hosts that always mean "this machine".
    if not allow_private and _host_matches(host, _BLOCKED_HOST_SUFFIXES):
        return UrlVerdict(False, f"Host '{host}' resolves to the local machine.", "private_host")

    # Optional DNS check: reject names that resolve into a blocked range.
    if resolver is not None and not allow_private:
        for addr in resolver(host):
            resolved = _coerce_ip(addr)
            if resolved is None:
                continue
            if str(resolved) in _METADATA_IPS:
                return UrlVerdict(
                    False, f"Host '{host}' resolves to a metadata endpoint.", "metadata")
            if _ip_is_blocked(resolved):
                return UrlVerdict(
                    False,
                    f"Host '{host}' resolves to private address {resolved}.",
                    "private_host")

    return UrlVerdict(True)


def is_safe_url(url: str, **kwargs) -> bool:
    """Convenience boolean wrapper around :func:`check_url`."""
    return bool(check_url(url, **kwargs))
