"""
Input sanitisation helpers shared by the decision engine and detectors.

Anything that may be attacker-controlled (URL hostnames, links found inside an
SMS, phone numbers) is reduced to a strict character whitelist and a hard
length cap before it can reach the Llama prompt or the terminal.
"""

import re
from urllib.parse import urlsplit

_HOST_OK = re.compile(r"[^a-z0-9.\-]")
_SIGNAL_OK = re.compile(r"[^\w\s.,:;()%/+\-@'&]")


def safe_host(url: str, max_len: int = 60) -> str:
    """Return only the lower-case hostname of a URL ('' if it cannot be parsed)."""
    if not url:
        return ""
    text = url.strip()
    if "://" not in text:
        text = "http://" + text
    try:
        host = urlsplit(text).hostname or ""
    except ValueError:
        host = text.split("://", 1)[-1].split("/", 1)[0]
    host = _HOST_OK.sub("", host.lower())
    return host[:max_len]


def safe_signal(signal: str, max_len: int = 120) -> str:
    """Strip control characters / markup from a detector signal and cap its length."""
    s = "".join(ch for ch in str(signal) if ch.isprintable())
    s = _SIGNAL_OK.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def mask_phone(number: str) -> str:
    """Show only the last 4 digits of a phone number (privacy + no prompt injection)."""
    digits = re.sub(r"\D", "", number)
    return ("XXXXXX" + digits[-4:]) if len(digits) >= 4 else "XXXX"
