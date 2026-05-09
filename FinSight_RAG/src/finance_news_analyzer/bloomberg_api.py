"""
Bloomberg B-PIPE API integration for FinSight RAG.

Bloomberg's B-PIPE Python library (blpapi) connects to a running Bloomberg
Terminal or Bloomberg enterprise B-PIPE server.

Installation:
    pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi

Connection requirements:
  1. Bloomberg Terminal running on your machine (connects to localhost:8194), OR
  2. Enterprise B-PIPE server (use custom host/port and app_name for auth)
  3. The 'blpapi' Python package installed

If any requirement is missing, all functions return empty lists and the system
falls back to public news sources (Google News, Yahoo Finance, Reuters, etc.).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Bloomberg configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BloombergConfig:
    """Settings for Bloomberg B-PIPE connection."""
    enabled: bool = False
    host: str = "localhost"
    port: int = 8194
    # Optional: enterprise B-PIPE application name for AuthType=APPNAME_AND_KEY
    app_name: str = ""
    # Status (populated after connection test)
    last_status: str = "Not tested"
    last_ok: bool = False


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def check_blpapi_installed() -> Tuple[bool, str]:
    """Return (is_installed, message)."""
    try:
        import blpapi  # noqa: F401
        return True, "blpapi package is installed."
    except ImportError:
        return False, (
            "blpapi not installed. "
            "Install via: pip install --index-url="
            "https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi"
        )


def check_bloomberg_connection(config: BloombergConfig) -> Tuple[bool, str]:
    """
    Try a quick connection to the Bloomberg B-PIPE server.

    Returns (success, status_message).
    Does NOT raise — always returns gracefully.
    """
    if not config.enabled:
        return False, "Bloomberg disabled in settings."

    installed, msg = check_blpapi_installed()
    if not installed:
        return False, msg

    try:
        import blpapi  # type: ignore

        opts = blpapi.SessionOptions()
        opts.setServerHost(config.host)
        opts.setServerPort(config.port)
        if config.app_name:
            opts.setAuthenticationOptions(
                f"AuthenticationMode=APPLICATION_ONLY;"
                f"ApplicationAuthType=APPNAME_AND_KEY;"
                f"ApplicationName={config.app_name}"
            )

        session = blpapi.Session(opts)
        if session.start():
            session.stop()
            return True, f"✅ Connected to Bloomberg at {config.host}:{config.port}"
        else:
            return False, f"❌ Cannot connect to Bloomberg at {config.host}:{config.port}"
    except Exception as exc:
        return False, f"❌ Bloomberg connection error: {exc}"


# ---------------------------------------------------------------------------
# News fetching via //blp/news service (Bloomberg Terminal / B-PIPE)
# ---------------------------------------------------------------------------

def fetch_bloomberg_news_blpapi(
    ticker: str,
    config: BloombergConfig,
    max_articles: int = 20,
) -> list[dict]:
    """
    Fetch Bloomberg news headlines for a ticker via the B-PIPE //blp/news service.

    Args:
        ticker:       Stock ticker, e.g. "NVDA" (converted to "NVDA US Equity" internally).
        config:       BloombergConfig with connection settings.
        max_articles: Maximum headlines to retrieve.

    Returns:
        List of dicts with keys: title, url, published_at, text, source.
        Returns [] if Bloomberg is unavailable or an error occurs.
    """
    if not config.enabled:
        return []

    installed, _ = check_blpapi_installed()
    if not installed:
        return []

    try:
        import blpapi  # type: ignore

        opts = blpapi.SessionOptions()
        opts.setServerHost(config.host)
        opts.setServerPort(config.port)
        if config.app_name:
            opts.setAuthenticationOptions(
                f"AuthenticationMode=APPLICATION_ONLY;"
                f"ApplicationAuthType=APPNAME_AND_KEY;"
                f"ApplicationName={config.app_name}"
            )

        session = blpapi.Session(opts)
        if not session.start():
            return []

        articles: list[dict] = []

        # ── Method 1: //blp/news HeadlineRequest ──────────────────────────
        try:
            if session.openService("//blp/news"):
                news_svc = session.getService("//blp/news")
                req = news_svc.createRequest("HeadlineRequest")
                # Bloomberg ticker format: "NVDA US Equity"
                bb_ticker = f"{ticker.upper()} US Equity"
                req.set("ticker", bb_ticker)
                req.set("maxRows", max_articles)
                session.sendRequest(req)

                done = False
                while not done:
                    ev = session.nextEvent(500)
                    for msg in ev:
                        _parse_headline_msg(msg, articles)
                    if ev.eventType() in (
                        blpapi.Event.RESPONSE,
                        blpapi.Event.REQUEST_STATUS,
                        blpapi.Event.SESSION_STATUS,
                    ):
                        done = True
        except Exception:
            pass  # Fall through to Method 2

        # ── Method 2: //blp/refdata NEWS_HEADLINE field ───────────────────
        if not articles:
            try:
                if session.openService("//blp/refdata"):
                    ref_svc = session.getService("//blp/refdata")
                    req = ref_svc.createRequest("ReferenceDataRequest")
                    bb_ticker = f"{ticker.upper()} US Equity"
                    req.getElement("securities").appendValue(bb_ticker)
                    req.getElement("fields").appendValue("NEWS_HEADLINE")
                    req.getElement("fields").appendValue("NEWS_STORY_DATE")
                    session.sendRequest(req)

                    done = False
                    while not done:
                        ev = session.nextEvent(500)
                        for msg in ev:
                            _parse_refdata_msg(msg, articles, ticker)
                        if ev.eventType() in (
                            blpapi.Event.RESPONSE,
                            blpapi.Event.REQUEST_STATUS,
                        ):
                            done = True
            except Exception:
                pass

        session.stop()
        return articles

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Message parsers (handle blpapi element structures)
# ---------------------------------------------------------------------------

def _parse_headline_msg(msg, articles: list[dict]) -> None:
    """Parse a HeadlineData message from //blp/news."""
    try:
        msg_type = str(msg.messageType())
        if "Headline" not in msg_type and "headline" not in msg_type:
            return

        # Try various element names Bloomberg might use
        for container_name in ("headlines", "headlineData", "data"):
            if not msg.hasElement(container_name):
                continue
            container = msg.getElement(container_name)
            count = container.numValues() if hasattr(container, "numValues") else 0
            for i in range(min(count, 20)):
                try:
                    h = container.getValueAsElement(i)
                    title = _safe_get(h, "headline", "headlineText", "title")
                    url   = _safe_get(h, "url", "storyUrl")
                    pub   = _safe_get(h, "publishedAt", "storyDate", "date")
                    snip  = _safe_get(h, "snippet", "summary", "bodyText")
                    if title:
                        articles.append({
                            "title":        title[:300],
                            "url":          url or "",
                            "published_at": pub or datetime.now(timezone.utc).isoformat(),
                            "text":         f"{title}. {snip}".strip()[:500],
                            "source":       "Bloomberg",
                        })
                except Exception:
                    continue
    except Exception:
        pass


def _parse_refdata_msg(msg, articles: list[dict], ticker: str) -> None:
    """Parse a ReferenceDataResponse from //blp/refdata for NEWS_HEADLINE."""
    try:
        if not msg.hasElement("securityData"):
            return
        sec_data = msg.getElement("securityData")
        for i in range(sec_data.numValues()):
            sec = sec_data.getValueAsElement(i)
            if not sec.hasElement("fieldData"):
                continue
            field_data = sec.getElement("fieldData")
            headline = _safe_get(field_data, "NEWS_HEADLINE")
            if headline:
                articles.append({
                    "title":        headline[:300],
                    "url":          f"https://bba.bloomberg.net/securities/{ticker}",
                    "published_at": datetime.now(timezone.utc).isoformat(),
                    "text":         headline,
                    "source":       "Bloomberg",
                })
    except Exception:
        pass


def _safe_get(element, *field_names: str, default: str = "") -> str:
    """Try multiple field name variations and return the first non-empty string."""
    for name in field_names:
        try:
            if element.hasElement(name):
                val = element.getElementAsString(name)
                if val:
                    return str(val)
        except Exception:
            continue
    return default
