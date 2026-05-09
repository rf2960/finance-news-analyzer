"""
macro_events.py
---------------
Fetches high-impact macro and geopolitical news headlines (wars, summits,
sanctions, trade deals, Fed/ECB decisions, elections, natural disasters, etc.)
from public RSS feeds and returns them as a structured context block
ready for injection into the agent pipeline.

Sources scanned
---------------
• Reuters (World / Business / Politics)
• BBC News (World)
• AP News (Top Stories / World)
• Al Jazeera (World)
• CNBC (Economy / Politics)
• MarketWatch (Economy)

Keyword filters
---------------
The module uses a two-tier keyword system:
  TIER-1  – direct market movers: war, summit, sanctions, rate decision, tariff
  TIER-2  – broader macro: GDP, inflation, election, trade deal, central bank
Headlines matching TIER-1 get higher weight; TIER-2 are included if there is
room left up to `max_events` (default 10).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Macro RSS feeds
# ---------------------------------------------------------------------------
_MACRO_FEEDS: list[dict] = [
    {"name": "Reuters World",      "url": "https://feeds.reuters.com/reuters/worldNews"},
    {"name": "Reuters Business",   "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"name": "Reuters Politics",   "url": "https://feeds.reuters.com/reuters/politicsNews"},
    {"name": "BBC World",          "url": "http://feeds.bbci.co.uk/news/world/rss.xml"},
    {"name": "AP Top Stories",     "url": "https://rsshub.app/apnews/topics/ap-top-news"},
    {"name": "CNBC Economy",       "url": "https://www.cnbc.com/id/20910258/device/rss/rss.html"},
    {"name": "CNBC Politics",      "url": "https://www.cnbc.com/id/10000113/device/rss/rss.html"},
    {"name": "MarketWatch Economy","url": "https://feeds.marketwatch.com/marketwatch/economy-politics/"},
    {"name": "Al Jazeera World",   "url": "https://www.aljazeera.com/xml/rss/all.xml"},
]

# ---------------------------------------------------------------------------
# High-impact keyword sets
# ---------------------------------------------------------------------------
_TIER1_KEYWORDS: set[str] = {
    # Conflict / geopolitics
    "war", "conflict", "invasion", "ceasefire", "missile", "airstrike",
    "nuclear", "nato", "sanctions", "blockade", "coup", "assassination",
    # Summits / diplomacy
    "summit", "bilateral", "meeting", "negotiations", "deal", "treaty",
    "agreement", "diplomacy", "talks", "tariff", "trade war", "embargo",
    # Central bank
    "fed", "federal reserve", "rate hike", "rate cut", "fomc",
    "ecb", "boe", "boj", "pboc", "interest rate", "quantitative easing",
    "quantitative tightening",
    # Major political
    "president", "election", "impeach", "resign", "executive order",
    "stimulus", "bailout", "default", "debt ceiling", "shutdown",
}

_TIER2_KEYWORDS: set[str] = {
    "gdp", "inflation", "recession", "unemployment", "jobs report",
    "cpi", "pce", "manufacturing", "pmi", "retail sales",
    "trade deficit", "budget", "fiscal", "currency", "devaluation",
    "oil", "energy", "commodity", "opec", "supply chain",
    "bank", "financial crisis", "credit", "liquidity",
}


def _keyword_score(text: str) -> tuple[int, int]:
    """Return (tier1_hits, tier2_hits) for a headline + summary."""
    t = text.lower()
    t1 = sum(1 for kw in _TIER1_KEYWORDS if kw in t)
    t2 = sum(1 for kw in _TIER2_KEYWORDS if kw in t)
    return t1, t2


def _clean(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def fetch_macro_events(
    ticker: str = "",
    company: str = "",
    sector: str = "",
    max_events: int = 10,
    timeout: int = 6,
) -> dict[str, Any]:
    """
    Fetch and filter high-impact macro/geopolitical headlines.

    Parameters
    ----------
    ticker    : Stock ticker symbol — used for static keyword mapping.
    company   : Full company name (from yfinance) — used for dynamic keyword derivation.
    sector    : Sector (from yfinance) — used for sector-specific keyword boosts.
    max_events: Maximum number of events to include (default 10).
    timeout   : HTTP timeout per feed in seconds.

    Returns
    -------
    dict with keys:
        "events"   – list of event dicts {headline, source, summary, tier, published}
        "summary"  – formatted string for LLM prompt injection
        "error"    – None on success, message on failure
    """
    try:
        import feedparser
    except ImportError:
        return {
            "events": [],
            "summary": "Macro events unavailable (feedparser not installed).",
            "error": "feedparser not installed",
        }

    tk = ticker.upper() if ticker else ""
    # Combine static mapping + dynamic terms from company/sector
    ticker_terms = _build_search_terms(tk, company, sector)

    tier1_events: list[dict] = []
    tier2_events: list[dict] = []

    for feed_info in _MACRO_FEEDS:
        try:
            feed = feedparser.parse(feed_info["url"], request_headers={"User-Agent": "FinSightRAG/1.0"})
            entries = feed.get("entries", []) or []
            for entry in entries[:30]:   # cap per feed
                title   = _clean(entry.get("title", ""))
                summary = _clean(entry.get("summary", "") or entry.get("description", ""))
                combined = f"{title} {summary}"

                t1, t2 = _keyword_score(combined)
                if t1 == 0 and t2 == 0:
                    continue

                # Count how many ticker-specific terms match (higher boost = more relevance)
                matching_terms = sum(1 for term in ticker_terms if term in combined.lower())
                if matching_terms > 0:
                    t1 += matching_terms * 3   # 3 points per matching specific term

                pub = entry.get("published", "")
                evt = {
                    "headline":  title,
                    "source":    feed_info["name"],
                    "summary":   summary[:300],
                    "tier":      1 if t1 > 0 else 2,
                    "score":     t1 * 2 + t2,
                    "published": pub,
                }
                if t1 > 0:
                    tier1_events.append(evt)
                else:
                    tier2_events.append(evt)
        except Exception:
            continue

    # Sort by score descending (ticker-specific events naturally bubble to top)
    tier1_events.sort(key=lambda x: -x["score"])
    tier2_events.sort(key=lambda x: -x["score"])

    seen_words: set[str] = set()
    deduplicated: list[dict] = []
    for evt in tier1_events + tier2_events:
        key_words = set(re.findall(r"\b\w{5,}\b", evt["headline"].lower()))
        if key_words & seen_words and len(key_words & seen_words) > 2:
            continue  # too similar to already-included event
        seen_words |= key_words
        deduplicated.append(evt)
        if len(deduplicated) >= max_events:
            break

    # Count how many events are ticker-specific vs. generic macro
    ticker_relevant_count = 0
    if ticker_terms:
        for evt in deduplicated:
            combined = f"{evt['headline']} {evt['summary']}".lower()
            if any(term in combined for term in ticker_terms):
                ticker_relevant_count += 1

    summary = _format_macro_summary(ticker, deduplicated)
    return {
        "events": deduplicated,
        "summary": summary,
        "error": None,
        "ticker_relevant_count": ticker_relevant_count,
    }


# ── Sector → macro keyword mapping ────────────────────────────────────────────
_SECTOR_TERMS: dict[str, list[str]] = {
    "Technology":            ["chip", "semiconductor", "ai", "tech regulation", "antitrust", "export controls"],
    "Energy":                ["oil", "crude", "opec", "energy", "natural gas", "pipeline", "refinery"],
    "Financials":            ["fed", "interest rate", "bank", "credit", "liquidity", "monetary policy"],
    "Financial Services":    ["fed", "interest rate", "bank", "credit", "fintech", "crypto", "regulation"],
    "Consumer Cyclical":     ["retail", "consumer", "tariff", "trade", "spending", "recession"],
    "Consumer Defensive":    ["inflation", "food", "supply chain", "tariff", "consumer price"],
    "Healthcare":            ["drug", "fda", "clinical trial", "regulation", "pharma", "biotech"],
    "Industrials":           ["manufacturing", "supply chain", "trade", "infrastructure", "defense", "contract"],
    "Materials":             ["commodity", "gold", "copper", "mining", "inflation", "supply"],
    "Real Estate":           ["interest rate", "fed", "housing", "mortgage", "reit"],
    "Utilities":             ["energy", "power grid", "regulation", "rate", "climate"],
    "Communication Services":["regulation", "antitrust", "media", "telecom", "broadband"],
    "Unknown":               [],
}

# Static ticker-to-terms mapping for common ETFs and ADRs
_TICKER_TERMS: dict[str, list[str]] = {
    "FXI":  ["china", "chinese", "xi jinping", "beijing", "prc", "sino", "shanghai", "hong kong", "renminbi", "yuan"],
    "KWEB": ["china", "chinese tech", "beijing", "regulation", "alibaba", "tencent", "baidu"],
    "MCHI": ["china", "chinese", "beijing", "shanghai", "hong kong"],
    "EEM":  ["emerging markets", "developing", "brics", "india", "brazil", "indonesia"],
    "GLD":  ["gold", "precious metals", "safe haven", "inflation", "fed"],
    "IAU":  ["gold", "precious metals", "central bank"],
    "GDX":  ["gold", "mining", "precious metals"],
    "SLV":  ["silver", "precious metals"],
    "USO":  ["oil", "crude", "opec", "energy", "wti", "brent"],
    "XLE":  ["oil", "energy", "crude", "opec", "exxon", "chevron"],
    "TLT":  ["treasury", "fed", "interest rate", "bonds", "yield", "fomc"],
    "IEF":  ["treasury", "fed", "interest rate", "bonds"],
    "SPY":  ["us economy", "federal reserve", "recession", "stimulus", "s&p"],
    "QQQ":  ["tech", "technology", "ai", "nasdaq", "antitrust"],
    "EWJ":  ["japan", "boj", "yen", "nikkei", "tokyo", "japanese"],
    "EWZ":  ["brazil", "latin america", "lula", "real"],
    "EWG":  ["germany", "europe", "ecb", "euro", "dax"],
    "EWU":  ["uk", "britain", "boe", "sterling", "london"],
    "BABA": ["china", "alibaba", "beijing", "antitrust", "xi jinping"],
    "JD":   ["china", "jd.com", "beijing", "chinese consumer"],
    "NVDA": ["nvidia", "ai chip", "semiconductor", "export controls", "data center"],
    "AMD":  ["amd", "chip", "semiconductor", "data center", "export"],
    "TSLA": ["china", "tesla", "ev", "electric vehicle", "tariff", "elon"],
    "INTC": ["intel", "chip", "semiconductor", "manufacturing"],
    "AAPL": ["apple", "china", "tariff", "iphone", "supply chain"],
    "MSFT": ["microsoft", "ai", "cloud", "antitrust", "regulation"],
    "GOOGL":["google", "alphabet", "antitrust", "ai", "regulation"],
    "META": ["meta", "facebook", "regulation", "antitrust", "social media"],
    "AMZN": ["amazon", "aws", "antitrust", "tariff", "e-commerce"],
    "NFLX": ["netflix", "streaming", "regulation", "media"],
}


def _build_search_terms(ticker: str, company: str, sector: str) -> list[str]:
    """
    Build a prioritised list of macro search terms from ticker, company name,
    and sector. Terms are deduplicated and lowercase.
    """
    terms: list[str] = []

    # 1. Static ticker map (highest relevance)
    if ticker in _TICKER_TERMS:
        terms.extend(_TICKER_TERMS[ticker])

    # 2. Dynamic terms from company name
    company_lower = company.lower()
    # Extract meaningful words from company name (skip generic words)
    _STOP = {"inc", "corp", "ltd", "plc", "group", "company", "co", "holdings",
              "the", "and", "of", "for", "in", "fund", "etf", "trust", "select",
              "spdr", "ishares", "vanguard", "blackrock", "invesco", "ultra", "pro"}
    for word in re.split(r"[\s\-,.()/]+", company_lower):
        if len(word) >= 4 and word not in _STOP:
            terms.append(word)

    # 3. Detect country from company name
    _COUNTRY_MAP = {
        "china": ["china", "chinese", "beijing", "shanghai", "renminbi", "yuan", "xi jinping"],
        "japan": ["japan", "japanese", "boj", "yen", "nikkei", "tokyo"],
        "germany": ["germany", "german", "ecb", "euro", "dax"],
        "brazil": ["brazil", "brazilian", "latin america"],
        "india": ["india", "indian", "modi", "sensex"],
        "europe": ["europe", "european", "ecb", "euro"],
        "hong kong": ["hong kong", "hkex", "hkd"],
        "korea": ["korea", "korean", "kospi", "won"],
    }
    for country, kws in _COUNTRY_MAP.items():
        if country in company_lower:
            terms.extend(kws)

    # 4. Sector terms
    sector_terms = _SECTOR_TERMS.get(sector, _SECTOR_TERMS.get("Unknown", []))
    terms.extend(sector_terms)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        t = t.lower().strip()
        if t and t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _format_macro_summary(ticker: str, events: list[dict]) -> str:
    if not events:
        return "No high-impact macro/geopolitical events found in current news feeds."

    ticker_note = f" (relevant to {ticker})" if ticker else ""
    lines = [
        f"=== HIGH-IMPACT MACRO & GEOPOLITICAL EVENTS{ticker_note} ===",
        f"Sourced from {len(set(e['source'] for e in events))} news outlets. "
        f"These events may materially affect market direction and should be "
        f"incorporated into the investment thesis.",
        "",
    ]
    for i, evt in enumerate(events, 1):
        tier_label = "⚠️ TIER-1 (direct market mover)" if evt["tier"] == 1 else "ℹ️  TIER-2 (macro context)"
        lines.append(f"[{i}] {tier_label}")
        lines.append(f"    Source: {evt['source']}")
        lines.append(f"    Headline: {evt['headline']}")
        if evt["summary"] and evt["summary"] != evt["headline"]:
            lines.append(f"    Context: {evt['summary'][:200]}")
        lines.append("")
    lines.append("=" * 55)
    return "\n".join(lines)
