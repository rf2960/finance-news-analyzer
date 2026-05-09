"""
News Ingestion Layer for FinSight RAG.

Fetches financial articles from:
  1. Yahoo Finance  – ticker-specific news via yfinance
  2. Bloomberg RSS  – high-authority macro / market news
  3. Reuters RSS    – business & technology feeds
  4. CNBC RSS       – top business stories
  5. MarketWatch RSS– top financial stories
  6. Benzinga RSS   – finance-focused news aggregator

Each article is normalised to a NewsArticle dataclass and tagged with
a source credibility weight used downstream by the RAG and agent layers.
"""
from __future__ import annotations

import hashlib
import html
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

# ---------------------------------------------------------------------------
# Source credibility registry (mirrors the proposal's "High-Authority" tiers)
# ---------------------------------------------------------------------------
SOURCE_WEIGHTS: dict[str, float] = {
    "Bloomberg": 0.95,
    "Wall Street Journal": 0.90,
    "Financial Times": 0.88,
    "Reuters": 0.85,
    "Barron's": 0.82,
    "CNBC": 0.78,
    "Yahoo Finance": 0.72,
    "MarketWatch": 0.70,
    "Seeking Alpha": 0.65,
    "Benzinga": 0.60,
    "The Motley Fool": 0.55,
    "Unknown": 0.50,
}

# RSS endpoints — all publicly accessible
RSS_FEEDS: dict[str, str] = {
    # Bloomberg moved their public RSS; use Google News aggregated Bloomberg stories
    "Bloomberg": "https://news.google.com/rss/search?q=bloomberg+markets+finance&hl=en-US&gl=US&ceid=US:en",
    "Reuters": "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Tech": "https://feeds.reuters.com/reuters/technologyNews",
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Benzinga": "https://www.benzinga.com/feed",
    "Yahoo Finance": "https://finance.yahoo.com/rss/topstories",
    "Yahoo Finance News": "https://finance.yahoo.com/news/rssindex",
    "Seeking Alpha": "https://seekingalpha.com/feed.xml",
}

_HTTP_TIMEOUT = 10  # seconds


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class NewsArticle:
    citation_id: str          # Deterministic hash of url+title
    source: str               # Publisher name
    title: str
    url: str
    published_at: str         # ISO-8601 UTC string
    text: str                 # Title + summary concatenated
    ticker: str = ""          # Populated when article is ticker-specific
    credibility_weight: float = 0.65

    def is_relevant(self, ticker: str, company: str = "") -> bool:
        """
        Return True if title or text mentions the ticker or company name.

        Matches:
          - exact ticker symbol (e.g. "NVDA")
          - full company name (e.g. "NVIDIA Corporation")
          - first word of company name with 4+ chars (e.g. "nvidia", "microsoft")
          - common abbreviations (e.g. "nvidia's", "microsoft's")
        """
        combined = (self.title + " " + self.text).lower()

        # Ticker symbol match (case-insensitive)
        if ticker.lower() in combined:
            return True

        if not company:
            return False

        # Full company name
        needle_co = company.lower()
        if len(needle_co) > 3 and needle_co in combined:
            return True

        # First meaningful word of the company name (handles "Microsoft Corporation" → "microsoft")
        words = [w for w in re.split(r"[\s\-]+", company) if len(w) >= 4]
        for word in words[:2]:
            if word.lower() in combined:
                return True

        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_html(raw: str) -> str:
    """Strip HTML tags and unescape entities."""
    raw = html.unescape(raw or "")
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"\s+", " ", raw)
    return raw.strip()


def _make_citation_id(source: str, url: str, title: str) -> str:
    """Create a short deterministic ID from content fingerprint."""
    fingerprint = f"{source}|{url}|{title}"
    return hashlib.md5(fingerprint.encode()).hexdigest()[:10].upper()


def _normalise_date(raw: str) -> str:
    """
    Try several date formats and return an ISO-8601 UTC string.
    Falls back to now() if nothing matches.
    """
    if not raw:
        return datetime.now(timezone.utc).isoformat()
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat()
        except ValueError:
            continue
    return datetime.now(timezone.utc).isoformat()


def _weight_for(source_name: str) -> float:
    """Return the credibility weight for a publisher, with fuzzy matching."""
    for key, weight in SOURCE_WEIGHTS.items():
        if key.lower() in source_name.lower():
            return weight
    return SOURCE_WEIGHTS["Unknown"]


def _is_title_repetition(title: str, summary: str) -> bool:
    """
    Check if summary is just the title repeated (common in poorly-formed RSS).
    Returns True if summary is 90%+ similar to title.
    """
    if not title or not summary:
        return False
    
    title_clean = title.lower().strip()
    summary_clean = summary.lower().strip()
    
    # Exact match
    if title_clean == summary_clean:
        return True
    
    # Summary starts with title
    if summary_clean.startswith(title_clean):
        return True
    
    # Calculate word overlap
    title_words = set(re.findall(r'\w+', title_clean))
    summary_words = set(re.findall(r'\w+', summary_clean))
    
    if not title_words:
        return False
    
    overlap = len(title_words & summary_words) / len(title_words)
    return overlap > 0.9


def _fetch_article_content(url: str, source_name: str, max_length: int = 800) -> str:
    """
    Fetch and extract article content from URL when RSS doesn't provide it.
    Uses simple HTML parsing to extract main content.
    Returns empty string if fetching fails.
    """
    if not url or not url.startswith('http'):
        return ""
    
    try:
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        
        with urllib.request.urlopen(req, timeout=8) as response:
            html_content = response.read().decode('utf-8', errors='ignore')
        
        # Remove script and style tags
        html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL)
        html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL)
        
        # Extract content from common article containers
        content_patterns = [
            r'<article[^>]*>(.*?)</article>',
            r'<div[^>]*class=["\'][^"\']*article-body[^"\']*["\'][^>]*>(.*?)</div>',
            r'<div[^>]*class=["\'][^"\']*story-body[^"\']*["\'][^>]*>(.*?)</div>',
            r'<div[^>]*class=["\'][^"\']*content[^"\']*["\'][^>]*>(.*?)</div>',
            r'<p[^>]*>(.*?)</p>',  # Fallback to paragraphs
        ]
        
        extracted = ""
        for pattern in content_patterns:
            matches = re.findall(pattern, html_content, flags=re.DOTALL | re.IGNORECASE)
            if matches:
                # Take first few matches and join
                extracted = ' '.join(matches[:5])
                break
        
        if not extracted:
            return ""
        
        # Clean HTML tags
        extracted = _clean_html(extracted)
        
        # Extract sentences (simple approach)
        sentences = re.split(r'[.!?]+\s+', extracted)
        
        # Filter out very short sentences and navigation/UI text
        meaningful_sentences = []
        for sent in sentences:
            sent = sent.strip()
            # Skip short sentences, common UI text, and cookie notices
            if len(sent) < 30:
                continue
            if any(skip in sent.lower() for skip in ['cookie', 'subscribe', 'sign up', 'follow us', 'share this', 'advertisement']):
                continue
            meaningful_sentences.append(sent)
        
        # Take first 3-5 meaningful sentences
        content = '. '.join(meaningful_sentences[:5])
        
        # Limit length
        if len(content) > max_length:
            content = content[:max_length].rsplit('.', 1)[0] + '...'
        
        return content if len(content) > 50 else ""
        
    except Exception:
        # Silent fail - we tried our best
        return ""


# ---------------------------------------------------------------------------
# Yahoo Finance (ticker-specific)
# ---------------------------------------------------------------------------

def fetch_yahoo_news(ticker: str) -> list[NewsArticle]:
    """
    Fetch ticker-specific news items via the yfinance library.
    Returns an empty list gracefully if yfinance is unavailable.
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        return []

    try:
        yt = yf.Ticker(ticker)
        raw_news = yt.news or []
    except Exception:
        return []

    articles: list[NewsArticle] = []
    for item in raw_news:
        # yfinance wraps news in a 'content' dict in newer versions
        content = item.get("content") or {}
        if isinstance(content, dict) and content:
            title = _clean_html(content.get("title") or item.get("title", ""))
            summary = _clean_html(content.get("description") or content.get("body") or "")
            url = (
                content.get("canonicalUrl", {}).get("url")
                or content.get("clickThroughUrl", {}).get("url")
                or item.get("link", "")
            )
            pub_raw = content.get("pubDate") or ""
            provider_info = content.get("provider") or {}
            source = provider_info.get("displayName") or "Yahoo Finance"
        else:
            title = _clean_html(item.get("title", ""))
            summary = _clean_html(item.get("summary") or "")
            url = item.get("link") or ""
            ts = item.get("providerPublishTime")
            pub_raw = (
                datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                if ts
                else ""
            )
            source = item.get("publisher") or "Yahoo Finance"

        if not title:
            continue

        text = f"{title}. {summary}".strip()
        articles.append(
            NewsArticle(
                citation_id=_make_citation_id(source, url, title),
                source=source,
                title=title,
                url=url,
                published_at=_normalise_date(pub_raw),
                text=text,
                ticker=ticker.upper(),
                credibility_weight=_weight_for(source),
            )
        )
    return articles


# ---------------------------------------------------------------------------
# RSS feeds (general financial news)
# ---------------------------------------------------------------------------

def _fetch_rss_with_feedparser(source_name: str, url: str) -> list[NewsArticle]:
    """Use feedparser library if available."""
    import feedparser  # type: ignore  # noqa: F401
    feed = feedparser.parse(url)
    articles: list[NewsArticle] = []
    for entry in feed.get("entries", []):
        title = _clean_html(entry.get("title") or "")
        
        # Try multiple content fields in order of richness
        summary = ""
        # 1. Try content:encoded (full article HTML)
        if "content" in entry and entry.content:
            content_parts = entry.content if isinstance(entry.content, list) else [entry.content]
            summary = _clean_html(content_parts[0].get("value", ""))
        
        # 2. Try summary_detail
        if not summary and "summary_detail" in entry:
            summary = _clean_html(entry.summary_detail.get("value", ""))
        
        # 3. Try regular summary/description
        if not summary:
            summary = _clean_html(entry.get("summary") or entry.get("description") or "")
        
        # 4. If summary is just title repeated, try to extract from content
        if summary and _is_title_repetition(title, summary):
            # Try to get richer content if available
            if hasattr(entry, 'content') and entry.content:
                summary = _clean_html(entry.content[0].get("value", ""))
        
        link = entry.get("link") or ""
        pub_raw = entry.get("published") or entry.get("updated") or ""
        if not title:
            continue
        
        # Build meaningful text - only use what we have
        if summary and len(summary) > len(title) + 10:
            text = f"{title}. {summary}".strip()
        else:
            # Just use the title - don't make excuses
            text = title
            
        articles.append(
            NewsArticle(
                citation_id=_make_citation_id(source_name, link, title),
                source=source_name,
                title=title,
                url=link,
                published_at=_normalise_date(pub_raw),
                text=text,
                credibility_weight=_weight_for(source_name),
            )
        )
    return articles


def _fetch_rss_with_urllib(source_name: str, url: str) -> list[NewsArticle]:
    """Fallback RSS parser using only stdlib."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FinSightRAG/1.0 (financial research bot)"},
        )
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            raw_xml = resp.read()
    except (urllib.error.URLError, OSError):
        return []

    try:
        root = ET.fromstring(raw_xml)
    except ET.ParseError:
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    # Handle both RSS <item> and Atom <entry> shapes
    items = root.findall(".//item") or root.findall(".//atom:entry", ns)
    articles: list[NewsArticle] = []
    for item in items:
        def _txt(tag: str, default: str = "") -> str:
            el = item.find(tag) or item.find(f"atom:{tag}", ns)
            return _clean_html(el.text or default) if el is not None else default

        title = _txt("title")
        
        # Try content:encoded first (richer content)
        content_encoded = ""
        for content_tag in ["content:encoded", "content"]:
            content_el = item.find(content_tag)
            if content_el is not None and content_el.text:
                content_encoded = _clean_html(content_el.text)
                break
        
        summary = content_encoded or _txt("description") or _txt("summary")
        
        # Strip title repetition from summary
        if summary and _is_title_repetition(title, summary):
            summary = ""
        
        link_el = item.find("link") or item.find("atom:link", ns)
        link = (
            (link_el.text or link_el.get("href") or "")
            if link_el is not None
            else ""
        )
        pub_raw = _txt("pubDate") or _txt("published") or _txt("updated")
        if not title:
            continue
        
        # Only use summary if it adds meaningful content
        if summary and len(summary) > len(title) + 10:
            text = f"{title}. {summary}".strip()
        else:
            text = title
        articles.append(
            NewsArticle(
                citation_id=_make_citation_id(source_name, link, title),
                source=source_name,
                title=title,
                url=link,
                published_at=_normalise_date(pub_raw),
                text=text,
                credibility_weight=_weight_for(source_name),
            )
        )
    return articles


def fetch_rss_feed(source_name: str, url: str) -> list[NewsArticle]:
    """Fetch an RSS/Atom feed, preferring feedparser if installed."""
    try:
        return _fetch_rss_with_feedparser(source_name, url)
    except ImportError:
        return _fetch_rss_with_urllib(source_name, url)
    except Exception:
        return _fetch_rss_with_urllib(source_name, url)


# ---------------------------------------------------------------------------
# Google News ticker-specific RSS (public, no auth required)
# ---------------------------------------------------------------------------

def fetch_google_news_ticker(ticker: str, company: str = "") -> list[NewsArticle]:
    """
    Fetch Google News RSS for a specific ticker symbol.
    Returns recent articles mentioning the ticker from any source.
    """
    query = f"{ticker}+stock" if not company else f"{ticker}+{company.split()[0]}+stock"
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    articles = fetch_rss_feed("Google News", url)
    # Tag articles with the ticker for downstream relevance
    for art in articles:
        if not art.ticker:
            art.ticker = ticker.upper()
        # Inherit credibility from actual publisher if detectable in title
        title_lower = art.title.lower()
        for src, w in SOURCE_WEIGHTS.items():
            if src.lower() in title_lower:
                art.credibility_weight = max(art.credibility_weight, w * 0.9)
    return articles


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest_news(
    ticker: str,
    company: str = "",
    include_rss: bool = True,
    rss_sources: list[str] | None = None,
    relevance_filter: bool = True,
    bloomberg_config=None,  # BloombergConfig | None
) -> list[NewsArticle]:
    """
    Aggregate news articles from Bloomberg B-PIPE, Yahoo Finance, and RSS feeds.

    Args:
        ticker:            Stock ticker, e.g. ``"NVDA"``.
        company:           Human-readable company name for relevance matching.
        include_rss:       Whether to pull from RSS feeds in addition to Yahoo.
        rss_sources:       Which RSS sources to query; defaults to all.
        relevance_filter:  If True, only keep articles mentioning the ticker.
        bloomberg_config:  :class:`BloombergConfig` instance, or None to skip.

    Returns:
        Deduplicated list of :class:`NewsArticle` objects, Bloomberg first.
    """
    seen_ids: set[str] = set()
    all_articles: list[NewsArticle] = []

    # 0. Bloomberg B-PIPE (highest authority – fetched first if configured)
    if bloomberg_config is not None:
        try:
            from src.finance_news_analyzer.bloomberg_api import (  # type: ignore
                fetch_bloomberg_news_blpapi,
            )
            bb_articles = fetch_bloomberg_news_blpapi(ticker, bloomberg_config)
            for art_dict in bb_articles:
                art = NewsArticle(
                    citation_id=_make_citation_id(
                        "Bloomberg",
                        art_dict.get("url", ""),
                        art_dict.get("title", ""),
                    ),
                    source="Bloomberg",
                    title=_clean_html(art_dict.get("title", "")),
                    url=art_dict.get("url", ""),
                    published_at=_normalise_date(art_dict.get("published_at", "")),
                    text=_clean_html(art_dict.get("text", "")),
                    ticker=ticker.upper(),
                    credibility_weight=SOURCE_WEIGHTS["Bloomberg"],
                )
                if art.citation_id not in seen_ids and art.title:
                    seen_ids.add(art.citation_id)
                    all_articles.append(art)
        except Exception:
            pass  # Bloomberg unavailable – continue to other sources

    # 1. Yahoo Finance (ticker-specific via yfinance API)
    for art in fetch_yahoo_news(ticker):
        if art.citation_id not in seen_ids:
            seen_ids.add(art.citation_id)
            all_articles.append(art)

    # 2. Google News ticker-specific RSS (always — gives Bloomberg, Reuters, CNBC hits)
    if include_rss:
        try:
            for art in fetch_google_news_ticker(ticker, company):
                if art.citation_id not in seen_ids:
                    seen_ids.add(art.citation_id)
                    all_articles.append(art)
        except Exception:
            pass

    # 3. Broad RSS feeds (Bloomberg, Reuters, CNBC, MarketWatch…)
    if include_rss:
        sources_to_fetch = rss_sources or list(RSS_FEEDS.keys())
        # Prioritise Bloomberg first
        ordered = sorted(
            sources_to_fetch,
            key=lambda s: -SOURCE_WEIGHTS.get(s, SOURCE_WEIGHTS["Unknown"]),
        )
        for src in ordered:
            feed_url = RSS_FEEDS.get(src)
            if not feed_url:
                continue
            try:
                for art in fetch_rss_feed(src, feed_url):
                    if art.citation_id not in seen_ids:
                        seen_ids.add(art.citation_id)
                        all_articles.append(art)
            except Exception:
                continue  # Never crash if a single feed fails

    # 3. Relevance filter
    if relevance_filter and ticker:
        all_articles = [
            a for a in all_articles if a.is_relevant(ticker, company)
        ]

    # 4. Sort: Bloomberg first, then by date descending
    all_articles.sort(
        key=lambda a: (-a.credibility_weight, a.published_at),
        reverse=False,
    )
    # Reverse the date part after the credibility sort
    all_articles.sort(key=lambda a: a.credibility_weight, reverse=True)

    return all_articles
