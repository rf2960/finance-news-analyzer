"""
Stock Universe Screener for FinSight RAG.

Provides two complementary methods to build a stock watchlist:

1. top_stocks_by_market_activity()
   Pulls price/volume data for the NASDAQ-100 + S&P 500 top names via yfinance
   and ranks by trading volume or price.

2. discover_stocks_from_news()
   Scans live RSS feeds, extracts ticker mentions, and returns the most
   discussed stocks – a "news-driven" universe that captures event-driven
   opportunities regardless of market cap.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Static ticker universe: NASDAQ-100 + notable S&P 500 names
# ---------------------------------------------------------------------------

# Core NASDAQ-100 tickers (as of Q1 2026, representative list)
NASDAQ100_TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "COST", "NFLX", "ASML", "AZN", "QCOM", "AMD", "ADBE", "CSCO", "PEP",
    "INTU", "HON", "CMCSA", "AMAT", "AMGN", "BKNG", "ISRG", "VRTX", "MELI",
    "PANW", "CDNS", "SNPS", "LRCX", "KLAC", "REGN", "GILD", "ADI", "MRVL",
    "CTAS", "FTNT", "CRWD", "DXCM", "TTD", "VRSK", "ROP", "PCAR", "NXPI",
    "MNST", "CEG", "IDXX", "EA", "FAST", "ROST", "EXC", "CPRT", "GEHC",
    "ILMN", "WBD", "ON", "ZS", "DDOG", "TEAM", "MU", "ARM", "DELL",
    "MAR", "SBUX", "MDLZ", "ODFL", "KDP", "CTSH", "BIIB", "FANG", "SIRI",
    "ADP", "TTWO", "PYPL", "CHTR", "PAYX", "ORLY", "CSGP",
]

# Additional S&P 500 high-volume / high-cap names
SP500_EXTRA_TICKERS: list[str] = [
    "BRK-B", "LLY", "JPM", "V", "UNH", "XOM", "WMT", "MA", "JNJ", "PG",
    "HD", "CVX", "MRK", "ABBV", "KO", "BAC", "CRM", "PFE", "ACN", "MCD",
    "DIS", "GE", "VZ", "INTC", "WFC", "GS", "MS", "T", "RTX", "CAT",
    "UBER", "COIN", "PLTR", "RBLX", "HOOD", "RIVN", "LCID", "NIO", "BABA",
    "TSM", "SOFI", "AMC", "GME", "SMCI", "IONQ", "QBTS", "RGTI",
]

# All unique tickers
ALL_TICKERS: list[str] = list(dict.fromkeys(NASDAQ100_TICKERS + SP500_EXTRA_TICKERS))


# ---------------------------------------------------------------------------
# Stock data dataclass
# ---------------------------------------------------------------------------

@dataclass
class StockSnapshot:
    ticker: str
    name: str = ""
    sector: str = ""
    price: float = 0.0
    day_change_pct: float = 0.0
    volume: int = 0
    avg_volume: int = 0
    volume_ratio: float = 0.0   # today_volume / avg_volume
    market_cap: float = 0.0     # in billions
    pe_ratio: Optional[float] = None
    news_mentions: int = 0       # from news scan
    mention_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker":          self.ticker,
            "name":            self.name or self.ticker,
            "sector":          self.sector or "—",
            "price":           round(self.price, 2),
            "day_change_pct":  round(self.day_change_pct * 100, 2),
            "volume":          self.volume,
            "avg_volume":      self.avg_volume,
            "volume_ratio":    round(self.volume_ratio, 2),
            "market_cap_b":    round(self.market_cap / 1e9, 1) if self.market_cap else 0.0,
            "pe_ratio":        round(self.pe_ratio, 1) if self.pe_ratio else None,
            "news_mentions":   self.news_mentions,
        }


# ---------------------------------------------------------------------------
# Top stocks by volume / price (yfinance)
# ---------------------------------------------------------------------------

def top_stocks_by_market_activity(
    n: int = 100,
    sort_by: str = "volume",   # "volume", "volume_ratio", "price", "market_cap"
    tickers: list[str] | None = None,
) -> list[StockSnapshot]:
    """
    Fetch snapshot data for a universe of tickers and rank by the chosen metric.

    Args:
        n:       Maximum number of stocks to return.
        sort_by: Ranking criterion: "volume", "volume_ratio", "price", "market_cap".
        tickers: Custom ticker list; defaults to ALL_TICKERS.

    Returns:
        List of StockSnapshot objects, best-ranked first.
    """
    universe = tickers or ALL_TICKERS
    results: list[StockSnapshot] = []

    try:
        import yfinance as yf  # type: ignore

        # Batch download – one network call for all tickers
        raw = yf.download(
            tickers=" ".join(universe),
            period="5d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        for tkr in universe:
            try:
                # Navigate multi-level columns
                if len(universe) == 1:
                    prices = raw
                else:
                    prices = raw[tkr] if tkr in raw.columns.get_level_values(0) else None

                if prices is None or prices.empty or len(prices) < 2:
                    continue

                last_close  = float(prices["Close"].iloc[-1])
                prev_close  = float(prices["Close"].iloc[-2])
                day_chg     = (last_close - prev_close) / prev_close if prev_close else 0.0
                last_vol    = int(prices["Volume"].iloc[-1]) if "Volume" in prices else 0
                avg_vol_5d  = int(prices["Volume"].mean()) if "Volume" in prices else 0
                vol_ratio   = last_vol / avg_vol_5d if avg_vol_5d else 0.0

                # Light info fetch (may fail – non-blocking)
                name = tkr
                sector = ""
                mkt_cap = 0.0
                pe = None
                try:
                    info = yf.Ticker(tkr).fast_info  # faster than .info
                    mkt_cap = float(getattr(info, "market_cap", 0) or 0)
                    name    = getattr(info, "company_name", None) or tkr
                except Exception:
                    pass

                results.append(StockSnapshot(
                    ticker       = tkr,
                    name         = name,
                    sector       = sector,
                    price        = last_close,
                    day_change_pct = day_chg,
                    volume       = last_vol,
                    avg_volume   = avg_vol_5d,
                    volume_ratio = vol_ratio,
                    market_cap   = mkt_cap,
                    pe_ratio     = pe,
                ))
            except Exception:
                continue

    except Exception:
        return []

    # Sort
    key_map = {
        "volume":       lambda s: s.volume,
        "volume_ratio": lambda s: s.volume_ratio,
        "price":        lambda s: s.price,
        "market_cap":   lambda s: s.market_cap,
    }
    key_fn = key_map.get(sort_by, key_map["volume"])
    results.sort(key=key_fn, reverse=True)
    return results[:n]


# ---------------------------------------------------------------------------
# News-driven stock discovery
# ---------------------------------------------------------------------------

# Three extraction patterns in priority order:
# 1. Parenthesized tickers: Apple (AAPL) — most reliable in financial news
# 2. Dollar-sign tickers: $NVDA or $TSLA — common in social/financial media
# 3. Plain uppercase 2-5 char words — fallback, only accepted if in known list
_TICKER_PARENS = re.compile(r'\(([A-Z]{2,5})\)')
_TICKER_DOLLAR = re.compile(r'\$([A-Z]{2,5})\b')
_TICKER_PLAIN  = re.compile(r'\b([A-Z]{2,5})\b')

# Words that look like tickers but aren't – filter list
_STOP_WORDS: set[str] = {
    "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF", "IN", "IS",
    "IT", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE",
    "AND", "ARE", "BUT", "CAN", "CEO", "COO", "CTO", "DAY", "DID", "FOR",
    "GET", "GOT", "HAD", "HAS", "HER", "HIS", "HOW", "ITS", "LET", "MAY",
    "NEW", "NOT", "NOW", "OLD", "ONE", "OUR", "OUT", "OWN", "PUT", "SAY",
    "SEE", "SET", "THE", "TOO", "TWO", "USE", "WAS", "WHO", "WHY", "YET",
    "YOU", "CNBC", "SAID", "TOLD", "WILL", "FROM", "BEEN", "HAVE", "SAYS",
    "WITH", "THIS", "THAT", "WERE", "THEY", "ALSO", "MORE", "THAN", "WHAT",
    "WHEN", "THEN", "INTO", "SOME", "EACH", "SUCH", "LIKE", "OVER", "MOST",
    "YEAR", "WEEK", "DAYS", "MADE", "MAKE", "JUST", "BOTH", "AFTER", "FIRST",
    "ABOUT", "WHICH", "THEIR", "COULD", "OTHER", "STOCK", "SHARE", "TRADE",
    "PRICE", "YIELD", "INDEX", "RALLY", "BULLS", "BEARS", "RATES", "BONDS",
    "GAINS", "DROPS", "FALLS", "SURGE", "CLOSE", "OPENS", "BEATS", "MISSES",
    "REPORT", "MARKET", "STOCKS", "SHARES", "GROWTH", "RECORD", "PROFIT",
    "REVENUE", "BILLION", "MILLION", "PERCENT", "QUARTER", "ANALYST",
    "COMPANY", "SECTOR", "DOLLAR", "ENERGY", "HEALTH", "GLOBAL", "WEEKLY",
    "OUTLOOK", "RESULTS", "CHINA", "INDIA", "JAPAN", "EUROPE", "RUSSIA",
    "TRUMP", "BIDEN", "FOMC", "NYSE", "REIT", "WALL", "HIGH", "WEEK",
    "TECH", "FUND", "CORP", "CORP", "BLOG", "NEWS", "DATA", "CALL",
    "BEAT", "MISS", "ROSE", "FELL", "RISE", "FALL", "GAIN", "LOSS",
    "GDP", "CPI", "PCE", "ESG", "IPO", "ETF", "SEC", "FED", "CEO",
}

# Tickers we know are real
_KNOWN_TICKERS: set[str] = set(ALL_TICKERS)


def _extract_tickers(text: str) -> list[str]:
    """
    Extract potential ticker symbols from text using three patterns.
    Parenthesized and dollar-sign formats are accepted unconditionally.
    Plain uppercase words are accepted only if they are in _KNOWN_TICKERS.
    """
    found: dict[str, int] = {}  # ticker → priority (lower = more confident)

    # Pattern 1: (AAPL) — highest confidence
    for m in _TICKER_PARENS.findall(text):
        tkr = m.upper()
        if tkr not in _STOP_WORDS:
            found[tkr] = min(found.get(tkr, 1), 1)

    # Pattern 2: $AAPL — high confidence
    for m in _TICKER_DOLLAR.findall(text):
        tkr = m.upper()
        if tkr not in _STOP_WORDS:
            found[tkr] = min(found.get(tkr, 2), 2)

    # Pattern 3: plain AAPL — only if in known universe
    for m in _TICKER_PLAIN.findall(text):
        tkr = m.upper()
        if tkr in _KNOWN_TICKERS and tkr not in _STOP_WORDS:
            found[tkr] = min(found.get(tkr, 3), 3)

    return list(found.keys())


def discover_stocks_from_news(
    n: int = 25,
    rss_urls: list[tuple[str, str]] | None = None,
) -> list[StockSnapshot]:
    """
    Scan recent financial RSS feeds, extract ticker mentions, and return the
    top-N most discussed stocks. Uses 7-day Google News queries so results
    are not limited to today's articles.

    Args:
        n:        Maximum tickers to return.
        rss_urls: List of (source_name, url) tuples; defaults to key feeds.

    Returns:
        List of StockSnapshot objects sorted by news_mentions desc.
    """
    if rss_urls is None:
        rss_urls = [
            # Yahoo Finance — headlines commonly use (TICKER) format
            ("Yahoo Finance",      "https://finance.yahoo.com/rss/topstories"),
            ("Yahoo Finance News", "https://finance.yahoo.com/news/rssindex"),
            # Benzinga — uses (TICKER) extensively
            ("Benzinga",           "https://www.benzinga.com/feed"),
            # MarketWatch
            ("MarketWatch",        "https://feeds.marketwatch.com/marketwatch/topstories/"),
            # CNBC
            ("CNBC",               "https://www.cnbc.com/id/100003114/device/rss/rss.html"),
            # Reuters business + tech
            ("Reuters",            "https://feeds.reuters.com/reuters/businessNews"),
            ("Reuters Tech",       "https://feeds.reuters.com/reuters/technologyNews"),
            # Google News — recent 7-day windows on key financial topics
            ("Google/Earnings",    "https://news.google.com/rss/search?q=stock+earnings+results+beats&hl=en-US&gl=US&ceid=US:en&as_qdr=w"),
            ("Google/Tech",        "https://news.google.com/rss/search?q=nasdaq+tech+stock+NVDA+AAPL+MSFT&hl=en-US&gl=US&ceid=US:en&as_qdr=w"),
            ("Google/Markets",     "https://news.google.com/rss/search?q=S%26P+500+stock+market+rally+weekly&hl=en-US&gl=US&ceid=US:en&as_qdr=w"),
            ("Google/Finance",     "https://news.google.com/rss/search?q=stock+upgrade+downgrade+analyst+target&hl=en-US&gl=US&ceid=US:en&as_qdr=w"),
        ]

    mention_counts: dict[str, int] = {}
    mention_sources: dict[str, list[str]] = {}

    for source_name, url in rss_urls:
        try:
            articles = _fetch_rss_text(url)
            for text in articles:
                for tkr in _extract_tickers(text):
                    mention_counts[tkr] = mention_counts.get(tkr, 0) + 1
                    mention_sources.setdefault(tkr, [])
                    if source_name not in mention_sources[tkr]:
                        mention_sources[tkr].append(source_name)
        except Exception:
            continue

    # Also scan yfinance news headlines for top known active tickers
    # (fallback — ensures results even when RSS feeds return little content)
    try:
        import yfinance as yf
        _sample = ["NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA", "AMD",
                   "GOOGL", "INTC", "MU", "COIN", "PLTR", "SMCI", "SOFI"]
        for tkr in _sample:
            try:
                news = yf.Ticker(tkr).news or []
                if news:
                    cnt = min(len(news), 3)  # weight by article count, cap at 3
                    mention_counts[tkr] = mention_counts.get(tkr, 0) + cnt
                    mention_sources.setdefault(tkr, [])
                    if "Yahoo Finance" not in mention_sources[tkr]:
                        mention_sources[tkr].append("Yahoo Finance")
            except Exception:
                continue
    except Exception:
        pass

    if not mention_counts:
        return []

    # Sort by mention count
    sorted_tickers = sorted(
        mention_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Fetch basic price data for top results
    top_tkrs = [t for t, _ in sorted_tickers[:n * 2] if t not in _STOP_WORDS]
    price_data = _batch_get_prices(top_tkrs[:50])

    results: list[StockSnapshot] = []
    for tkr, cnt in sorted_tickers:
        if tkr in _STOP_WORDS:
            continue
        snap = price_data.get(tkr, StockSnapshot(ticker=tkr))
        snap.news_mentions  = cnt
        snap.mention_sources = mention_sources.get(tkr, [])
        results.append(snap)
        if len(results) >= n:
            break

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_rss_text(url: str, timeout: int = 8) -> list[str]:
    """Fetch RSS feed and return a list of title+summary strings."""
    import urllib.request
    import urllib.error
    import xml.etree.ElementTree as ET
    import html

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FinSightRAG/1.0 (stock screener)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)
        texts = []
        for item in items:
            def _t(tag: str) -> str:
                el = item.find(tag) or item.find(f"atom:{tag}", ns)
                return html.unescape(el.text or "") if el is not None else ""
            texts.append(f"{_t('title')} {_t('description')} {_t('summary')}")
        return texts
    except Exception:
        return []


def _batch_get_prices(tickers: list[str]) -> dict[str, StockSnapshot]:
    """Fetch basic price/volume for a list of tickers. Returns dict keyed by ticker."""
    if not tickers:
        return {}
    result: dict[str, StockSnapshot] = {}
    try:
        import yfinance as yf  # type: ignore
        raw = yf.download(
            tickers=" ".join(tickers),
            period="2d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
        for tkr in tickers:
            try:
                if len(tickers) == 1:
                    prices = raw
                else:
                    prices = raw[tkr] if tkr in raw.columns.get_level_values(0) else None
                if prices is None or prices.empty or len(prices) < 2:
                    continue
                last  = float(prices["Close"].iloc[-1])
                prev  = float(prices["Close"].iloc[-2])
                vol   = int(prices["Volume"].iloc[-1]) if "Volume" in prices else 0
                chg   = (last - prev) / prev if prev else 0.0
                result[tkr] = StockSnapshot(
                    ticker=tkr,
                    price=last,
                    day_change_pct=chg,
                    volume=vol,
                )
            except Exception:
                continue
    except Exception:
        pass
    return result
