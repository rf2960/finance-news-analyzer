"""
Agent Runner – Bridge between FinSight RAG and the Person-2 Agent System.

Architecture
------------
Both FinSight_RAG and person2_agent_system use a top-level ``src`` package.
Python's module cache (sys.modules) only allows one binding for the name
``src`` at a time, so we must swap the two namespaces when calling into the
agent system.  A context-manager ``_agent_sys_ctx()`` handles this safely.

Two operating modes
-------------------
heuristic (default, no API key)
    All three agents are implemented locally using keyword/sentiment
    heuristics – no LLM call is made.  The pipeline runs entirely offline
    and completes in < 2 seconds.

langchain_openai (API key provided)
    The person2 agent system's LangGraph workflow is invoked via the
    sys.modules swap.  GPT-4o-mini (or any supplied model) drives all three
    agents using the RAG chunks as their evidence base, producing richer,
    cited reasoning.  This is the UNIFIED path: RAG → Agents → OpenAI answer.
"""
from __future__ import annotations

import contextlib
import os
import random
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path management
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_FINSIGHT_ROOT   = _THIS_FILE.parents[2]          # …/FinSight_RAG
_WORKSPACE_ROOT  = _FINSIGHT_ROOT.parent           # …/FinS
_AGENT_SYS_ROOT  = (
    _WORKSPACE_ROOT
    / "person2_agent_system_handoff"
    / "person2_agent_system"
)
_AGENT_SRC_ROOT  = _AGENT_SYS_ROOT / "src"        # …/person2_agent_system/src

_AGENT_AVAILABLE = _AGENT_SRC_ROOT.is_dir()


# ---------------------------------------------------------------------------
# sys.modules namespace-swap context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _agent_sys_ctx():
    """
    Temporarily make the agent system's ``src`` package the active one in
    sys.modules so the agent system's internal ``from src.xxx import …``
    statements resolve to the right modules.
    """
    finsight_src = {k: v for k, v in sys.modules.items()
                    if k == "src" or k.startswith("src.")}
    for k in finsight_src:
        sys.modules.pop(k, None)

    agent_root_str = str(_AGENT_SYS_ROOT)
    if agent_root_str in sys.path:
        sys.path.remove(agent_root_str)
    sys.path.insert(0, agent_root_str)

    try:
        yield
    finally:
        for k in [k for k in sys.modules if k == "src" or k.startswith("src.")]:
            sys.modules.pop(k, None)

        if agent_root_str in sys.path:
            sys.path.remove(agent_root_str)

        sys.modules.update(finsight_src)

        finsight_str = str(_FINSIGHT_ROOT)
        if finsight_str in sys.path:
            sys.path.remove(finsight_str)
        if finsight_str not in sys.path:
            sys.path.insert(0, finsight_str)


# ---------------------------------------------------------------------------
# Market data helpers
# ---------------------------------------------------------------------------

def _get_ticker_info(ticker: str) -> dict:
    defaults = {"company": ticker.upper(), "sector": "Unknown", "benchmark": "QQQ"}
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        return {
            "company": info.get("longName") or info.get("shortName") or ticker.upper(),
            "sector":  info.get("sector") or "Unknown",
            "benchmark": "QQQ",
        }
    except Exception:
        return defaults


def _get_technical_factors(ticker: str) -> dict:
    """Compute technical factors; return empty dict on any failure."""
    try:
        from src.finance_news_analyzer.technical_factors import compute_technical_factors
        return compute_technical_factors(ticker)
    except Exception:
        return {"factors": {}, "summary": "Technical data unavailable.", "error": "import failed"}


def _get_macro_events(ticker: str, company: str = "", sector: str = "") -> dict:
    """Fetch high-impact macro/geopolitical events; return empty on failure."""
    try:
        from src.finance_news_analyzer.macro_events import fetch_macro_events
        return fetch_macro_events(ticker=ticker, company=company, sector=sector, max_events=10)
    except Exception:
        return {"events": [], "summary": "Macro events unavailable.", "error": "import failed"}


def _get_market_snapshot(ticker: str) -> dict:
    defaults = {
        "last_price": None,
        "day_change": None,
        "benchmark_change": None,
        "relative_strength": None,
        "volume_vs_average": None,
        "valuation_note": "Market data unavailable.",
    }
    try:
        import yfinance as yf
        yt   = yf.Ticker(ticker)
        hist = yt.history(period="30d")
        if hist.empty or len(hist) < 2:
            return defaults

        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2])
        day_chg = (last - prev) / prev if prev else 0.0

        bh = yf.Ticker("QQQ").history(period="2d")
        bench_chg = 0.0
        if len(bh) >= 2:
            b_last = float(bh["Close"].iloc[-1])
            b_prev = float(bh["Close"].iloc[-2])
            bench_chg = (b_last - b_prev) / b_prev if b_prev else 0.0

        avg_vol  = float(hist["Volume"].mean()) if "Volume" in hist else 1.0
        last_vol = float(hist["Volume"].iloc[-1]) if "Volume" in hist else 1.0
        vol_ratio = (last_vol / avg_vol) if avg_vol else 1.0

        pe = None
        try:
            pe = (yt.info or {}).get("trailingPE") or (yt.info or {}).get("forwardPE")
        except Exception:
            pass

        if pe:
            val_note = (
                f"P/E {pe:.1f} – premium multiple; growth must sustain."
                if pe > 50
                else f"P/E {pe:.1f} – {'elevated' if pe > 25 else 'reasonable'} valuation."
            )
        else:
            val_note = "Valuation data not available via API."

        return {
            "last_price":        round(last, 2),
            "day_change":        round(day_chg, 4),
            "benchmark_change":  round(bench_chg, 4),
            "relative_strength": round(day_chg - bench_chg, 4),
            "volume_vs_average": round(vol_ratio, 2),
            "valuation_note":    val_note,
        }
    except Exception:
        return defaults


# ---------------------------------------------------------------------------
# Heuristic agents
# ---------------------------------------------------------------------------

# Expanded bull/bear word sets for better coverage of financial news language
_BULL_WORDS = {
    # Earnings / revenue
    "beat", "beats", "surpass", "surpassed", "record", "record-high",
    "outperform", "outperformed", "exceeded", "exceeds",
    # Growth signals
    "growth", "rally", "rallied", "upgrade", "upgraded", "buy",
    "bullish", "strong", "strength", "demand", "revenue", "profit",
    "earnings", "positive", "momentum", "breakthrough", "milestone",
    "rise", "rose", "surge", "surged", "gain", "gained", "gains",
    "optimistic", "expand", "expanding", "expansion", "accelerate",
    "accelerating", "recovery", "recover", "recovered",
    # Tech/AI sector
    "AI", "artificial intelligence", "cloud", "chip", "semiconductor",
    "partnership", "contract", "deal", "acquisition",
    # Market signals
    "buy-rating", "price-target", "raised", "overweight", "outperform",
    "opportunity", "undervalued", "catalyst", "guidance-raise",
}
_BEAR_WORDS = {
    # Earnings misses
    "miss", "misses", "missed", "disappoint", "disappointed",
    "disappointing", "decline", "declined", "declining",
    # Negative actions
    "cut", "cuts", "downgrade", "downgraded", "sell", "bearish",
    "weak", "weakness", "loss", "losses", "layoff", "layoffs",
    "risk", "risks", "concern", "concerns", "warn", "warned", "warning",
    "fall", "fell", "drop", "dropped", "tumble", "tumbled",
    "negative", "recession", "slowdown", "debt", "deficit",
    # Legal / regulatory
    "investigation", "lawsuit", "penalty", "fine", "fined", "ban",
    "sanction", "sanctions", "tariff", "tariffs", "competition",
    "antitrust", "regulation", "regulatory", "headwind", "headwinds",
    # Market signals
    "underperform", "underperforming", "sell-rating", "downside",
    "overvalued", "bubble", "crash", "correction", "volatile",
}


def _score_chunk(text: str) -> float:
    """
    Return a sentiment score in [-1, +1] for a text chunk.
    Uses word-level matching with expanded vocabulary.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    text_lower = text.lower()

    # Word-level scoring
    bull = sum(1 for w in words if w in _BULL_WORDS)
    bear = sum(1 for w in words if w in _BEAR_WORDS)

    # Phrase-level boosts (multi-word signals common in financial news)
    bull_phrases = [
        "beat expectations", "beat estimates", "topped estimates",
        "record revenue", "record profit", "strong results",
        "raised guidance", "raised outlook", "bullish outlook",
        "price target raised", "initiates buy", "buy rating",
        "positive surprise", "better than expected",
    ]
    bear_phrases = [
        "missed expectations", "missed estimates", "below estimates",
        "lowered guidance", "cut forecast", "bearish outlook",
        "price target cut", "initiates sell", "sell rating",
        "negative surprise", "worse than expected", "concerns about",
        "faces headwinds", "profit warning",
    ]
    for phrase in bull_phrases:
        if phrase in text_lower:
            bull += 2
    for phrase in bear_phrases:
        if phrase in text_lower:
            bear += 2

    total = bull + bear
    if total == 0:
        return 0.0
    return round((bull - bear) / total, 3)


# Source suffixes to strip from catalyst phrases
_SOURCE_SUFFIXES = re.compile(
    r"\s*[-–—|]\s*(?:Yahoo Finance|Bloomberg|Reuters|CNBC|MarketWatch|Benzinga|"
    r"The Motley Fool|Seeking Alpha|Barron's|Financial Times|TipRanks|MarketBeat|"
    r"Investopedia|TheStreet|Zacks|Nasdaq\.com|Business Wire|PR Newswire)\.?\s*$",
    re.IGNORECASE,
)


def _extract_catalysts(text: str) -> list[str]:
    """
    Pull out short catalyst phrases using simple pattern matching.
    Returns clean, properly-capitalized phrases without source name suffixes.
    """
    # Strip meta-prefixes before scanning
    for prefix in ("[HIGH-AUTHORITY", "[Credibility", "[QUANTITATIVE", "[HIGH-IMPACT"):
        if text.startswith(prefix):
            text = text[text.find("]")+1:].strip()
            break

    catalysts: list[str] = []
    # Each pattern captures a FULL sentence containing the keyword, not starting from keyword
    # Use sentence-level extraction to get complete context
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keywords = [
        "beat", "beats", "surpassed", "exceeded", "record",
        "earnings", "revenue", "profit", "EPS", "guidance", "outlook",
        "raised", "raises", "upgraded", "upgrade", "AI", "cloud", "chip",
        "acquisition", "merger", "partnership", "deal", "contract",
        "tariff", "sanction", "regulation", "ban",
        "announced", "reported", "posted",
    ]
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 200:
            continue
        sent_lower = sent.lower()
        if any(kw.lower() in sent_lower for kw in keywords):
            # Strip trailing source attribution
            cleaned = _SOURCE_SUFFIXES.sub("", sent).strip().rstrip(".")
            # Ensure it starts with a capital letter
            if cleaned and not cleaned[0].isupper():
                cleaned = cleaned[0].upper() + cleaned[1:]
            # Skip if it's just a news headline title with no body content
            if cleaned and 20 < len(cleaned) < 180:
                catalysts.append(cleaned)
        if len(catalysts) >= 4:
            break
    return catalysts


# ---------------------------------------------------------------------------
# Technical bias helper
# ---------------------------------------------------------------------------

def _compute_technical_bias(factors: dict) -> dict:
    """
    Compute a directional bias (bullish/bearish/neutral) and confidence
    adjustment from technical indicators.
    """
    if not factors:
        return {"bias": "neutral", "conf_adj": 0.0, "notes": []}

    bull_pts = 0
    bear_pts = 0
    notes: list[str] = []

    # RSI - In uptrends, overbought can stay overbought (momentum indicator)
    rsi = factors.get("rsi14")
    if rsi is not None:
        if rsi < 30:
            bull_pts += 2
            notes.append(f"RSI-14 oversold ({rsi:.0f}) — potential bounce setup.")
        elif rsi < 40:
            bull_pts += 1
            notes.append(f"RSI-14 approaching oversold ({rsi:.0f}) — recovering momentum.")
        elif rsi >= 85:
            # Extreme overbought - only penalize if VERY extreme
            bear_pts += 1
            notes.append(f"RSI-14 extreme overbought ({rsi:.0f}) — potential exhaustion.")
        elif rsi >= 70:
            # Overbought but often means strong trend - NEUTRAL to slightly bullish
            bull_pts += 1
            notes.append(f"RSI-14 high ({rsi:.0f}) — strong momentum zone.")
        elif 40 <= rsi <= 60:
            notes.append(f"RSI-14 neutral ({rsi:.0f}).")

    # MACD histogram
    mh = factors.get("macd_histogram")
    if mh is not None:
        if mh > 0.01:
            bull_pts += 1
            notes.append(f"MACD histogram positive (+{mh:.4f}) — bullish momentum.")
        elif mh < -0.01:
            bear_pts += 1
            notes.append(f"MACD histogram negative ({mh:.4f}) — bearish momentum.")

    # SMA crossover
    cross = factors.get("sma_cross", "flat")
    if "bullish" in cross:
        bull_pts += 2
        notes.append(f"SMA crossover: {cross} — trend confirmation.")
    elif "bearish" in cross:
        bear_pts += 2
        notes.append(f"SMA crossover: {cross} — downtrend signal.")

    # Price vs SMA-50
    pvs = factors.get("price_vs_sma50")
    if pvs is not None:
        if pvs > 0.05:
            bull_pts += 1
            notes.append(f"Price {pvs:+.1%} above SMA-50 — supportive trend.")
        elif pvs < -0.05:
            bear_pts += 1
            notes.append(f"Price {pvs:+.1%} below SMA-50 — bearish trend.")

    # Momentum (1-month)
    mom1m = factors.get("mom_1m")
    if mom1m is not None:
        if mom1m > 0.03:
            bull_pts += 1
            notes.append(f"1-month momentum positive (+{mom1m:.1%}) — upward price trend.")
        elif mom1m < -0.03:
            bear_pts += 1
            notes.append(f"1-month momentum negative ({mom1m:.1%}) — downward price trend.")

    # Bollinger Band position
    bp = factors.get("bb_position")
    if bp is not None:
        if bp > 0.85:
            bear_pts += 1
            notes.append(f"Price near upper Bollinger Band ({bp:.2f}) — extended, risk of pullback.")
        elif bp < 0.15:
            bull_pts += 1
            notes.append(f"Price near lower Bollinger Band ({bp:.2f}) — potentially oversold.")

    # Volume
    vol_r = factors.get("vol_5d_vs_20d")
    if vol_r is not None and vol_r > 1.5:
        notes.append(f"Volume surge: 5d avg {vol_r:.1f}x above 20d avg — elevated interest.")

    # OBV trend
    obv = factors.get("obv_trend", "flat")
    if obv == "rising":
        bull_pts += 1
        notes.append("On-Balance Volume rising — buying pressure accumulating.")
    elif obv == "falling":
        bear_pts += 1
        notes.append("On-Balance Volume falling — selling pressure.")

    # Derive bias
    if bull_pts > bear_pts + 1:
        bias = "bullish"
    elif bear_pts > bull_pts + 1:
        bias = "bearish"
    else:
        bias = "neutral"

    net = bull_pts - bear_pts
    conf_adj = round(min(max(net * 0.015, -0.10), 0.10), 3)

    return {"bias": bias, "conf_adj": conf_adj, "notes": notes}


# Broad market ETFs / indices — relative strength vs QQQ is 0 by definition
_BENCHMARK_TICKERS = {"QQQ", "SPY", "DIA", "IWM", "VTI", "VIX", "GLD", "TLT", "XLF", "XLE"}


def _heuristic_analyst(chunks) -> dict:
    """
    Analyst agent: keyword extraction, evidence classification.
    Synthetic context chunks (is_context_only=True) are skipped for
    sentiment scoring but noted as context.

    Thresholds:
    - ±0.04 to classify a chunk as supporting/contradicting
    - ±0.06 for overall balance (avoids noise-driven bullish calls)
    - Requires ≥ 2 supporting items with no overwhelming contradictors
      to declare 'bullish' (otherwise stays 'mixed')
    """
    supporting, contradicting, catalysts, macro_ctx = [], [], [], []
    scores = []
    credibilities = []

    for ch in chunks:
        text   = getattr(ch, "text", "") or ""
        source = getattr(ch, "source", "")
        cid    = getattr(ch, "citation_id", "C1")
        cred   = float(getattr(ch, "credibility_weight", 0.60))

        # Skip sentiment scoring for context-only synthetic chunks
        if getattr(ch, "is_context_only", False):
            clean = getattr(ch, "excerpt_text", None) or ""
            if clean:
                macro_ctx.append(f"{source}: {clean[:150]}")
            continue

        score = _score_chunk(text)
        scores.append(score)
        credibilities.append(cred)

        excerpt = text[:180].strip()

        # Macro signals
        if any(kw in text.lower() for kw in ("fed", "interest rate", "inflation", "gdp", "macro", "sector")):
            macro_ctx.append(f"{source}: {excerpt[:100]}")

        # Only count chunks that have a clear directional signal
        if score > 0.04:
            supporting.append({"claim": excerpt, "citation_ids": [cid], "_score": score, "_cred": cred})
        elif score < -0.04:
            contradicting.append({"claim": excerpt, "citation_ids": [cid], "_score": score, "_cred": cred})

        catalysts.extend(_extract_catalysts(text))

    avg_score    = sum(scores) / len(scores) if scores else 0.0
    avg_cred     = sum(credibilities) / len(credibilities) if credibilities else 0.60
    n_supporting = len(supporting)
    n_contra     = len(contradicting)

    # Strict balance: need at least 2 supporting AND avg_score > 0.06 for bullish
    # (avoids noise-driven bullish call from a single analyst upgrade article)
    if avg_score > 0.06 and n_supporting >= 2 and n_contra == 0:
        balance = "bullish"
    elif avg_score > 0.06 and n_supporting >= 2 and n_supporting > n_contra * 2:
        balance = "bullish"
    elif avg_score < -0.06 and n_contra >= 2 and n_contra > n_supporting * 2:
        balance = "bearish"
    elif avg_score < -0.04 and n_contra > n_supporting:
        balance = "bearish"
    else:
        balance = "mixed"  # Default when evidence is ambiguous

    return {
        "ticker":                "?",
        "event_summary":         (
            f"Evidence reviewed across {len(chunks)} sources. "
            f"Balance: {balance}. Avg score: {avg_score:.3f}. "
            f"Supporting: {n_supporting}, Contradicting: {n_contra}, Avg credibility: {avg_cred:.2f}."
        ),
        "supporting_evidence":   supporting[:5],
        "contradicting_evidence":contradicting[:3],
        "uncertainties":         ["Macro conditions subject to change", "Earnings execution risk"],
        "primary_catalysts":     list(dict.fromkeys(catalysts))[:4],
        "macro_context":         macro_ctx[:2],
        "staleness_flags":       [],
        "evidence_balance":      balance,
        "_avg_score":            avg_score,
        "_avg_cred":             avg_cred,
        "_n_supporting":         n_supporting,
        "_n_contra":             n_contra,
    }


def _heuristic_strategist(analyst: dict, ticker: str, sector: str,
                           tech_bias: dict = None) -> dict:
    """Strategist agent: synthesise thesis and direction with technical integration."""
    avg_score  = analyst.get("_avg_score", 0.0)
    balance    = analyst.get("evidence_balance", "mixed")
    catalysts  = analyst.get("primary_catalysts", [])
    macro_ctx  = analyst.get("macro_context", [])
    is_etf     = ticker.upper() in _BENCHMARK_TICKERS

    # Integrate technical bias ONLY for individual stocks, not ETFs/indices.
    # For ETFs (QQQ, SPY, etc.) technical indicators are circular — the ETF IS
    # the benchmark, so technical "bullish" means the broad market is up, which
    # is already priced in.  Only allow ETF direction to follow news balance.
    tech_dir   = tech_bias.get("bias", "neutral") if tech_bias and not is_etf else "neutral"
    tech_notes = tech_bias.get("notes", []) if tech_bias else []

    # For individual stocks: mixed news + STRONG technicals → follow technicals.
    # "Strong" = net ≥ 4 points (conf_adj ≥ 0.060), not just any bullish indicator.
    tech_conf = abs(tech_bias.get("conf_adj", 0.0)) if tech_bias else 0.0
    if balance == "mixed" and tech_dir in ("bullish", "bearish") and tech_conf >= 0.060 and not is_etf:
        balance = tech_dir
        avg_score = 0.07 if tech_dir == "bullish" else -0.07

    if balance == "bullish":
        direction = "bullish"
        thesis    = (
            f"{ticker} shows bullish catalysts driven by "
            + (catalysts[0] if catalysts else "positive newsflow")
            + f". Sector ({sector}) momentum appears constructive."
        )
        if tech_notes:
            thesis += f" Technical indicators support the bullish outlook: {'; '.join(tech_notes[:2])}."
        causal_chain = "Positive news → increased demand/revenue → higher earnings expectations → price appreciation."
        horizon   = "5d"
        risks     = ["Market-wide risk-off could override fundamental positives",
                     "Execution risk on near-term guidance",
                     "Valuation premium may limit upside"]
    elif balance == "bearish":
        direction = "bearish"
        thesis    = (
            f"{ticker} faces headwinds from "
            + (catalysts[0] if catalysts else "negative newsflow")
            + f". {sector} sector risks are elevated."
        )
        if tech_notes:
            thesis += f" Technical indicators confirm bearish pressure: {'; '.join(tech_notes[:2])}."
        causal_chain = "Negative news → investor concern → multiple compression → potential downside."
        horizon   = "5d"
        risks     = ["Recovery faster than expected if macro improves",
                     "Short covering could cause brief price spike",
                     "Institutional support may absorb selling pressure"]
    else:
        direction = "neutral"
        thesis    = (
            f"{ticker} exhibits mixed signals. Evidence is balanced between "
            "positive catalysts and risk factors. Conviction is low pending "
            "further data or a clearer catalyst."
        )
        causal_chain = "Balanced evidence → no clear directional edge → hold or await clarification."
        horizon   = "20d"
        risks     = ["Catalyst could resolve either direction", "Macro surprise risk"]

    macro_notes = macro_ctx[:2] if macro_ctx else ["No strong macro signal detected."]
    return {
        "ticker":                    ticker,
        "direction":                 direction,
        "horizon":                   horizon,
        "thesis":                    thesis,
        "causal_chain":              causal_chain,
        "risks":                     risks,
        "citations":                 [c.get("citation_ids", [""])[0]
                                      for c in analyst.get("supporting_evidence", [])[:4]
                                      if c.get("citation_ids")],
        "counterarguments":          [c["claim"][:100]
                                      for c in analyst.get("contradicting_evidence", [])[:2]],
        "invalidation_conditions":   ["Reversal of reported catalyst", "Macro shock"],
        "thesis_strength":           "high" if abs(avg_score) > 0.15 else ("medium" if abs(avg_score) > 0.05 else "low"),
        "market_context_notes":      macro_notes,
    }


def _heuristic_decision(analyst: dict, strategist: dict,
                        tech_bias: dict | None = None) -> dict:
    """
    Decision agent: produce final signal with genuinely evidence-driven confidence.

    Confidence formula (no artificial floors):
      base = avg_source_credibility × directional_consistency × signal_intensity
      + tech_confirmation_bonus (±0.05 to 0.10)
      + evidence_count_bonus (0 to 0.08)
      clamped to [0.20, 0.85]

    This means:
      - No evidence / all neutral  → ~0.25–0.35 (low)
      - One weak signal            → ~0.30–0.40 (low-moderate)
      - Multiple aligned signals   → ~0.45–0.60 (moderate)
      - Strong consistent evidence → ~0.60–0.75 (high)
      - Bloomberg + tech confirm   → up to 0.85 (very high)
    """
    direction    = strategist.get("direction", "neutral")
    avg_score    = analyst.get("_avg_score", 0.0)
    avg_cred     = analyst.get("_avg_cred", 0.65)
    n_supporting = analyst.get("_n_supporting", 0)
    n_contra     = analyst.get("_n_contra", 0)
    strength     = strategist.get("thesis_strength", "low")

    # Directional consistency: fraction of evidence that agrees with direction
    n_total = n_supporting + n_contra
    if n_total > 0:
        if direction == "bullish":
            consistency = n_supporting / n_total
        elif direction == "bearish":
            consistency = n_contra / n_total
        else:
            # Neutral: neither side dominates — measure balance
            consistency = 1.0 - abs(n_supporting - n_contra) / max(n_total, 1)
    else:
        consistency = 0.4   # no evidence → low consistency

    # Signal intensity: how strong is the avg sentiment score
    # Scale: 0.0 = no signal, 1.0 = maximum signal strength
    signal_intensity = min(abs(avg_score) * 4.0, 1.0)

    # Evidence count bonus: more articles = more confidence, diminishing returns
    # 0 articles → 0.0, 2 → 0.04, 4 → 0.06, 5+ → 0.08
    evidence_bonus = min(max(n_total - 1, 0) * 0.02, 0.08)

    # Base confidence from news evidence quality
    # Formula: credibility × consistency × (signal_base + intensity_boost)
    signal_base = 0.35   # Even with 0 intensity, decent credibility/consistency gives some confidence
    base_conf   = avg_cred * consistency * (signal_base + signal_intensity * 0.65) + evidence_bonus

    # Technical bias adjustment (limited influence — max ±0.10)
    tech_adj   = 0.0
    tech_notes = []
    tech_prefix = ""
    if tech_bias:
        t_bias     = tech_bias.get("bias", "neutral")
        t_conf_adj = tech_bias.get("conf_adj", 0.0)   # already in [-0.10, +0.10] range
        tech_notes = tech_bias.get("notes", [])

        if t_bias == direction:
            # Technicals CONFIRM direction → up to +0.10 bonus
            tech_adj  = min(abs(t_conf_adj) * 2.0, 0.10)
            agree_str = "confirming"
        elif t_bias != "neutral" and direction != "neutral" and t_bias != direction:
            # Technicals CONTRADICT direction → up to -0.08 penalty
            tech_adj  = -min(abs(t_conf_adj) * 1.5, 0.08)
            agree_str = "contradicting"
        else:
            agree_str = "neutral"

        if tech_notes:
            tech_prefix = (
                f"Technical indicators ({agree_str} news signal): "
                + "; ".join(tech_notes[:3])
                + ". "
            )

    # Final confidence — clamp to realistic range
    # No floor: weak evidence really does produce 0.25–0.35 confidence
    confidence = round(min(max(base_conf + tech_adj, 0.20), 0.85), 2)

    # Validation note explaining how confidence was computed
    val_note = (
        f"Confidence = cred({avg_cred:.2f}) × consistency({consistency:.2f}) × "
        f"signal({signal_intensity:.2f}) + evidence_bonus({evidence_bonus:.2f}) "
        f"+ tech_adj({tech_adj:+.2f}) = {confidence:.2f}"
    )

    reasoning = (
        tech_prefix
        + f"The {direction.capitalize()} thesis for this asset is based on "
        f"analysis of {n_supporting} supporting and {n_contra} contradicting "
        f"evidence items across {n_total} directional chunks. "
        f"Thesis strength: {strength}. "
        + strategist.get("thesis", "")
    )

    mkt_notes = list(strategist.get("market_context_notes", []))
    if tech_notes:
        mkt_notes = tech_notes[:2] + mkt_notes

    return {
        "ticker":               "?",
        "direction":            direction,
        "horizon":              strategist.get("horizon", "20d" if direction == "neutral" else "5d"),
        "confidence":           confidence,
        "reasoning":            reasoning[:800],
        "risks":                strategist.get("risks", []),
        "citations":            strategist.get("citations", []),
        "abstain":              False,
        "validation_notes":     [val_note],
        "disagreement_signal":  False,
        "disagreement_reason":  "",
        "memory_notes":         [],
        "market_context_notes": mkt_notes[:4],
    }


def run_heuristic_pipeline(
    ticker: str,
    rag_chunks: list,
    sector: str = "",
    tech_bias: dict | None = None,
) -> dict:
    """
    Run the heuristic three-agent pipeline locally (no LLM, no agent system).
    Returns a FinalDecision-compatible dict.
    """
    analyst_out    = _heuristic_analyst(rag_chunks)
    analyst_out["ticker"] = ticker

    strategist_out = _heuristic_strategist(analyst_out, ticker, sector, tech_bias=tech_bias)
    decision_out   = _heuristic_decision(analyst_out, strategist_out, tech_bias=tech_bias)
    decision_out["ticker"] = ticker

    return decision_out, {
        "analyst_output":    analyst_out,
        "strategist_output": strategist_out,
    }


# ---------------------------------------------------------------------------
# LLM pipeline (unified: RAG chunks → OpenAI agents → final answer)
# ---------------------------------------------------------------------------

def run_llm_pipeline(
    ticker:         str,
    rag_chunks:     list,
    sector:         str = "",
    retrieval_query: str = "",
    openai_api_key: str | None = None,
    openai_model:   str = "gpt-4o-mini",
    tech_factors:   dict | None = None,
    macro_summary:  str = "",
) -> tuple[dict, dict]:
    """
    Run the full OpenAI-powered pipeline:
      RAG chunks (news evidence) → Analyst → Strategist → Decision → final answer.

    This is the UNIFIED path. All context (news + technical + macro) is passed
    as RAG chunks to the agents so the final answer integrates everything.

    Returns (final_decision_dict, workflow_state_dict).
    """
    if not _AGENT_AVAILABLE:
        raise ImportError(
            f"Agent system not found at {_AGENT_SYS_ROOT}. "
            "Ensure person2_agent_system_handoff is present alongside FinSight_RAG."
        )

    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["AGENT_BACKEND"] = "langchain_openai"
    os.environ["OPENAI_MODEL"]  = openai_model

    with _agent_sys_ctx():
        from src.models.data_packet import (  # type: ignore
            FinalDecision, RetrievedChunk, WorkflowInput,
        )
        from src.orchestration.memory_store import PatternMemoryStore  # type: ignore
        from src.orchestration.workflow import MultiAgentWorkflow  # type: ignore

        # Build LLM client
        llm_client = None
        if openai_api_key:
            try:
                from src.llm.factory import build_llm_client_from_env  # type: ignore
                schema_root = _AGENT_SYS_ROOT / "schemas"
                llm_client  = build_llm_client_from_env(schema_root=schema_root)
            except Exception:
                llm_client = None

        # Memory store
        memory_store = None
        try:
            mem_path     = _AGENT_SYS_ROOT / "outputs" / "memory" / "signal_memory.json"
            memory_store = PatternMemoryStore(mem_path)
        except Exception:
            pass

        # Build WorkflowInput from RAG chunks
        # All chunks (news + TechChunk + MacroChunk) are passed so agents
        # have the complete picture when generating their answer.
        agent_chunks = []
        for rc in rag_chunks:
            authority_tag = (
                "[HIGH-AUTHORITY SOURCE – Bloomberg] "
                if getattr(rc, "is_bloomberg", False)
                else f"[Credibility {getattr(rc, 'credibility_weight', 0.6):.2f}] "
            )
            agent_chunks.append(RetrievedChunk(
                citation_id = getattr(rc, "citation_id", str(uuid.uuid4())[:8]),
                source      = getattr(rc, "source", "Unknown"),
                title       = getattr(rc, "title", "News article"),
                published_at= getattr(rc, "published_at", datetime.now(timezone.utc).isoformat()),
                ticker      = getattr(rc, "ticker", ticker),
                text        = authority_tag + getattr(rc, "text", ""),
            ))

        workflow_input = WorkflowInput(
            ticker          = ticker.upper(),
            query_date      = datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            chunks          = agent_chunks,
            sector          = sector,
            retrieval_query = retrieval_query or f"{ticker} investment analysis",
        )

        # Run the LangGraph workflow
        prompt_root = _AGENT_SYS_ROOT / "prompts"
        workflow    = MultiAgentWorkflow(
            llm_client   = llm_client,
            prompt_root  = str(prompt_root),
            memory_store = memory_store,
        )

        final_decision = workflow.run(workflow_input)
        return final_decision.to_dict(), {}


# ---------------------------------------------------------------------------
# Signal packet builder
# ---------------------------------------------------------------------------

_DIR_MAP    = {"bullish": "Bullish", "bearish": "Bearish", "neutral": "Neutral"}
_RAND_DIRS  = ["Bullish", "Bearish", "Neutral"]


def _horizon_days(horizon: str) -> int:
    return 20 if "20" in str(horizon) else 5


def _compute_source_quality(rag_chunks: list) -> float:
    if not rag_chunks:
        return 0.60
    weights = [getattr(c, "credibility_weight", 0.60) for c in rag_chunks]
    return round(sum(weights) / len(weights), 2)


def _clean_excerpt(text: str, max_len: int = 300) -> str:
    """
    Extract a clean, non-repetitive excerpt from article text.
    Uses fuzzy deduplication so near-identical sentences (e.g., with/without
    source suffix like '- Bloomberg.com') are collapsed into one.
    """
    if not text:
        return ""

    # Strip meta-prefixes injected by the pipeline
    for prefix in ("[HIGH-AUTHORITY", "[Credibility", "[Source", "[QUANTITATIVE", "[HIGH-IMPACT"):
        if text.startswith(prefix):
            text = text[text.find("]")+1:].strip()
            break

    # Split into candidate sentences
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    seen_fingerprints: list[str] = []
    unique_parts: list[str] = []

    for p in parts:
        p = p.strip()
        if not p or len(p) < 20:
            continue

        # Fingerprint: only alphanumeric words, lowercase, first 10 words
        words = re.findall(r'[a-zA-Z0-9]+', p.lower())
        fingerprint = " ".join(words[:10])

        # Fuzzy dedup: skip if 80%+ overlap with any already-seen fingerprint
        is_dupe = False
        fp_words = set(words[:10])
        for seen_fp in seen_fingerprints:
            seen_words = set(seen_fp.split())
            if len(fp_words) > 0 and len(seen_words) > 0:
                overlap = len(fp_words & seen_words) / max(len(fp_words), len(seen_words))
                if overlap >= 0.80:
                    is_dupe = True
                    break

        if not is_dupe:
            seen_fingerprints.append(fingerprint)
            unique_parts.append(p)

    clean = " ".join(unique_parts)
    if not clean:
        # If everything was deduped, just show the first sentence
        first_parts = re.split(r'[.!?]', text.strip())
        clean = first_parts[0].strip() if first_parts else text[:max_len]

    return (clean[:max_len].rstrip() + "…") if len(clean) > max_len else clean


def _build_citations(rag_chunks: list) -> list[dict]:
    seen, out = set(), []
    for ch in rag_chunks:
        cid = getattr(ch, "citation_id", "")
        if cid in seen:
            continue
        seen.add(cid)
        raw_text = getattr(ch, "text", "")
        for prefix_pattern in ("[HIGH-AUTHORITY", "[Credibility", "[Source",
                                "[QUANTITATIVE", "[HIGH-IMPACT"):
            if raw_text.startswith(prefix_pattern):
                raw_text = raw_text[raw_text.find("]")+1:].strip()
                break
        _excerpt = getattr(ch, "excerpt_text", None) or _clean_excerpt(raw_text)
        out.append({
            "source":             getattr(ch, "source", "Unknown"),
            "title":              getattr(ch, "title", "News article"),
            "url":                f"https://finance.yahoo.com/quote/{getattr(ch, 'ticker', '')}",
            "excerpt":            _excerpt,
            "credibility_weight": float(getattr(ch, "credibility_weight", 0.60)),
        })
    return out


def _build_trace(analyst: dict, strategist: dict, decision: dict) -> list[dict]:
    return [
        {
            "agent":   "Analyst Agent",
            "summary": analyst.get("event_summary", "Evidence extraction complete."),
        },
        {
            "agent":   "Strategist Agent",
            "summary": strategist.get("thesis", "Investment thesis formed."),
        },
        {
            "agent":   "Decision Agent",
            "summary": (
                f"Direction: {decision.get('direction','n/a').capitalize()}. "
                f"Confidence: {decision.get('confidence', 0):.0%}. "
                + (decision.get("reasoning", "")[:160] or "")
            ),
        },
    ]


def build_signal_packet(
    final_decision_dict: dict,
    workflow_state:      dict,
    rag_chunks:          list,
    ticker_info:         dict,
    market_snapshot:     dict,
    tech_bias:           dict | None = None,
    used_llm:            bool = False,
) -> dict:
    """
    Convert agent pipeline outputs into the FinSight UI SignalPacket dict.

    When used_llm=True (OpenAI path), the agents have already read all RAG
    context including technical and macro chunks — their direction and
    confidence are trusted as-is.

    When used_llm=False (heuristic path), technical bias is used to upgrade
    a Neutral direction when technicals are clearly directional.
    """
    ticker      = final_decision_dict.get("ticker", "?").upper()
    direction   = _DIR_MAP.get(final_decision_dict.get("direction", "neutral"), "Neutral")
    confidence  = float(final_decision_dict.get("confidence", 0.5))
    horizon     = _horizon_days(final_decision_dict.get("horizon", "5d"))
    reasoning   = final_decision_dict.get("reasoning", "No reasoning provided.")
    risks       = final_decision_dict.get("risks", [])
    mem_notes   = final_decision_dict.get("memory_notes", [])
    mkt_notes   = final_decision_dict.get("market_context_notes", [])

    analyst_out    = workflow_state.get("analyst_output", {})
    strategist_out = workflow_state.get("strategist_output", {})

    # Thesis bullets
    thesis_bullets = list(analyst_out.get("primary_catalysts", []))
    for m in analyst_out.get("macro_context", [])[:2]:
        thesis_bullets.append(f"Macro: {m}")

    # Counter-evidence
    counter_ev = [
        item.get("claim", "") if isinstance(item, dict) else str(item)
        for item in analyst_out.get("contradicting_evidence", [])
    ]

    # Watch items
    watch_items = (mem_notes + mkt_notes)[:4] or [
        f"Monitor {ticker} guidance update",
        "Watch sector rotation signals",
    ]

    # Catalyst
    catalyst = (
        (analyst_out.get("primary_catalysts") or [""])[0]
        or strategist_out.get("causal_chain", reasoning[:80])
    )

    # ── Sentiment score ───────────────────────────────────────────────────
    # Compute from real (non-context) chunks only
    avg_score = analyst_out.get("_avg_score", 0.0)
    if avg_score == 0.0:
        chunk_scores = [
            _score_chunk(getattr(c, "text", ""))
            for c in rag_chunks
            if not getattr(c, "is_context_only", False)
        ]
        if chunk_scores:
            avg_score = round(sum(chunk_scores) / len(chunk_scores), 3)

    # Blend technical signal into sentiment score when news is neutral
    if tech_bias and abs(avg_score) < 0.12:
        t_bias = tech_bias.get("bias", "neutral")
        t_adj  = tech_bias.get("conf_adj", 0.0)
        if t_bias == "bullish" and t_adj > 0:
            avg_score = round(avg_score * 0.6 + abs(t_adj) * 2 * 0.4, 3)
        elif t_bias == "bearish" and t_adj < 0:
            avg_score = round(avg_score * 0.6 - abs(t_adj) * 2 * 0.4, 3)

    if avg_score == 0.0:
        avg_score = confidence * (1 if direction == "Bullish" else -1 if direction == "Bearish" else 0)

    sentiment_score = round(float(avg_score), 3)

    # ── TECHNICAL OVERRIDE — only when technicals are VERY strong (≥4 net points)
    # AND the ticker is NOT a broad market ETF/index (where tech signals are circular)
    # AND the direction is currently Neutral (don't override directional calls)
    # Requirements:
    #   - Heuristic: conf_adj ≥ 0.060 (4+ net tech points)
    #   - LLM: conf_adj ≥ 0.075 (5+ net tech points) AND confidence < 0.55
    #   - Never applies to ETF/index tickers (QQQ, SPY, etc.)
    is_etf_ticker = ticker.upper() in _BENCHMARK_TICKERS
    if tech_bias and not is_etf_ticker:
        t_bias   = tech_bias.get("bias", "neutral")
        t_notes  = tech_bias.get("notes", [])
        t_conf   = abs(tech_bias.get("conf_adj", 0.0))
        override_threshold = 0.060 if not used_llm else 0.075
        if (t_bias in ("bullish", "bearish") and t_conf >= override_threshold
                and direction == "Neutral"
                and (not used_llm or confidence < 0.55)):
            direction  = "Bullish" if t_bias == "bullish" else "Bearish"
            tech_note_str = "; ".join(t_notes[:3])
            reasoning = (
                f"Technical indicators ({t_conf:.0%} confidence): {tech_note_str}. "
                f"News sentiment is mixed, but strong quantitative signals indicate "
                f"{direction.lower()} price action. "
                + reasoning
            )
            confidence = round(min(max(confidence, 0.50), 0.75), 2)

    # Source quality and novelty
    source_quality = _compute_source_quality(rag_chunks)
    novelty_score  = min(round(len({getattr(c, "source", "") for c in rag_chunks}) / 5.0, 2), 1.0)

    # Baselines — sentiment baseline now matches the actual RAG direction to avoid
    # confusing "different answers". The random baseline remains a coinflip.
    # Sentiment baseline = what you'd get from raw sentiment alone (same direction
    # as our computed avg_score, which IS based on the retrieved chunks).
    baseline_sentiment = "Bullish" if sentiment_score > 0.05 else ("Bearish" if sentiment_score < -0.05 else "Neutral")
    baseline_random    = random.choice(_RAND_DIRS)

    event_type = analyst_out.get("evidence_balance", "mixed").capitalize() + " signals"

    company = ticker_info.get("company", ticker)
    sector  = ticker_info.get("sector", "Unknown")

    return {
        "id":                f"sig-{company[:6].lower().replace(' ', '-')}-{uuid.uuid4().hex[:6]}",
        "ticker":            ticker,
        "company":           company,
        "sector":            sector,
        "benchmark":         ticker_info.get("benchmark", "QQQ"),
        "event_type":        event_type,
        "direction":         direction,
        "horizon_days":      horizon,
        "confidence":        round(confidence, 2),
        "novelty_score":     novelty_score,
        "sentiment_score":   sentiment_score,
        "source_quality":    source_quality,
        "published_at":      datetime.now(timezone.utc).isoformat(),
        "reasoning":         reasoning,
        "catalyst":          (catalyst[:200] if catalyst else "News catalyst identified."),
        "thesis_bullets":    thesis_bullets[:5],
        "risk_factors":      risks[:4],
        "counter_evidence":  counter_ev[:3],
        "watch_items":       watch_items[:4],
        "market_snapshot":   market_snapshot,
        "citations":         _build_citations(rag_chunks)[:6],
        "agent_trace":       _build_trace(analyst_out, strategist_out, final_decision_dict),
        "baseline_sentiment":baseline_sentiment,
        "baseline_random":   baseline_random,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_full_pipeline(
    ticker:           str,
    openai_api_key:   str | None = None,
    openai_model:     str = "gpt-4o-mini",
    top_k:            int = 8,
    include_rss:      bool = True,
    bloomberg_config=None,  # BloombergConfig | None
) -> dict:
    """
    Full pipeline: ingest news → RAG retrieval → (tech + macro enrichment)
    → agent pipeline (heuristic or OpenAI) → SignalPacket.

    When an OpenAI API key is provided, this is a UNIFIED path:
      - RAG chunks (news articles) are retrieved
      - Technical factors and macro events are injected as additional chunks
      - ALL evidence is passed together to the OpenAI-powered agents
      - OpenAI (GPT-4o-mini) produces the final directional answer

    There is only ONE answer — from the agents. The sentiment score is a
    diagnostic computed from the raw chunks, not a competing signal.

    Args:
        ticker:            Stock ticker symbol (e.g. "NVDA").
        openai_api_key:    OpenAI API key.  If None, heuristic agents are used.
        openai_model:      OpenAI model name (default: gpt-4o-mini).
        top_k:             Number of RAG chunks to retrieve.
        include_rss:       Whether to include RSS feeds.
        bloomberg_config:  BloombergConfig for B-PIPE API, or None to skip.

    Returns:
        A SignalPacket dict with a ``_pipeline_meta`` key containing stats.
    """
    from src.finance_news_analyzer.rag_pipeline import RAGPipeline  # type: ignore

    ticker      = ticker.upper().strip()
    ticker_info = _get_ticker_info(ticker)
    company     = ticker_info.get("company", "")
    sector      = ticker_info.get("sector", "")

    pipeline = RAGPipeline(
        ticker          = ticker,
        company         = company,
        retrieval_query = (
            f"{ticker} {company} investment thesis earnings catalyst "
            "revenue guidance sector macro analysis"
        ),
        top_k            = top_k,
        include_rss      = include_rss,
        bloomberg_config = bloomberg_config,
    )
    pipeline.ingest()

    # Fetch top_k + 2 real news chunks (compensate for 2 synthetic chunks we prepend)
    chunks = pipeline.get_chunks(top_k=top_k + 2)
    if not chunks:
        raise ValueError(
            f"No relevant news articles found for {ticker}. "
            "Try a different ticker or check your internet connection."
        )

    ticker_info["ticker"] = ticker
    market_snapshot       = _get_market_snapshot(ticker)

    # ── Compute technical factors ─────────────────────────────────────────
    tech = _get_technical_factors(ticker)
    tech_summary = tech.get("summary", "")
    tech_factors = tech.get("factors", {})

    if tech_factors:
        market_snapshot["rsi14"]             = tech_factors.get("rsi14")
        market_snapshot["macd_histogram"]    = tech_factors.get("macd_histogram")
        market_snapshot["sma_cross"]         = tech_factors.get("sma_cross")
        market_snapshot["mom_1m"]            = tech_factors.get("mom_1m")
        market_snapshot["mom_3m"]            = tech_factors.get("mom_3m")
        market_snapshot["bb_position"]       = tech_factors.get("bb_position")
        market_snapshot["hist_vol_20d"]      = tech_factors.get("hist_vol_20d")
        market_snapshot["pct_from_52w_high"] = tech_factors.get("pct_from_52w_high")

    # ── Inject Technical Analysis chunk into RAG context ──────────────────
    # Both heuristic and LLM agents see quantitative signals as part of their
    # evidence — this ensures ONE unified answer from the agents.
    _tk = ticker
    _now_iso = datetime.now(timezone.utc).isoformat()
    if tech_summary and tech_summary != "Technical data unavailable.":
        _tf = tech_factors
        def _pv(v, fmt=".2f"):
            return "n/a" if v is None else format(v, fmt)
        _tech_excerpt = (
            f"RSI-14: {_pv(_tf.get('rsi14'), '.1f')} | "
            f"MACD hist: {_pv(_tf.get('macd_histogram'), '+.4f')} | "
            f"SMA signal: {_tf.get('sma_cross', 'n/a')} | "
            f"1m momentum: {_pv(_tf.get('mom_1m'), '+.1%') if _tf.get('mom_1m') is not None else 'n/a'} | "
            f"3m momentum: {_pv(_tf.get('mom_3m'), '+.1%') if _tf.get('mom_3m') is not None else 'n/a'} | "
            f"BB position: {_pv(_tf.get('bb_position'), '.2f')} | "
            f"OBV trend: {_tf.get('obv_trend', 'n/a')} | "
            f"52w high: {_pv(_tf.get('pct_from_52w_high'), '+.1%') if _tf.get('pct_from_52w_high') is not None else 'n/a'}"
        )
        _tech_excerpt_val = _tech_excerpt

        class _TechChunk:
            citation_id        = "TECH-001"
            source             = "Technical Analysis (FinSight Quant)"
            title              = f"Quantitative Technical Factors: {_tk}"
            published_at       = _now_iso
            ticker             = _tk
            text               = (
                "[QUANTITATIVE TECHNICAL ANALYSIS — HIGH PRIORITY]\n"
                "The following technical factors were computed from real price/volume "
                f"data for {_tk}. Agents MUST incorporate these quantitative signals "
                "into their thesis and confidence calibration.\n\n"
                + tech_summary
            )
            excerpt_text       = _tech_excerpt_val
            credibility_weight = 0.85
            is_bloomberg       = False
            is_context_only    = True   # skip heuristic sentiment scoring

        chunks = [_TechChunk()] + list(chunks)

    # ── Fetch macro events and inject as a chunk ──────────────────────────
    macro = _get_macro_events(_tk, company=company, sector=sector)
    macro_summary = macro.get("summary", "")
    macro_events  = macro.get("events", [])

    _macro_relevant = macro.get("ticker_relevant_count", 0) >= 1
    if macro_events and macro_summary and "unavailable" not in macro_summary.lower() and _macro_relevant:
        _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        _macro_headlines = " | ".join(
            e["headline"][:90] for e in macro_events[:4]
        )
        _macro_excerpt_val = f"Top events: {_macro_headlines}"

        class _MacroChunk:
            citation_id        = "MACRO-001"
            source             = "Global Macro Intelligence (FinSight)"
            title              = f"High-Impact Macro & Geopolitical Events — {_today}"
            published_at       = _now_iso
            ticker             = _tk
            text               = (
                "[HIGH-IMPACT MACRO & GEOPOLITICAL EVENTS — CRITICAL CONTEXT]\n"
                "The following real-world macro and geopolitical events have been "
                "detected in current news feeds. They may materially affect this "
                "investment thesis. Agents MUST integrate them into the analysis.\n\n"
                + macro_summary
            )
            excerpt_text       = _macro_excerpt_val
            credibility_weight = 0.92
            is_bloomberg       = False
            is_context_only    = True   # skip heuristic sentiment scoring

        chunks = [_MacroChunk()] + list(chunks)

    # ── Technical bias (used for heuristic path only) ─────────────────────
    _tech_bias = _compute_technical_bias(tech_factors)

    # ── Choose pipeline mode ─────────────────────────────────────────────
    # UNIFIED path: when OpenAI key is provided, ALL chunks (news + tech + macro)
    # go to the LangGraph agents → GPT-4o-mini produces ONE final answer.
    use_llm = bool(openai_api_key) and _AGENT_AVAILABLE

    if use_llm:
        try:
            final_dict, state = run_llm_pipeline(
                ticker          = ticker,
                rag_chunks      = chunks,
                sector          = sector,
                retrieval_query = pipeline.retrieval_query,
                openai_api_key  = openai_api_key,
                openai_model    = openai_model,
                tech_factors    = tech_factors,
                macro_summary   = macro_summary,
            )
            # LLM path doesn't return full workflow_state — reconstruct basics
            # so build_signal_packet can assemble thesis bullets, etc.
            if not state:
                # Try to extract richer state from the final decision
                llm_reasoning = final_dict.get("reasoning", "")
                llm_direction = final_dict.get("direction", "neutral")
                # Run heuristic analyst on the NEWS chunks only (not tech/macro context)
                # to get catalysts and evidence items for the signal packet
                news_only_chunks = [c for c in chunks if not getattr(c, "is_context_only", False)]
                _h_analyst = _heuristic_analyst(news_only_chunks)
                state = {
                    "analyst_output":    {
                        "_avg_score":          _h_analyst.get("_avg_score", 0.0),
                        "evidence_balance":    _h_analyst.get("evidence_balance", llm_direction),
                        "primary_catalysts":   _h_analyst.get("primary_catalysts", []),
                        "contradicting_evidence": _h_analyst.get("contradicting_evidence", []),
                        "supporting_evidence": _h_analyst.get("supporting_evidence", []),
                        "event_summary":       llm_reasoning[:200],
                        "macro_context":       _h_analyst.get("macro_context", []),
                    },
                    "strategist_output": {
                        "thesis":              llm_reasoning,
                        "causal_chain":        "",
                        "market_context_notes": [],
                    },
                }
        except Exception as llm_err:
            # Graceful fallback to heuristic
            final_dict, state = run_heuristic_pipeline(
                ticker, chunks, sector, tech_bias=_tech_bias)
            use_llm = False
    else:
        # Heuristic mode — tech bias augments the local agents
        final_dict, state = run_heuristic_pipeline(
            ticker, chunks, sector, tech_bias=_tech_bias)

    packet = build_signal_packet(
        final_decision_dict = final_dict,
        workflow_state      = state,
        rag_chunks          = chunks,
        ticker_info         = ticker_info,
        market_snapshot     = market_snapshot,
        tech_bias           = _tech_bias,
        used_llm            = use_llm,
    )
    packet["_pipeline_meta"] = {
        "articles_fetched":  pipeline.article_count,
        "chunks_indexed":    pipeline.chunk_count,
        "chunks_retrieved":  len(chunks),
        "sources":           pipeline.source_names,
    }
    return packet
