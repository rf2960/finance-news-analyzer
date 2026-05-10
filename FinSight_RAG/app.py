from __future__ import annotations

import json
import threading
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.finance_news_analyzer.evaluation import (
    attach_forward_returns,
    build_metric_table,
    load_prices,
    load_signals,
    summarize_method,
)

# Lazy import so the UI loads even if optional agent deps are missing
def _try_import_runner():
    try:
        from src.finance_news_analyzer.agent_runner import run_full_pipeline
        return run_full_pipeline
    except Exception:
        return None


ROOT = Path(__file__).resolve().parent
SIGNALS_PATH = ROOT / "demo_data" / "signals.json"
PRICES_PATH = ROOT / "demo_data" / "prices.csv"
EVALUATION_SIGNALS_PATH = ROOT / "demo_data" / "evaluation_signals.json"


# ── Helpers for market-scan research-packet creation ──────────────────────────

def _create_scan_packet(snap) -> dict:
    """Convert a StockSnapshot into a minimal SignalPacket for the sidebar."""
    import uuid
    from datetime import datetime, timezone
    chg_str = f"{snap.day_change_pct*100:+.3f}%" if snap.day_change_pct else "n/a"
    vol_str  = f"{snap.volume_ratio:.2f}x" if snap.volume_ratio else "n/a"
    bullets  = [
        f"Price: ${snap.price:.2f}" if snap.price else "",
        f"Day change: {chg_str}",
        f"Volume vs 5d avg: {vol_str}",
    ]
    if snap.news_mentions:
        bullets.append(f"News mentions: {snap.news_mentions} articles")
    bullets = [b for b in bullets if b]
    sources_str = ", ".join(snap.mention_sources[:3]) if snap.mention_sources else "volume scan"
    return {
        "id":               f"sig-{snap.ticker.lower()}-scan-{uuid.uuid4().hex[:6]}",
        "ticker":           snap.ticker,
        "company":          snap.name or snap.ticker,
        "sector":           snap.sector or "Unknown",
        "benchmark":        "QQQ",
        "event_type":       "Market scan",
        "direction":        "Neutral",
        "horizon_days":     5,
        "confidence":       0.50,
        "novelty_score":    round(min(snap.volume_ratio / 3.0, 1.0), 2) if snap.volume_ratio else 0.50,
        "sentiment_score":  0.0,
        "source_quality":   0.65,
        "published_at":     datetime.now(timezone.utc).isoformat(),
        "reasoning": (
            f"{snap.ticker} identified via market scan. "
            f"Price ${snap.price:.2f}, day change {chg_str}, "
            f"volume {vol_str} vs average. "
            "Run Live Analysis for a full agent signal."
        ),
        "catalyst":         f"Market scan — discovered via {sources_str}",
        "thesis_bullets":   bullets,
        "risk_factors":     ["No RAG analysis performed", "Confidence set to neutral (0.50)"],
        "counter_evidence": [],
        "watch_items":      [f"Run Live Analysis on {snap.ticker} for a full investment signal"],
        "market_snapshot": {
            "last_price":        round(snap.price, 2) if snap.price else None,
            "day_change":        round(snap.day_change_pct, 6) if snap.day_change_pct else None,
            "benchmark_change":  0.0,
            "relative_strength": 0.0,
            "volume_vs_average": round(snap.volume_ratio, 2) if snap.volume_ratio else None,
            "valuation_note":    "Market scan data only — no full RAG analysis.",
        },
        "citations": [],
        "agent_trace": [
            {"agent": "Market Scan", "summary": f"Stock discovered via {sources_str}."},
        ],
        "baseline_sentiment": "Neutral",
        "baseline_random":    "Neutral",
    }


def _append_to_signals(path: Path, packet: dict):
    """Append a packet to signals.json; return (success, message)."""
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        ids = {s.get("id") for s in existing}
        # Also check if ticker already has a scan packet
        scan_tickers = {s["ticker"] for s in existing if "scan" in s.get("id", "")}
        if packet["id"] in ids:
            return False, f"{packet['ticker']} scan packet already exists."
        if packet["ticker"] in scan_tickers:
            # Update existing scan packet
            for i, s in enumerate(existing):
                if s["ticker"] == packet["ticker"] and "scan" in s.get("id", ""):
                    existing[i] = packet
                    break
        else:
            existing.append(packet)
        path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
        return True, f"{packet['ticker']} saved."
    except Exception as err:
        return False, str(err)


@st.cache_data(ttl=300)
def _get_index_snapshot():
    """Fetch 1-day return for SPY, QQQ, DIA, IWM (market indices)."""
    result = {}
    try:
        import yfinance as yf
        for sym in ("SPY", "QQQ", "DIA", "IWM"):
            try:
                h = yf.Ticker(sym).history(period="2d")
                if len(h) >= 2:
                    last = float(h["Close"].iloc[-1])
                    prev = float(h["Close"].iloc[-2])
                    result[sym] = round((last - prev) / prev * 100, 3) if prev else 0.0
            except Exception:
                result[sym] = None
    except Exception:
        pass
    return result


@st.cache_data(ttl=3600)
def _load_real_prices_df(tickers: tuple) -> "pd.DataFrame":
    """
    Fetch 60-day OHLCV for the given tickers from yfinance and return a
    long-format DataFrame compatible with load_prices() output.
    Columns: ticker, date, open, high, low, close, volume
    """
    rows = []
    try:
        import yfinance as yf
        for tkr in tickers:
            try:
                h = yf.Ticker(tkr).history(period="60d")
                if h.empty:
                    continue
                for dt, row in h.iterrows():
                    rows.append({
                        "ticker": tkr,
                        "date": str(dt.date()),
                        "open": float(row.get("Open", 0)),
                        "high": float(row.get("High", 0)),
                        "low": float(row.get("Low", 0)),
                        "close": float(row.get("Close", 0)),
                        "volume": int(row.get("Volume", 0)),
                    })
            except Exception:
                continue
    except Exception:
        pass
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume"])


st.set_page_config(
    page_title="FinSight RAG",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


st.markdown(
    """
    <style>
    :root {
        --ink: #17202a;
        --muted: #667085;
        --line: #d9e1e8;
        --panel: #ffffff;
        --soft: #f5f7fa;
        --accent: #126c83;
        --accent-soft: #e7f3f6;
        --good: #0f7a4f;
        --bad: #b42318;
        --warn: #946200;
    }
    .block-container {
        padding-top: 0.45rem;
        padding-bottom: 2rem;
        max-width: 1380px;
    }
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    .stDeployButton,
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    h1 {
        font-size: 1.65rem !important;
        line-height: 1.15 !important;
        margin-bottom: 0.15rem !important;
        letter-spacing: 0;
    }
    h2, h3 {
        letter-spacing: 0;
    }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stDataFrame"] {
        font-size: 0.86rem;
    }
    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.55rem 0.7rem;
        min-height: 78px;
    }
    div[data-testid="stMetric"] label {
        color: var(--muted);
        font-size: 0.72rem !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.18rem !important;
        color: var(--ink) !important;
    }
    div[data-testid="stMetric"] {
        background: var(--panel) !important;
    }
    .app-shell {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(180deg, #fbfcfd 0%, #ffffff 42%);
        padding: 0.95rem 1rem;
        margin-top: 0.15rem;
        margin-bottom: 0.75rem;
        color: var(--ink);
    }
    .eyebrow {
        color: var(--accent);
        font-size: 0.72rem;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 0.25rem;
    }
    .terminal-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.55rem;
        margin-top: 0.8rem;
    }
    .strip-cell {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        padding: 0.55rem 0.65rem;
        color: var(--ink);
    }
    .strip-label {
        color: var(--muted);
        font-size: 0.7rem;
        margin-bottom: 0.15rem;
    }
    .strip-value {
        color: var(--ink);
        font-weight: 700;
        font-size: 0.95rem;
    }
    .signal-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--panel);
        padding: 0.72rem 0.8rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        color: var(--ink);
    }
    .signal-head {
        display: flex;
        justify-content: space-between;
        gap: 0.6rem;
        align-items: flex-start;
        margin-bottom: 0.35rem;
    }
    .ticker {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--ink);
    }
    .company {
        color: var(--muted);
        font-size: 0.72rem;
    }
    .pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.14rem 0.48rem;
        font-size: 0.68rem;
        font-weight: 700;
        border: 1px solid transparent;
        white-space: nowrap;
    }
    .pill-bullish {
        background: #e9f8f0;
        color: var(--good);
        border-color: #bde7ce;
    }
    .pill-bearish {
        background: #fff0ee;
        color: var(--bad);
        border-color: #ffd0ca;
    }
    .pill-neutral {
        background: #f2f4f7;
        color: #475467;
        border-color: #d0d5dd;
    }
    .compact-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.45rem;
        margin: 0.5rem 0;
    }
    .mini-stat {
        background: var(--soft);
        border-radius: 7px;
        padding: 0.42rem 0.5rem;
    }
    .mini-label {
        color: var(--muted);
        font-size: 0.66rem;
        margin-bottom: 0.1rem;
    }
    .mini-value {
        font-weight: 700;
        font-size: 0.82rem;
        color: var(--ink);
    }
    .section-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: -0.3rem;
        margin-bottom: 0.55rem;
    }
    .analysis-box {
        border-left: 3px solid var(--accent);
        background: var(--accent-soft);
        padding: 0.6rem 0.75rem;
        border-radius: 7px;
        font-size: 0.84rem;
        margin-bottom: 0.65rem;
        color: var(--ink);
    }
    .source-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.62rem 0.68rem;
        margin-bottom: 0.5rem;
        background: #fff;
        color: var(--ink);
    }
    .source-title {
        font-size: 0.84rem;
        font-weight: 700;
        color: var(--ink);
    }
    .source-meta {
        color: var(--muted);
        font-size: 0.7rem;
        margin-bottom: 0.25rem;
    }
    .small-list li {
        margin-bottom: 0.18rem;
    }
    @media (max-width: 900px) {
        .terminal-strip,
        .compact-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = load_signals(SIGNALS_PATH)
    prices = load_prices(PRICES_PATH)
    evaluated = attach_forward_returns(signals, prices)
    if evaluated.empty or "id" not in evaluated.columns:
        evaluated = signals.copy()
        for _ec in ["return_5d","return_20d","entry_date","entry_close","exit_date_5d","exit_date_20d"]:
            if _ec not in evaluated.columns: evaluated[_ec] = None
    return signals, prices, evaluated


@st.cache_data
def load_historical_evaluation_sample() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the bundled historical sample used when live signals are too recent to score."""
    if not EVALUATION_SIGNALS_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()

    sample_signals = load_signals(EVALUATION_SIGNALS_PATH)
    sample_prices = load_prices(PRICES_PATH)
    sample_evaluated = attach_forward_returns(sample_signals, sample_prices)
    return sample_signals, sample_evaluated


def has_forward_coverage(evaluated_df: pd.DataFrame, horizon_days: int = 5) -> bool:
    ret_col = f"return_{horizon_days}d"
    return (
        not evaluated_df.empty
        and ret_col in evaluated_df.columns
        and evaluated_df[ret_col].notna().any()
    )


def build_dashboard_metrics(evaluated_df: pd.DataFrame) -> pd.DataFrame:
    """Build the Evaluation tab table from actual attached forward returns."""
    rows = []
    method_map = {
        "direction": "Multi-Agent RAG",
        "baseline_sentiment": "Sentiment Baseline",
        "baseline_random": "Random Baseline",
    }

    for horizon in (5, 20):
        return_col = f"return_{horizon}d"
        coverage = (
            evaluated_df[return_col].notna().mean()
            if return_col in evaluated_df.columns and len(evaluated_df)
            else 0.0
        )
        for column, label in method_map.items():
            summary = summarize_method(evaluated_df, column, horizon)
            rows.append(
                {
                    "method": label,
                    "horizon": f"{horizon}d",
                    "signals": summary.signals,
                    "hit_rate": summary.hit_rate,
                    "avg_signed_return": summary.avg_signed_return,
                    "avg_raw_return": summary.avg_raw_return,
                    "signal_coverage": coverage,
                }
            )
    return pd.DataFrame(rows)


def outcome_display_frame(evaluated_df: pd.DataFrame) -> pd.DataFrame:
    """Format signal-level RAG outcomes for the Evaluation tab."""
    if evaluated_df.empty:
        return pd.DataFrame()

    rows = []
    for _, row in evaluated_df.iterrows():
        ret5 = row.get("return_5d")
        ret20 = row.get("return_20d")
        rows.append(
            {
                "ticker": row.get("ticker"),
                "direction": row.get("direction"),
                "Confidence": pct(row.get("confidence"), 0),
                "5d Return": "" if pd.isna(ret5) else pct(ret5, 2),
                "5d Hit": "" if pd.isna(ret5) else ("Pass" if summarize_method(pd.DataFrame([row]), "direction", 5).hit_rate == 1 else "Miss"),
                "20d Return": "" if pd.isna(ret20) else pct(ret20, 2),
                "20d Hit": "" if pd.isna(ret20) else ("Pass" if summarize_method(pd.DataFrame([row]), "direction", 20).hit_rate == 1 else "Miss"),
            }
        )
    return pd.DataFrame(rows)


def pct(value: float | int | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.{decimals}%}"


def num(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.2f}{suffix}"


def signal_class(direction: str) -> str:
    return {
        "Bullish": "pill-bullish",
        "Bearish": "pill-bearish",
        "Neutral": "pill-neutral",
    }.get(direction, "pill-neutral")


def render_pill(label: str) -> str:
    return f'<span class="pill {signal_class(label)}">{label}</span>'


def list_html(items: list[str]) -> str:
    if not items:
        return "<span class='company'>No items.</span>"
    return "<ul class='small-list'>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def selected_prices(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    return prices[prices["ticker"] == ticker].sort_values("date").copy()


PRODUCT_BENCHMARKS = [
    {
        "product": "AlphaSense",
        "pattern": "Premium source library plus sentence-level citation and integrated workflow",
        "fin_sight_translation": "Evidence Audit keeps every thesis linked to source cards and credibility scores.",
    },
    {
        "product": "GNOMI",
        "pattern": "Real-time intelligence layer with verification, context, and global event monitoring",
        "fin_sight_translation": "Market Monitor separates raw sentiment from model conviction and watch items.",
    },
    {
        "product": "Quartr",
        "pattern": "Live earnings calls, transcripts, filings, reports, and LLM-ready event data",
        "fin_sight_translation": "Agent packets are structured so future earnings transcripts can drop into the same UI.",
    },
    {
        "product": "Fiscal.ai",
        "pattern": "Fundamental terminal with dashboards, KPIs, AI summaries, auditability, estimates, and IR content",
        "fin_sight_translation": "Thesis Workspace pairs qualitative claims with quant snapshots and valuation notes.",
    },
    {
        "product": "Koyfin / YCharts",
        "pattern": "Custom dashboards, watchlists, charts, portfolio context, and client-ready reports",
        "fin_sight_translation": "Evaluation Lab turns generated ideas into presentation-ready metrics and charts.",
    },
    {
        "product": "BloombergGPT literature",
        "pattern": "Domain-specific financial LLM evaluation across NLP tasks and finance benchmarks",
        "fin_sight_translation": "The project evaluates outputs against market returns, not only text quality.",
    },
]


METHODOLOGY_CHECKS = [
    ("Grounding", "Every final signal must cite retrieved evidence, source type, and timestamp."),
    ("No look-ahead", "News timestamp must precede entry price and forward-return labels."),
    ("Baseline discipline", "Compare RAG against random, sentiment, and eventually no-RAG LLM baselines."),
    ("Calibration", "Bucket hit rate by confidence to test whether confidence means anything."),
    ("Contrarian value", "Flag cases where sentiment and RAG direction disagree."),
    ("Presentation maturity", "Show thesis, risk, source audit, and outcome in one continuous workflow."),
]


signals, prices, evaluated = load_demo_data()

# ── Restore saved scan tickers from signals.json on first load ───────────────
# This ensures tickers saved with session_only=OFF appear in the sidebar after restarts
if "saved_scan_tickers" not in st.session_state:
    try:
        _persisted_sigs = json.loads(SIGNALS_PATH.read_text(encoding="utf-8"))
        _persisted_scan_tks = list(dict.fromkeys(
            s["ticker"] for s in _persisted_sigs if "scan" in s.get("id", "")
        ))
        if _persisted_scan_tks:
            st.session_state.saved_scan_tickers = _persisted_scan_tks
    except Exception:
        pass

# Augment demo prices with real yfinance data for NON-SCAN tickers only.
# Scan packets don't need price history — skipping them prevents slow re-fetches
# every time a new scan ticker is added.
try:
    _non_scan_for_prices = signals[~signals["id"].str.contains("scan", na=False)]
    _real_tickers = tuple(sorted(_non_scan_for_prices["ticker"].unique().tolist()))
    _real_prices = _load_real_prices_df(_real_tickers)
    if not _real_prices.empty:
        # Merge real prices on top of demo prices (real data overrides demo for matching tickers)
        _demo_only = prices[~prices["ticker"].isin(_real_prices["ticker"].unique())]
        prices = pd.concat([_demo_only, _real_prices], ignore_index=True)
        evaluated = attach_forward_returns(signals, prices)
        if evaluated.empty or "id" not in evaluated.columns:
            evaluated = signals.copy()
            for _ec in ["return_5d","return_20d","entry_date","entry_close","exit_date_5d","exit_date_20d"]:
                if _ec not in evaluated.columns: evaluated[_ec] = None
except Exception:
    pass

# Merge session-only scan packets (not persisted to disk) into signals DataFrame.
# We skip re-running attach_forward_returns here because scan packets have no real
# price data; this keeps "Save to Tickers" instantaneous.
_sess_pkts = st.session_state.get("scan_session_packets", [])
if _sess_pkts:
    try:
        _extra_rows = [{col: _p.get(col) for col in signals.columns} for _p in _sess_pkts]
        _extra_df = pd.DataFrame(_extra_rows, columns=signals.columns)
        signals = pd.concat([signals, _extra_df], ignore_index=True)
    except Exception:
        pass

# Deduplicate signals: keep highest confidence per (ticker, horizon_days) combination.
# This ensures both 5d and 20d signals are preserved for the same ticker.
try:
    signals = signals.sort_values("confidence", ascending=False).drop_duplicates(subset=["ticker", "horizon_days"], keep="first").reset_index(drop=True)
    # Recompute evaluated and metrics after dedup
    evaluated = attach_forward_returns(signals, prices)
    if evaluated.empty or "id" not in evaluated.columns:
        evaluated = signals.copy()
        for _ec in ["return_5d","return_20d","entry_date","entry_close","exit_date_5d","exit_date_20d"]:
            if _ec not in evaluated.columns: evaluated[_ec] = None
except Exception:
    pass

try:
    metrics = build_metric_table(evaluated)
except Exception:
    metrics = pd.DataFrame([
        {"method": "Multi-Agent RAG", "horizon": "5d", "signals": 0, "hit_rate": 0.0, "avg_signed_return": 0.0, "avg_raw_return": 0.0},
        {"method": "Multi-Agent RAG", "horizon": "20d", "signals": 0, "hit_rate": 0.0, "avg_signed_return": 0.0, "avg_raw_return": 0.0},
        {"method": "Sentiment Baseline", "horizon": "5d", "signals": 0, "hit_rate": 0.5, "avg_signed_return": 0.0, "avg_raw_return": 0.0},
        {"method": "Random Baseline", "horizon": "5d", "signals": 0, "hit_rate": 0.5, "avg_signed_return": 0.0, "avg_raw_return": 0.0},
    ])

eval_sample_signals, eval_sample_evaluated = load_historical_evaluation_sample()

with st.sidebar:
    _title_col, _gear_col = st.columns([0.82, 0.18])
    with _title_col:
        st.markdown("### FinSight RAG")
    with _gear_col:
        with st.popover("⚙️", help="API keys & Bloomberg settings"):
            st.markdown("#### ⚙️ Settings")

            # ── OpenAI ──────────────────────────────────────────────────
            st.markdown("**🔑 OpenAI API Key**")
            st.text_input(
                "OpenAI key",
                type="password",
                placeholder="sk-…  |  blank = heuristic mode",
                key="settings_openai_key",
                help="With a key GPT-4o-mini drives all three agents. "
                     "Without: fast local keyword-heuristic agents.",
            )
            st.caption("Leave blank for heuristic (offline) mode.")

            st.divider()

            # ── Bloomberg B-PIPE ─────────────────────────────────────────
            st.markdown("**📊 Bloomberg B-PIPE API**")
            st.caption(
                "Requires Bloomberg Terminal open (localhost:8194) "
                "or an enterprise B-PIPE server connection."
            )
            st.checkbox("Enable Bloomberg API", key="settings_bloomberg_enabled", value=False)
            if st.session_state.get("settings_bloomberg_enabled", False):
                st.text_input(
                    "Bloomberg Host",
                    value="localhost",
                    key="settings_bloomberg_host",
                )
                st.number_input(
                    "Bloomberg Port",
                    value=8194,
                    min_value=1,
                    max_value=65535,
                    key="settings_bloomberg_port",
                )
                st.text_input(
                    "App Name (enterprise B-PIPE only)",
                    placeholder="optional — leave blank for Terminal",
                    key="settings_bloomberg_app_name",
                )
                if st.button("Test Bloomberg Connection", key="test_bb_btn"):
                    try:
                        from src.finance_news_analyzer.bloomberg_api import (
                            BloombergConfig, check_bloomberg_connection,
                        )
                        _bb_cfg = BloombergConfig(
                            enabled=True,
                            host=st.session_state.get("settings_bloomberg_host", "localhost"),
                            port=int(st.session_state.get("settings_bloomberg_port", 8194)),
                            app_name=st.session_state.get("settings_bloomberg_app_name", ""),
                        )
                        _ok, _msg = check_bloomberg_connection(_bb_cfg)
                        if _ok:
                            st.success(_msg)
                        else:
                            st.warning(_msg)
                    except Exception as _bb_err:
                        st.error(f"Bloomberg test error: {_bb_err}")
            else:
                st.info(
                    "Bloomberg disabled. Toggle on to use B-PIPE news. "
                    "Falls back to Google News / RSS when disabled."
                )

            st.divider()

            # ── Ticker / Research Packet management ─────────────────────────
            st.markdown("**📋 Ticker Settings**")
            st.toggle(
                "Session-only tickers (not saved to disk)",
                key="settings_session_only",
                value=True,
                help=(
                    "ON: saved tickers live only in this browser session. "
                    "When you close the tab or reload, they are gone. "
                    "OFF: ticker packets are written to signals.json and survive restarts."
                ),
            )
            if st.session_state.get("settings_session_only", True):
                st.caption("Saved tickers are session-only (not persisted).")
            else:
                st.caption("Saved tickers will be written to signals.json.")
            st.markdown("")
            if st.button("Reset to demo packets", key="reset_packets_btn", use_container_width=True):
                try:
                    _existing = json.loads(SIGNALS_PATH.read_text(encoding="utf-8"))
                    _demo_only = [s for s in _existing if "scan" not in s.get("id", "")]
                    SIGNALS_PATH.write_text(
                        json.dumps(_demo_only, indent=2, default=str), encoding="utf-8"
                    )
                    # Clear ALL session state ticker tracking
                    st.session_state.scan_session_packets = []
                    st.session_state.saved_scan_tickers = []
                    st.session_state.live_result = None
                    st.session_state.live_meta = {}
                    load_demo_data.clear()
                    st.success(f"Reset! {len(_demo_only)} demo packets remain.")
                    st.rerun()
                except Exception as _rst_err:
                    st.error(f"Reset error: {_rst_err}")

    st.caption("Research console")
    # Sidebar Ticker: only the core research packets for Live/Monitor/Evidence
    # (Evaluation uses ALL signals independently via its own _eval_signals logic)
    _CORE_TICKERS = ["NVDA", "TSLA", "MSFT", "AMD"]
    # Include any tickers saved from Market Scan
    _saved_scan_tickers = st.session_state.get("saved_scan_tickers", [])
    _all_ticker_pool = _CORE_TICKERS + [t for t in _saved_scan_tickers if t not in _CORE_TICKERS]
    _core_options = ["All"] + [t for t in _all_ticker_pool if t in signals["ticker"].values.tolist()]
    selected_ticker_sidebar = st.selectbox(
        "Ticker",
        options=_core_options,
        key="sidebar_ticker",
        help="Selects the research packet for Live Analysis, Market Monitor, and Evidence Audit."
    )
    ticker_filter = selected_ticker_sidebar

    horizon_filter = st.radio(
        "Horizon", ["All", "5d", "20d"], horizontal=True,
        key="sidebar_horizon_filter",
    )
    direction_filter = st.multiselect(
        "Signal",
        ["Bullish", "Bearish", "Neutral"],
        default=["Bullish", "Bearish", "Neutral"],
    )
    # Auto-select best research packet for the chosen ticker
    if selected_ticker_sidebar == "All":
        selected_id = signals["id"].iloc[0] if len(signals) else None
    else:
        _t_signals = signals[signals["ticker"] == selected_ticker_sidebar]
        if not _t_signals.empty:
            best_row = _t_signals.sort_values("confidence", ascending=False).iloc[0]
            selected_id = best_row["id"]
        else:
            selected_id = signals["id"].iloc[0] if len(signals) else None

# Core tickers for non-Evaluation tabs (Live, Market Monitor, Evidence Audit)
_CORE_TICKERS = ["NVDA", "TSLA", "MSFT", "AMD"]
_saved_scan_tickers_main = st.session_state.get("saved_scan_tickers", [])
_core_tickers_extended = _CORE_TICKERS + [t for t in _saved_scan_tickers_main if t not in _CORE_TICKERS]
_core_signals = signals[signals["ticker"].isin(_core_tickers_extended)].copy()

filtered = _core_signals.copy()
if ticker_filter != "All":
    filtered = filtered[filtered["ticker"] == ticker_filter]
if horizon_filter != "All":
    filtered = filtered[filtered["horizon_days"] == int(horizon_filter.replace("d", ""))]
active_directions = direction_filter or ["Bullish", "Bearish", "Neutral"]
filtered = filtered[filtered["direction"].isin(active_directions)]

# Use core signals for Evidence Audit selection (selected packet)
_selected_source = _core_signals if (selected_id and selected_id in _core_signals["id"].values) else signals
selected = _selected_source.loc[_selected_source["id"] == selected_id].iloc[0].to_dict() if selected_id and selected_id in _selected_source["id"].values else signals.iloc[0].to_dict()
selected_eval = evaluated[evaluated["id"] == selected_id] if "id" in evaluated.columns else evaluated.head(0)

# Use _core_signals (not horizon-filtered) for header metrics so they don't zero out on 20d filter
avg_confidence = _core_signals["confidence"].mean() if len(_core_signals) else 0
avg_quality = _core_signals["source_quality"].mean() if "source_quality" in _core_signals.columns and len(_core_signals) else 0
# ── Pre-compute real forward returns for header hit rate display ──────────────
# This uses the same live yfinance evaluation as the Evaluation tab,
# so the header "5d RAG hit rate" matches the Backtest Diagnostics numbers.
@st.cache_data(ttl=3600)
def _compute_real_eval_header(signal_rows: tuple) -> pd.DataFrame:
    """Compute real forward returns from yfinance — used for header hit rate."""
    import yfinance as yf
    from datetime import datetime, timezone as _tz
    rows = []
    for tkr, direction, pub_at, horizon, confidence in signal_rows:
        try:
            try:
                sig_date = datetime.fromisoformat(str(pub_at)).date()
            except Exception:
                sig_date = datetime.now(_tz.utc).date()
            h = yf.Ticker(tkr).history(period="1y")
            if h.empty or len(h) < 3:
                continue
            h.index = [d.date() if hasattr(d, "date") else d for d in h.index]
            h.index = sorted(h.index)
            future = h[h.index >= sig_date]
            if future.empty or len(future) < 3:
                future = h.tail(25)
            entry = float(future["Close"].iloc[0])
            ret5d, hit5d = None, None
            if len(future) >= 6:
                exit5 = float(future["Close"].iloc[5])
                ret5d = (exit5 - entry) / entry if entry else 0.0
                hit5d = (direction == "Bullish" and ret5d > 0) or \
                        (direction == "Bearish" and ret5d < 0) or \
                        (direction == "Neutral")
            ret20d, hit20d = None, None
            if len(future) >= 21:
                exit20 = float(future["Close"].iloc[20])
                ret20d = (exit20 - entry) / entry if entry else 0.0
                hit20d = (direction == "Bullish" and ret20d > 0) or \
                         (direction == "Bearish" and ret20d < 0) or \
                         (direction == "Neutral")
            rows.append({"ticker": tkr, "direction": direction, "confidence": confidence,
                         "horizon_days": horizon, "ret5d": ret5d, "ret20d": ret20d,
                         "hit5d": hit5d, "hit20d": hit20d})
        except Exception:
            continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()

_header_eval_signals = signals[~signals["id"].str.contains("scan", na=False)].copy()
_header_eval_signals = _header_eval_signals.sort_values("confidence", ascending=False).drop_duplicates(subset=["ticker"], keep="first")
_header_sig_rows = tuple(
    (r["ticker"], r["direction"], str(r.get("published_at", "")),
     int(r.get("horizon_days", 5)), float(r.get("confidence", 0.5)))
    for _, r in _header_eval_signals.iterrows()
)
_header_real_eval = _compute_real_eval_header(_header_sig_rows)

def _header_hit(df, hz):
    col = f"hit{hz}d"
    if df.empty or col not in df.columns:
        return 0.0
    valid = df[df[f"ret{hz}d"].notna()]
    return float(valid[col].mean()) if len(valid) else 0.0

_hdr_hr5  = _header_hit(_header_real_eval, 5)
_hdr_hr20 = _header_hit(_header_real_eval, 20)

# Fall back to the bundled historical evaluation sample when current live
# signals are too recent to have realized forward-return labels.
_header_has_live_5d = (
    not _header_real_eval.empty
    and "ret5d" in _header_real_eval.columns
    and _header_real_eval["ret5d"].notna().any()
)
if _header_has_live_5d:
    hit_rate_5d = _hdr_hr5
elif has_forward_coverage(eval_sample_evaluated, 5):
    _sample_header_metrics = build_dashboard_metrics(eval_sample_evaluated)
    _sample_best_metric = _sample_header_metrics[
        (_sample_header_metrics["method"] == "Multi-Agent RAG")
        & (_sample_header_metrics["horizon"] == "5d")
    ]
    hit_rate_5d = _sample_best_metric["hit_rate"].iloc[0] if len(_sample_best_metric) else 0
else:
    best_metric = metrics[(metrics["method"] == "Multi-Agent RAG") & (metrics["horizon"] == "5d")]
    hit_rate_5d = best_metric["hit_rate"].iloc[0] if len(best_metric) else 0

st.markdown(
    f"""
    <div class="app-shell">
      <div class="eyebrow">Multi-agent financial news intelligence</div>
      <h1>FinSight RAG</h1>
      <div class="section-note">
        Evidence-grounded signal generation, source audit, market reaction, and forward-return evaluation.
      </div>
      <div class="terminal-strip">
        <div class="strip-cell"><div class="strip-label">Signals in view</div><div class="strip-value">{len(filtered)}</div></div>
        <div class="strip-cell"><div class="strip-label">Average confidence</div><div class="strip-value">{pct(avg_confidence, 0)}</div></div>
        <div class="strip-cell"><div class="strip-label">Source quality</div><div class="strip-value">{pct(avg_quality, 0)}</div></div>
        <div class="strip-cell"><div class="strip-label">5d RAG hit rate</div><div class="strip-value">{pct(hit_rate_5d, 0)}</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_live, tab_scan, tab_monitor, tab_evidence, tab_evaluation = st.tabs(
    ["🔴 Live Analysis", "📊 Market Scan", "📈 Market Monitor", "🔍 Evidence Audit", "🔬 Evaluation"]
)

# ── Live Analysis tab ──────────────────────────────────────────────────────
with tab_live:
    st.markdown("#### Live Multi-Agent Analysis")
    st.markdown(
        "<div class='section-note'>"
        "Fetches real news from Yahoo Finance, Bloomberg, Reuters, CNBC, and MarketWatch. "
        "Chunks and indexes the articles with TF-IDF retrieval, then routes the top evidence through "
        "the Analyst → Strategist → Decision agent pipeline to produce a structured investment signal."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Input controls ────────────────────────────────────────────────────
    ctrl_col, btn_col = st.columns([2.4, 0.8], gap="medium")
    with ctrl_col:
        live_ticker = st.text_input(
            "Ticker symbol",
            value=st.session_state.get("live_ticker", "NVDA"),
            placeholder="e.g. NVDA, MSFT, AAPL, TSLA",
        )
        live_rss = st.checkbox("Include RSS feeds (Bloomberg, Reuters, CNBC…)", value=True)
        # Horizon comes from the sidebar Horizon filter (not a separate radio here)
        _hz_sidebar = st.session_state.get("sidebar_horizon_filter", "5d")
        if _hz_sidebar == "20d":
            live_horizon = "20d"
            _hz_label = "**20d** (set in sidebar)"
        elif _hz_sidebar == "5d":
            live_horizon = "5d"
            _hz_label = "**5d** (set in sidebar)"
        else:  # "All" — don't force a specific horizon, default to 5d but label as user choice
            live_horizon = "5d"
            _hz_label = "**5d** (default — change in sidebar to set a specific horizon)"
        openai_key = st.session_state.get("settings_openai_key", "").strip() or ""
        _key_badge = "🔑 GPT-4o-mini" if openai_key else "🖥️ Heuristic"
        st.caption(f"Horizon: {_hz_label} · Mode: {_key_badge} (configure in ⚙️ Settings)")
    with btn_col:
        st.markdown("<div style='margin-top:1.55rem'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)

    st.divider()

    # ── Session state plumbing ─────────────────────────────────────────────
    if "live_result" not in st.session_state:
        st.session_state.live_result = None
    if "live_error" not in st.session_state:
        st.session_state.live_error = None
    if "live_meta" not in st.session_state:
        st.session_state.live_meta = {}

    if analyze_clicked:
        raw_ticker = (live_ticker or "").strip().upper()
        if not raw_ticker:
            st.warning("Please enter a ticker symbol.")
        else:
            run_pipeline = _try_import_runner()
            if run_pipeline is None:
                st.error(
                    "Agent system could not be imported. "
                    "Ensure `person2_agent_system_handoff/person2_agent_system/` exists and all "
                    "dependencies are installed (`pip install -r requirements.txt`)."
                )
            else:
                st.session_state.live_result = None
                st.session_state.live_error = None
                with st.spinner(
                    f"⏳ Fetching news for **{raw_ticker}** and running agent pipeline…  "
                    "(Bloomberg → Reuters → CNBC → Yahoo Finance → Analyst → Strategist → Decision)"
                ):
                    try:
                        # ── Bloomberg config from settings ─────────────────
                        _bb_enabled = st.session_state.get("settings_bloomberg_enabled", False)
                        _bb_config = None
                        if _bb_enabled:
                            try:
                                from src.finance_news_analyzer.bloomberg_api import BloombergConfig
                                _bb_config = BloombergConfig(
                                    enabled=True,
                                    host=st.session_state.get("settings_bloomberg_host", "localhost"),
                                    port=int(st.session_state.get("settings_bloomberg_port", 8194)),
                                    app_name=st.session_state.get("settings_bloomberg_app_name", ""),
                                )
                            except Exception:
                                _bb_config = None

                        # ── Resolve OpenAI key: inline → settings fallback ─
                        _openai_key = (
                            openai_key.strip()
                            or st.session_state.get("settings_openai_key", "").strip()
                            or None
                        )

                        packet = run_pipeline(
                            ticker=raw_ticker,
                            openai_api_key=_openai_key,
                            top_k=8,
                            include_rss=live_rss,
                            bloomberg_config=_bb_config,
                        )
                        # Store the horizon selection so it can be shown in results
                        # Always override horizon_days with the user's explicit selection
                        packet["horizon_days"] = 5 if live_horizon == "5d" else 20
                        st.session_state.live_result = packet
                        st.session_state.live_meta = packet.pop("_pipeline_meta", {})
                        st.session_state.live_error = None
                    except ValueError as ve:
                        st.session_state.live_error = str(ve)
                    except Exception as exc:
                        st.session_state.live_error = f"Pipeline error: {exc}"

    # ── Pipeline metadata banner ───────────────────────────────────────────
    meta = st.session_state.live_meta
    if meta:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Articles fetched", meta.get("articles_fetched", 0))
        m2.metric("Chunks indexed", meta.get("chunks_indexed", 0))
        m3.metric("Chunks retrieved", meta.get("chunks_retrieved", 0))
        m4.metric("Sources", len(meta.get("sources", [])))
        if meta.get("sources"):
            st.caption("Sources: " + " · ".join(meta["sources"]))
        st.divider()

    # ── Error display ──────────────────────────────────────────────────────
    if st.session_state.live_error:
        st.error(st.session_state.live_error)
        st.stop()

    # ── Result display ─────────────────────────────────────────────────────
    result = st.session_state.live_result
    if result is None:
        st.markdown(
            "<div class='analysis-box' style='text-align:center;color:var(--muted);'>"
            "Enter a ticker above and click <b>🔍 Analyze</b> to run the full pipeline."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        direction = result.get("direction", "Neutral")
        confidence = result.get("confidence", 0.0)
        horizon = result.get("horizon_days", 5)
        reasoning = result.get("reasoning", "")
        company = result.get("company", result.get("ticker", ""))
        sector = result.get("sector", "n/a")
        catalyst = result.get("catalyst", "")
        snap = result.get("market_snapshot") or {}

        # ── Signal headline card ───────────────────────────────────────────
        res_left, res_right = st.columns([1.1, 0.9], gap="medium")
        with res_left:
            st.markdown(
                f"""
                <div class="signal-card">
                  <div class="signal-head">
                    <div>
                      <div class="ticker">{result["ticker"]} {render_pill(direction)}</div>
                      <div class="company">{company} | {sector}</div>
                    </div>
                    <div class="company">{horizon}-day horizon</div>
                  </div>
                  <div class="analysis-box">{reasoning}</div>
                  <div class="compact-grid">
                    <div class="mini-stat"><div class="mini-label">Confidence</div>
                      <div class="mini-value">{pct(confidence, 0)}</div></div>
                    <div class="mini-stat"><div class="mini-label">Source quality</div>
                      <div class="mini-value">{pct(result.get("source_quality"), 0)}</div></div>
                    <div class="mini-stat"><div class="mini-label">Novelty</div>
                      <div class="mini-value">{pct(result.get("novelty_score"), 0)}</div></div>
                  </div>
                  <div class="company" style="margin-top:0.35rem">Catalyst: {catalyst}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Thesis drivers / risks / watch
            tb_col, rb_col = st.columns(2, gap="small")
            with tb_col:
                st.markdown("**Thesis drivers**")
                st.markdown(list_html(result.get("thesis_bullets", [])), unsafe_allow_html=True)
            with rb_col:
                st.markdown("**Risks & counter-evidence**")
                combined = result.get("risk_factors", []) + result.get("counter_evidence", [])
                st.markdown(list_html(combined), unsafe_allow_html=True)
            st.markdown("**Watch items**")
            st.markdown(list_html(result.get("watch_items", [])), unsafe_allow_html=True)

            # ── Real-world price chart (30-day) — in left column ────────────
            try:
                import yfinance as _yf30
                _hist30 = _yf30.Ticker(result["ticker"]).history(period="30d")
                if not _hist30.empty:
                    _close30 = _hist30["Close"]
                    _y_lo = _close30.min() * 0.975
                    _y_hi = _close30.max() * 1.025
                    _col_line = "#0f7a4f" if _close30.iloc[-1] >= _close30.iloc[0] else "#b42318"
                    _fill_col = "rgba(15,122,79,0.08)" if _close30.iloc[-1] >= _close30.iloc[0] else "rgba(180,35,24,0.06)"
                    st.markdown("**Recent price — 30 days**")
                    _fig30 = go.Figure(go.Scatter(
                        x=_hist30.index, y=_close30,
                        mode="lines", fill="tonexty",
                        fillcolor=_fill_col,
                        line=dict(color=_col_line, width=2),
                        name=result["ticker"],
                        hovertemplate="%{x|%b %d}: $%{y:.2f}<extra></extra>",
                    ))
                    _fig30.update_layout(
                        height=170,
                        margin=dict(l=4, r=4, t=4, b=4),
                        yaxis=dict(
                            title="USD",
                            range=[_y_lo, _y_hi],
                            tickprefix="$",
                            tickformat=".0f",
                        ),
                        xaxis_title="",
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(_fig30, use_container_width=True)
            except Exception:
                pass

        with res_right:
            # Market snapshot
            st.markdown("#### Market Snapshot")
            snap_rows = [
                ("Last price", num(snap.get("last_price"))),
                ("Day change", pct(snap.get("day_change"), 3)),
                ("QQQ benchmark change", pct(snap.get("benchmark_change"), 3)),
                ("Relative strength", pct(snap.get("relative_strength"), 3)),
                ("Volume / 30d avg", num(snap.get("volume_vs_average"), "x")),
            ]
            st.dataframe(
                pd.DataFrame(snap_rows, columns=["Metric", "Value"]),
                hide_index=True,
                use_container_width=True,
            )
            if snap.get("valuation_note"):
                st.caption(snap["valuation_note"])

            # Source citations
            st.markdown("#### Evidence Sources")
            citations = result.get("citations", [])
            for cit in citations[:5]:
                bloomberg_badge = (
                    "⭐ " if "bloomberg" in cit.get("source", "").lower() else ""
                )
                st.markdown(
                    f"""
                    <div class="source-card">
                      <div class="source-title">{bloomberg_badge}{cit.get("source","")}</div>
                      <div class="source-meta">{cit.get("title","")} | credibility {pct(cit.get("credibility_weight", 0), 0)}</div>
                      <div style="font-size:0.8rem">{cit.get("excerpt","")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if citations:
                cit_df = pd.DataFrame(citations)
                fig_cit = px.bar(
                    cit_df,
                    x="source",
                    y="credibility_weight",
                    color="source",
                    range_y=[0, 1],
                    labels={"credibility_weight": "Credibility", "source": ""},
                )
                fig_cit.update_layout(
                    height=180,
                    margin=dict(l=4, r=4, t=8, b=4),
                    showlegend=False,
                )
                st.plotly_chart(fig_cit, use_container_width=True)

        # ── Agent trace ────────────────────────────────────────────────────
        st.markdown("#### Agent Pipeline Trace")
        trace = result.get("agent_trace", [])
        for idx, step in enumerate(trace, start=1):
            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-title">{idx}. {step.get("agent","Agent")}</div>
                  <div style="font-size:0.83rem">{step.get("summary","")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ── Radar chart ────────────────────────────────────────────────────
        radar_col, save_col = st.columns([1.5, 0.5], gap="medium")
        with radar_col:
            lens_df = pd.DataFrame([
                {"lens": "Sentiment", "score": abs(result.get("sentiment_score", 0))},
                {"lens": "Novelty", "score": result.get("novelty_score", 0)},
                {"lens": "Source quality", "score": result.get("source_quality", 0)},
                {"lens": "Confidence", "score": confidence},
            ])
            fig_r = px.line_polar(lens_df, r="score", theta="lens", line_close=True, range_r=[0, 1])
            fig_r.update_traces(fill="toself", line_color="#126c83")
            fig_r.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            st.plotly_chart(fig_r, use_container_width=True)

        with save_col:
            st.markdown("#### Save to signals")
            st.caption(
                "Append this live result to `demo_data/signals.json` so it "
                "appears in the Market Monitor and Evaluation tabs."
            )
            if st.button("💾 Append to signals.json", use_container_width=True):
                try:
                    # Strip internal meta key before saving
                    save_packet = {k: v for k, v in result.items() if not k.startswith("_")}
                    existing = json.loads(SIGNALS_PATH.read_text(encoding="utf-8"))
                    # Avoid duplicate ids
                    existing_ids = {s.get("id") for s in existing}
                    if save_packet.get("id") not in existing_ids:
                        existing.append(save_packet)
                        SIGNALS_PATH.write_text(
                            json.dumps(existing, indent=2, default=str),
                            encoding="utf-8",
                        )
                        st.success("Saved! Reload the page to see it in Market Monitor.")
                        load_demo_data.clear()
                    else:
                        st.info("This signal ID already exists in signals.json.")
                except Exception as save_err:
                    st.error(f"Save failed: {save_err}")

# ── Market Scan tab ────────────────────────────────────────────────────────
with tab_scan:
    st.markdown("#### 📊 Market Scan")
    st.markdown(
        "<div class='section-note'>"
        "Two complementary discovery methods: <b>Top 100 by Volume/Price</b> uses yfinance data "
        "across the NASDAQ-100 + S&amp;P 500 universe. <b>News-Driven Discovery</b> scans live "
        "RSS feeds and extracts the most-mentioned tickers from today's headlines."
        "</div>",
        unsafe_allow_html=True,
    )

    scan_col_a, scan_col_b = st.columns(2, gap="large")

    # ── Section A: Top 100 by Market Activity ─────────────────────────────
    with scan_col_a:
        st.markdown("##### 🔢 Top Stocks by Market Activity")
        st.caption("Fetches 5-day OHLCV data for the full NASDAQ-100 + S&P 500 universe via yfinance.")
        sort_by_opt = st.selectbox(
            "Rank by",
            options=["volume", "volume_ratio", "price", "market_cap"],
            format_func=lambda x: {
                "volume":       "Trading Volume (absolute)",
                "volume_ratio": "Volume vs 5d Average (activity spike)",
                "price":        "Share Price",
                "market_cap":   "Market Capitalization",
            }.get(x, x),
            key="scan_sort_by",
        )
        top_n_opt = st.slider("Number of stocks to show", min_value=10, max_value=130, value=50, step=10, key="scan_top_n")

        if st.button("🔍 Run Volume/Price Scan", type="primary", use_container_width=True, key="run_volume_scan"):
            st.session_state.scan_volume_result = None
            st.session_state.scan_volume_error = None
            with st.spinner(f"Fetching market data for {top_n_opt} stocks from {len(__import__('src.finance_news_analyzer.stock_screener', fromlist=['ALL_TICKERS']).ALL_TICKERS)} tickers…"):
                try:
                    from src.finance_news_analyzer.stock_screener import top_stocks_by_market_activity
                    result_stocks = top_stocks_by_market_activity(n=top_n_opt, sort_by=sort_by_opt)
                    st.session_state.scan_volume_result = result_stocks
                except Exception as scan_err:
                    st.session_state.scan_volume_error = str(scan_err)

        if st.session_state.get("scan_volume_error"):
            st.error(f"Scan error: {st.session_state.scan_volume_error}")
        elif st.session_state.get("scan_volume_result"):
            vol_stocks = st.session_state.scan_volume_result
            st.caption(f"Showing top {len(vol_stocks)} stocks ranked by {sort_by_opt}.")
            rows = [s.to_dict() for s in vol_stocks]
            df_vol = pd.DataFrame(rows)
            # Format for display
            display_df = pd.DataFrame({
                "Ticker":     df_vol["ticker"],
                "Price $":    df_vol["price"].map(lambda x: f"{x:.2f}" if x else "n/a"),
                "Day Chg %":  df_vol["day_change_pct"].map(lambda x: f"{x:+.2f}%" if x else "n/a"),
                "Volume":     df_vol["volume"].map(lambda x: f"{int(x):,}" if x else "n/a"),
                "Vol/Avg":    df_vol["volume_ratio"].map(lambda x: f"{x:.2f}x" if x else "n/a"),
                "Mkt Cap $B": df_vol["market_cap_b"].map(lambda x: f"{x:.0f}" if x else "n/a"),
            })
            def _chg_style(val):
                try:
                    v = float(str(val).replace('%','').replace('+',''))
                    return 'color:#0f7a4f;font-weight:700' if v >= 0 else 'color:#b42318;font-weight:700'
                except Exception:
                    return ''
            try:
                _styled_vol = display_df.style.map(_chg_style, subset=["Day Chg %"])
                st.dataframe(_styled_vol, use_container_width=True, hide_index=True, height=420)
            except Exception:
                st.dataframe(display_df, use_container_width=True, hide_index=True, height=420)

            # Quick Analyze shortcut
            st.markdown("**Quick Analyze a stock from this list:**")
            quick_ticker = st.selectbox(
                "Select ticker",
                options=[s.ticker for s in vol_stocks],
                key="scan_quick_ticker_vol",
            )
            if st.button("🔍 Analyze in Live Analysis", key="scan_quick_analyze_vol", use_container_width=True):
                # Store ticker so the Live Analysis input pre-populates when user switches tab
                st.session_state["live_ticker"] = quick_ticker
                st.toast(f"✅ Ticker set to {quick_ticker} — switch to 🔴 Live Analysis and click Analyze.", icon="🔍")
            if st.button("💾 Save to Tickers", key="scan_save_vol", use_container_width=True):
                sel_snaps = [s for s in vol_stocks if s.ticker == quick_ticker]
                if sel_snaps:
                    _snap_packet = _create_scan_packet(sel_snaps[0])
                    # Always save to session packets so it appears in the signals DataFrame
                    _pkts = st.session_state.get("scan_session_packets", [])
                    _pkts = [p for p in _pkts if p["ticker"] != quick_ticker]
                    st.session_state.scan_session_packets = _pkts + [_snap_packet]
                    # Also register in saved_scan_tickers so it appears in the sidebar dropdown
                    _saved_tickers = st.session_state.get("saved_scan_tickers", [])
                    if quick_ticker not in _saved_tickers:
                        _saved_tickers = _saved_tickers + [quick_ticker]
                        st.session_state.saved_scan_tickers = _saved_tickers
                    # Optionally persist to disk if session_only is OFF
                    if not st.session_state.get("settings_session_only", True):
                        _append_to_signals(SIGNALS_PATH, _snap_packet)
                        load_demo_data.clear()
                    st.toast(f"✅ {quick_ticker} saved to ticker list. It will appear in the sidebar Ticker dropdown.", icon="💾")
        else:
            st.info("Click **Run Volume/Price Scan** to fetch data for the full NASDAQ-100 + S&P 500 universe.")

    # ── Section B: News-Driven Discovery ──────────────────────────────────
    with scan_col_b:
        st.markdown("##### 📰 News-Driven Stock Discovery")
        st.caption(
            "Scans Yahoo Finance, MarketWatch, Reuters, Benzinga, CNBC, and Google News RSS feeds. "
            "Extracts all ticker mentions and ranks by article frequency."
        )
        news_n_opt = st.slider("Number of tickers to return", min_value=10, max_value=50, value=25, step=5, key="scan_news_n")

        if st.button("📡 Discover from News", type="primary", use_container_width=True, key="run_news_scan"):
            st.session_state.scan_news_result = None
            st.session_state.scan_news_error = None
            with st.spinner("Scanning live RSS feeds for ticker mentions…"):
                try:
                    from src.finance_news_analyzer.stock_screener import discover_stocks_from_news
                    news_stocks = discover_stocks_from_news(n=news_n_opt)
                    st.session_state.scan_news_result = news_stocks
                except Exception as ne:
                    st.session_state.scan_news_error = str(ne)

        if "scan_news_result" not in st.session_state:
            st.session_state.scan_news_result = None
        if st.session_state.get("scan_news_error"):
            st.error(f"Discovery error: {st.session_state.scan_news_error}")
        elif st.session_state.get("scan_news_result") is not None:
            news_stocks = st.session_state.scan_news_result
            if not news_stocks:
                st.info(
                    "No tickers found yet — click **Discover from News** to scan recent financial headlines. "
                    "Results include stocks from the past 7 days of news."
                )
            else:
                st.caption(f"{len(news_stocks)} tickers found in recent news. Most-mentioned first.")

                # Chart: top 15 by mentions — use go.Bar directly (avoids all px.bar issues)
                chart_stocks = [s for s in news_stocks[:15] if s.news_mentions > 0]
                if chart_stocks:
                    _tickers  = [s.ticker for s in chart_stocks]
                    _mentions = [s.news_mentions for s in chart_stocks]
                    _chgs     = [s.day_change_pct * 100 if s.day_change_pct else 0.0 for s in chart_stocks]
                    _colors   = ["#0f7a4f" if v > 0 else ("#b42318" if v < 0 else "#667085") for v in _chgs]
                    fig_news = go.Figure(go.Bar(
                        x=_tickers,
                        y=_mentions,
                        marker_color=_colors,
                        text=[f"{v:+.3f}%" if v else "" for v in _chgs],
                        textposition="outside",
                        hovertemplate="<b>%{x}</b><br>Mentions: %{y}<extra></extra>",
                    ))
                    fig_news.update_layout(
                        height=220,
                        margin=dict(l=4, r=4, t=8, b=4),
                        yaxis_title="Mentions",
                        xaxis_title="",
                        showlegend=False,
                    )
                    st.plotly_chart(fig_news, use_container_width=True)

                # Table
                news_rows = [
                    {
                        "Ticker":    s.ticker,
                        "Mentions":  s.news_mentions,
                        "Price $":   f"{s.price:.2f}" if s.price else "n/a",
                        "Day Chg %": f"{s.day_change_pct*100:+.2f}%" if s.day_change_pct else "n/a",
                        "Sources":   ", ".join(s.mention_sources[:3]),
                    }
                    for s in news_stocks
                ]
                _news_df = pd.DataFrame(news_rows)
                def _chg_style_news(val):
                    try:
                        v = float(str(val).replace('%','').replace('+',''))
                        return 'color:#0f7a4f;font-weight:700' if v >= 0 else 'color:#b42318;font-weight:700'
                    except Exception:
                        return ''
                try:
                    _news_styled = _news_df.style.map(_chg_style_news, subset=["Day Chg %"])
                    st.dataframe(_news_styled, use_container_width=True, hide_index=True, height=280)
                except Exception:
                    st.dataframe(_news_df, use_container_width=True, hide_index=True, height=280)

                # Quick Analyze + Save (only shown when there are results)
                st.markdown("**Select a discovered stock:**")
                quick_news_ticker = st.selectbox(
                    "Select ticker",
                    options=[s.ticker for s in news_stocks],
                    key="scan_quick_ticker_news",
                )
                btn_n1, btn_n2 = st.columns(2, gap="small")
                with btn_n1:
                    if st.button("🔍 Analyze", key="scan_quick_analyze_news", use_container_width=True):
                        # Store ticker so the Live Analysis input pre-populates when user switches tab
                        st.session_state["live_ticker"] = quick_news_ticker
                        st.toast(f"✅ Ticker set to {quick_news_ticker} — switch to 🔴 Live Analysis and click Analyze.", icon="🔍")
                with btn_n2:
                    if st.button("💾 Save to Tickers", key="scan_save_news", use_container_width=True):
                        sel_news_snaps = [s for s in news_stocks if s.ticker == quick_news_ticker]
                        if sel_news_snaps:
                            _snap_packet = _create_scan_packet(sel_news_snaps[0])
                            # Always save to session packets so it appears in the signals DataFrame
                            _pkts = st.session_state.get("scan_session_packets", [])
                            _pkts = [p for p in _pkts if p["ticker"] != quick_news_ticker]
                            st.session_state.scan_session_packets = _pkts + [_snap_packet]
                            # Also register in saved_scan_tickers so it appears in the sidebar dropdown
                            _saved_tickers_n = st.session_state.get("saved_scan_tickers", [])
                            if quick_news_ticker not in _saved_tickers_n:
                                _saved_tickers_n = _saved_tickers_n + [quick_news_ticker]
                                st.session_state.saved_scan_tickers = _saved_tickers_n
                            # Optionally persist to disk if session_only is OFF
                            if not st.session_state.get("settings_session_only", True):
                                _append_to_signals(SIGNALS_PATH, _snap_packet)
                                load_demo_data.clear()
                            st.toast(f"✅ {quick_news_ticker} saved to ticker list. It will appear in the sidebar Ticker dropdown.", icon="💾")
        else:
            st.info(
                "Click **Discover from News** to scan recent financial headlines "
                "(past 7 days) for the most-discussed stocks."
            )

with tab_monitor:
    # ── Index strip ───────────────────────────────────────────────────────────
    idx = _get_index_snapshot()
    def _idx_chg(sym):
        v = idx.get(sym)
        if v is None: return "n/a"
        colour = "#0f7a4f" if v >= 0 else "#b42318"
        sign = "+" if v >= 0 else ""
        return f'<span style="color:{colour};font-weight:700">{sign}{v:.3f}%</span>'

    st.markdown(
        f"""
        <div style="display:flex;gap:1.2rem;flex-wrap:wrap;margin-bottom:0.6rem;padding:0.5rem 0.7rem;
                    background:var(--panel);border:1px solid var(--line);border-radius:8px;">
          <span style="color:var(--muted);font-size:0.72rem;align-self:center">Market Indices:</span>
          <span style="font-size:0.84rem;color:var(--ink)"><b>S&amp;P 500 (SPY)</b> {_idx_chg("SPY")}</span>
          <span style="font-size:0.84rem;color:var(--ink)"><b>NASDAQ (QQQ)</b> {_idx_chg("QQQ")}</span>
          <span style="font-size:0.84rem;color:var(--ink)"><b>Dow Jones (DIA)</b> {_idx_chg("DIA")}</span>
          <span style="font-size:0.84rem;color:var(--ink)"><b>Russell 2000 (IWM)</b> {_idx_chg("IWM")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sector impact from current signals ────────────────────────────────────
    sector_df = _core_signals[["ticker", "sector", "direction", "confidence", "sentiment_score"]].copy()
    if "sector" in sector_df.columns and not sector_df["sector"].isna().all():
        sector_agg = (
            sector_df.groupby("sector")
            .agg(
                signals=("ticker", "count"),
                avg_sentiment=("sentiment_score", "mean"),
                avg_confidence=("confidence", "mean"),
                tickers=("ticker", lambda x: ", ".join(x.tolist())),
            )
            .reset_index()
            .sort_values("avg_sentiment", ascending=False)
        )
        st.markdown(
            "<div class='section-note' style='margin-top:0.3rem'>Sector impact from current research packets:</div>",
            unsafe_allow_html=True,
        )
        sec_disp = sector_agg.copy()
        sec_disp["avg_sentiment"] = sec_disp["avg_sentiment"].map(lambda x: f"{x:+.3f}")
        sec_disp["avg_confidence"] = sec_disp["avg_confidence"].map(lambda x: f"{x:.0%}")
        sec_disp.columns = ["Sector", "Signals", "Avg Sentiment", "Avg Confidence", "Representative Stocks"]
        st.dataframe(sec_disp, use_container_width=True, hide_index=True)

    left, right = st.columns([1.25, 0.75], gap="medium")
    with left:
        st.markdown("#### Signal Queue")
        st.markdown(
            "<div class='section-note'>Compact ranking view for generated ideas, confidence, novelty, and disagreement checks.</div>",
            unsafe_allow_html=True,
        )
        if filtered.empty:
            _hz_note = f" with **{horizon_filter}** horizon" if horizon_filter != "All" else ""
            _tk_note = f" for **{ticker_filter}**" if ticker_filter != "All" else ""
            st.info(
                f"No signals found{_tk_note}{_hz_note}. "
                "Demo signals use a 5-day horizon — select **5d** or **All** in the sidebar, "
                "or run **Live Analysis** and save results to see them here."
            )
        for signal in filtered.sort_values(["confidence", "novelty_score"], ascending=False).to_dict("records"):
            snapshot = signal.get("market_snapshot", {}) or {}
            st.markdown(
                f"""
                <div class="signal-card">
                  <div class="signal-head">
                    <div>
                      <div class="ticker">{signal["ticker"]} {render_pill(signal["direction"])}</div>
                      <div class="company">{signal["company"]} | {signal.get("sector", "n/a")} | {signal.get("event_type", "n/a")}</div>
                    </div>
                    <div class="company">{signal["horizon_days"]} trading days</div>
                  </div>
                  <div class="analysis-box">{signal["reasoning"]}</div>
                  <div class="compact-grid">
                    <div class="mini-stat"><div class="mini-label">Confidence</div><div class="mini-value">{pct(signal["confidence"], 0)}</div></div>
                    <div class="mini-stat"><div class="mini-label">Novelty</div><div class="mini-value">{pct(signal.get("novelty_score"), 0)}</div></div>
                    <div class="mini-stat"><div class="mini-label">Rel. strength</div><div class="mini-value">{pct(snapshot.get("relative_strength"), 1)}</div></div>
                  </div>
                  <div class="company">Catalyst: {signal["catalyst"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("#### Market Pulse")
        pulse = _core_signals[["ticker", "direction", "confidence", "sentiment_score", "source_quality"]].copy()
        pulse["conviction"] = pulse["confidence"] * pulse["source_quality"]
        fig = px.scatter(
            pulse,
            x="sentiment_score",
            y="confidence",
            size="conviction",
            color="direction",
            text="ticker",
            color_discrete_map={"Bullish": "#0f7a4f", "Bearish": "#b42318", "Neutral": "#667085"},
            labels={"sentiment_score": "News sentiment", "confidence": "Model confidence"},
            range_x=[-1, 1],
            range_y=[0, 1],
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(height=310, margin=dict(l=10, r=10, t=20, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Signal Confidence Audit")
        st.caption(
            "Shows signals where the AI/RAG model direction differs from "
            "raw news sentiment — a divergence flag highlights where model "
            "conviction goes against the news tone (contrarian or overriding signals). "
            "Only current tickers are shown."
        )
        # Filter divergence to only tickers currently tracked (no stale demo tickers)
        _current_tkrs = set(_core_tickers_extended)
        _core_signals_for_div = _core_signals[_core_signals["ticker"].isin(_current_tkrs)].copy()
        divergence = _core_signals_for_div[
            _core_signals_for_div["direction"] != _core_signals_for_div["baseline_sentiment"]
        ]
        if divergence.empty:
            st.caption("✅ No divergence — RAG direction aligns with raw news sentiment for all tracked tickers.")
        else:
            for row in divergence.to_dict("records"):
                _div_reason = (
                    "Technical signals override mixed news sentiment."
                    if "technical" in row.get("reasoning", "").lower()
                    else "Model conviction differs from raw news tone — review evidence."
                )
                st.markdown(
                    f"""
                    <div class="source-card">
                      <div class="source-title">{row["ticker"]}: Model {render_pill(row["direction"])} vs raw sentiment {render_pill(row["baseline_sentiment"])}</div>
                      <div class="source-meta">{row.get("catalyst", "")[:120]}</div>
                      <div style="font-size:0.79rem;color:var(--muted)">{_div_reason}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ── AI Refresh section ─────────────────────────────────────────────
        _ai_key_mon = st.session_state.get("settings_openai_key", "").strip()
        if _ai_key_mon:
            st.divider()
            st.markdown("#### 🔄 Refresh Signals with AI")
            st.caption("OpenAI key detected — click to re-run the RAG pipeline for the tickers shown in the Signal Queue using GPT-4o-mini.")
            # Only refresh tickers that appear in the Signal Queue (core + saved scan tickers)
            _refresh_tickers_mon = [t for t in _core_tickers_extended if t in signals["ticker"].values]
            if _refresh_tickers_mon and st.button(
                f"🔄 Refresh {len(_refresh_tickers_mon)} signal(s) with AI",
                key="mon_ai_refresh", use_container_width=True
            ):
                run_mon = _try_import_runner()
                if run_mon:
                    _mon_errors = []
                    for _mt in _refresh_tickers_mon:
                        with st.spinner(f"⏳ Analyzing {_mt}…"):
                            try:
                                _pkt_m = run_mon(ticker=_mt, openai_api_key=_ai_key_mon, top_k=8)
                                _pkt_m.pop("_pipeline_meta", None)
                                _ex_m = json.loads(SIGNALS_PATH.read_text(encoding="utf-8"))
                                # Replace ONLY same-ticker + same-horizon entries.
                                # This preserves manually-added 20d signals when refreshing 5d.
                                _new_hz = _pkt_m.get("horizon_days", 5)
                                _ex_m = [s for s in _ex_m if not (
                                    s.get("ticker") == _pkt_m.get("ticker") and
                                    s.get("horizon_days") == _new_hz and
                                    "scan" not in s.get("id", "")
                                )]
                                _ex_m.append(_pkt_m)
                                SIGNALS_PATH.write_text(json.dumps(_ex_m, indent=2, default=str), encoding="utf-8")
                            except Exception as _me:
                                _mon_errors.append(f"{_mt}: {_me}")
                    load_demo_data.clear()
                    if _mon_errors:
                        st.warning("Some refreshes failed: " + "; ".join(_mon_errors))
                    else:
                        st.success(f"Refreshed {len(_refresh_tickers_mon)} signal(s) with AI.")
                    # Rerun so the Signal Queue immediately shows the updated signals
                    st.rerun()

with tab_evidence:
    left, right = st.columns([1.05, 0.95], gap="medium")
    with left:
        st.markdown("#### Source Audit")
        st.markdown(
            "<div class='section-note'>Broad market intelligence across key sectors and asset classes — "
            "semiconductors, technology, energy, precious metals, and macro themes.</div>",
            unsafe_allow_html=True,
        )

        # ── Broad market source cards ─────────────────────────────────────
        _MARKET_SOURCES = [
            {
                "topic": "Semiconductors",
                "icon": "🔬",
                "source": "Reuters / Bloomberg",
                "sentiment": "Bullish",
                "credibility": 0.88,
                "title": "AI chip demand drives semiconductor supercycle — analysts raise targets on NVDA, AMD, AVGO",
                "excerpt": (
                    "Wall Street analysts broadly raised price targets on leading semiconductor names after "
                    "blowout data-center revenue figures. NVIDIA's H100/H200 allocation remains constrained "
                    "through 2025, while AMD's MI300X ramp is accelerating faster than expected. "
                    "TSMC capacity expansions in Arizona and Japan are seen as structural tailwinds for the sector."
                ),
            },
            {
                "topic": "Technology (Broad)",
                "icon": "💻",
                "source": "CNBC / Yahoo Finance",
                "sentiment": "Bullish",
                "credibility": 0.82,
                "title": "Mega-cap tech earnings beat; AI capex cycle intact — MSFT, GOOGL, META lift S&P 500",
                "excerpt": (
                    "Microsoft, Alphabet, and Meta all reported stronger-than-expected earnings driven by "
                    "cloud and AI monetization. Microsoft Azure grew 31% YoY; Google Cloud crossed $12B quarterly. "
                    "Analysts debate whether elevated capex (>$200B combined in FY25) is value-creating or dilutive, "
                    "but near-term earnings beats have pushed sector multiples higher."
                ),
            },
            {
                "topic": "Gold & Precious Metals",
                "icon": "🥇",
                "source": "MarketWatch / FT",
                "sentiment": "Bullish",
                "credibility": 0.85,
                "title": "Gold tests all-time highs above $2,400 as central banks accelerate reserve diversification",
                "excerpt": (
                    "Gold spot prices surged past $2,400/oz amid continued central-bank buying from China, "
                    "India, and emerging markets. Fed rate-cut expectations and a softer dollar provided additional "
                    "support. Silver and platinum followed, with the gold/silver ratio compressing. "
                    "ETF inflows into GLD and IAU accelerated for the third consecutive month."
                ),
            },
            {
                "topic": "Energy & Oil",
                "icon": "⛽",
                "source": "Bloomberg / EIA",
                "sentiment": "Neutral",
                "credibility": 0.80,
                "title": "Crude oil range-bound between $78–$85 as OPEC+ cuts offset demand slowdown fears",
                "excerpt": (
                    "WTI crude consolidated near $82/bbl as OPEC+ maintained voluntary cuts through Q3. "
                    "U.S. shale output hit a new record, capping upside. IEA revised 2025 demand growth lower "
                    "on China's slowing industrial activity. Natural gas prices remain depressed in North America "
                    "despite higher European LNG premiums."
                ),
            },
            {
                "topic": "Macro & Rates",
                "icon": "🏦",
                "source": "WSJ / Fed Watch",
                "sentiment": "Neutral",
                "credibility": 0.90,
                "title": "Fed holds rates steady; markets price two cuts in H2 2025 — Treasury curve steepens",
                "excerpt": (
                    "The Federal Reserve kept the fed funds rate at 5.25–5.50% for the fourth consecutive meeting. "
                    "Fed Chair Powell signaled patience, citing sticky services inflation. Markets now price "
                    "~48bps of cuts by year-end. The 10-year Treasury yield rose to 4.62%, steepening the curve "
                    "and pressuring long-duration growth equities."
                ),
            },
            {
                "topic": "Electric Vehicles & Clean Energy",
                "icon": "⚡",
                "source": "Electrek / Reuters",
                "sentiment": "Bearish",
                "credibility": 0.75,
                "title": "EV demand softens; automakers cut production targets as inventory piles up",
                "excerpt": (
                    "Tesla, GM, and Ford revised EV production targets downward amid softer consumer demand "
                    "and elevated inventory levels at dealerships. Charging infrastructure buildout lags expectations. "
                    "Lithium carbonate prices dropped 30% YTD, pressuring battery suppliers. However, "
                    "policy tailwinds from the IRA continue to support longer-term sector investment."
                ),
            },
            {
                "topic": "Financials & Banking",
                "icon": "🏛️",
                "source": "Bloomberg / CNBC",
                "sentiment": "Bullish",
                "credibility": 0.83,
                "title": "Big banks report record trading revenue; net interest margins stabilizing",
                "excerpt": (
                    "JPMorgan, Goldman Sachs, and Morgan Stanley each reported stronger-than-expected Q1 results, "
                    "with equities and FICC trading desks posting multi-year highs. Net interest income is "
                    "stabilizing as deposit repricing moderates. Loan growth remains tepid but credit quality "
                    "held up better than feared. Regional banks face continued commercial real estate headwinds."
                ),
            },
            {
                "topic": "Consumer & Retail",
                "icon": "🛒",
                "source": "WSJ / Retail Dive",
                "sentiment": "Neutral",
                "credibility": 0.74,
                "title": "Consumer spending resilient at upper income; lower-income cohort shows stress",
                "excerpt": (
                    "U.S. retail sales surprised to the upside in April, led by home improvement and electronics. "
                    "However, discount retailers like Dollar General and Dollar Tree flagged increasing financial "
                    "stress among lower-income shoppers. Credit card delinquencies rose to a 10-year high. "
                    "Luxury goods remain robust, with LVMH and Hermès reporting continued strong demand from wealthy consumers."
                ),
            },
        ]

        # ── Topic filter ──────────────────────────────────────────────────
        _all_topics = [s["topic"] for s in _MARKET_SOURCES]
        _selected_topics = st.multiselect(
            "Filter by sector / theme",
            options=_all_topics,
            default=_all_topics,
            key="ev_topic_filter",
        )
        _sentiment_filter_ev = st.multiselect(
            "Filter by sentiment",
            options=["Bullish", "Neutral", "Bearish"],
            default=["Bullish", "Neutral", "Bearish"],
            key="ev_sentiment_filter",
        )

        _filtered_sources = [
            s for s in _MARKET_SOURCES
            if s["topic"] in _selected_topics and s["sentiment"] in _sentiment_filter_ev
        ]

        st.markdown(
            f"<div class='section-note'>{len(_filtered_sources)} source(s) matching filters.</div>",
            unsafe_allow_html=True,
        )

        # Representative ETF/ticker per theme for AI analysis
        _THEME_TICKERS = {
            "Semiconductors": "SOXX",
            "Technology (Broad)": "QQQ",
            "Gold & Precious Metals": "GLD",
            "Energy & Oil": "XLE",
            "Macro & Rates": "TLT",
            "Electric Vehicles & Clean Energy": "ICLN",
            "Financials & Banking": "XLF",
            "Consumer & Retail": "XLY",
        }

        for _ms in _filtered_sources:
            _pill_cls = {"Bullish": "pill-bullish", "Bearish": "pill-bearish", "Neutral": "pill-neutral"}.get(_ms["sentiment"], "pill-neutral")
            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-title">
                    {_ms["icon"]} {_ms["topic"]}
                    &nbsp;<span class="pill {_pill_cls}">{_ms["sentiment"]}</span>
                  </div>
                  <div class="source-meta">
                    {_ms["source"]} &nbsp;|&nbsp; credibility {int(_ms["credibility"] * 100)}%
                  </div>
                  <div style="font-size:0.8rem;font-weight:600;margin-bottom:0.2rem">{_ms["title"]}</div>
                  <div style="font-size:0.79rem;color:var(--muted)">{_ms["excerpt"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            # AI Analyze button per card
            _topic_key = _ms["topic"].replace(" ", "_").replace("&", "and").replace("/", "_")
            _rep_ticker = _THEME_TICKERS.get(_ms["topic"], "SPY")
            if st.button(
                f"🤖 Analyze {_ms['topic']} ({_rep_ticker})",
                key=f"ev_analyze_{_topic_key}",
                use_container_width=True,
            ):
                st.session_state["ev_selected_topic"] = _ms["topic"]
                st.session_state["ev_selected_ticker"] = _rep_ticker
                _run_ev = _try_import_runner()
                if _run_ev:
                    _ev_ai_k = st.session_state.get("settings_openai_key", "").strip()
                    with st.spinner(f"⏳ Running AI analysis for **{_ms['topic']}** ({_rep_ticker})…"):
                        try:
                            _ev_p = _run_ev(
                                ticker=_rep_ticker,
                                openai_api_key=_ev_ai_k or None,
                                top_k=8,
                            )
                            _ev_p.pop("_pipeline_meta", None)
                            st.session_state["ev_topic_result"] = _ev_p
                            st.session_state["ev_topic_error"] = None
                        except Exception as _ev_e:
                            st.session_state["ev_topic_result"] = None
                            st.session_state["ev_topic_error"] = str(_ev_e)

        # ── Credibility bar chart for all sources ─────────────────────────
        if _filtered_sources:
            _cred_df = pd.DataFrame([
                {"Topic": s["topic"], "Credibility": s["credibility"], "Sentiment": s["sentiment"]}
                for s in _filtered_sources
            ])
            _cred_color_map = {"Bullish": "#0f7a4f", "Bearish": "#b42318", "Neutral": "#667085"}
            fig_cred = px.bar(
                _cred_df,
                x="Credibility",
                y="Topic",
                color="Sentiment",
                orientation="h",
                range_x=[0, 1],
                color_discrete_map=_cred_color_map,
                labels={"Credibility": "Source credibility", "Topic": ""},
            )
            fig_cred.update_layout(
                height=max(200, len(_filtered_sources) * 38),
                margin=dict(l=8, r=8, t=10, b=8),
                legend_title_text="Sentiment",
            )
            st.plotly_chart(fig_cred, use_container_width=True)
    with right:
        # ── Per-topic AI analysis results ─────────────────────────────────────
        _ev_topic_res = st.session_state.get("ev_topic_result")
        _ev_topic_err = st.session_state.get("ev_topic_error")
        _ev_sel_topic = st.session_state.get("ev_selected_topic")
        _ev_sel_ticker = st.session_state.get("ev_selected_ticker")

        if _ev_sel_topic:
            st.markdown(f"#### 🤖 AI Analysis: {_ev_sel_topic}")
            if _ev_topic_err:
                st.error(f"Analysis error: {_ev_topic_err}")
            elif _ev_topic_res:
                _t_dir = _ev_topic_res.get("direction", "Neutral")
                _t_conf = _ev_topic_res.get("confidence", 0.0)
                st.markdown(
                    f"""
                    <div class="signal-card">
                      <div class="signal-head">
                        <div>
                          <div class="ticker">{_ev_sel_ticker} {render_pill(_t_dir)}</div>
                          <div class="company">{_ev_topic_res.get("company","")}</div>
                        </div>
                        <div class="company">{pct(_t_conf, 0)} confidence</div>
                      </div>
                      <div class="analysis-box">{_ev_topic_res.get("reasoning","")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                _t_cits = _ev_topic_res.get("citations", [])
                if _t_cits:
                    st.markdown("**Evidence sources:**")
                    for _tc in _t_cits[:4]:
                        st.markdown(
                            f'<div class="source-card"><div class="source-title">{_tc.get("source","")}</div>'
                            f'<div class="source-meta">{_tc.get("title","")} | credibility {pct(float(_tc.get("credibility_weight",0)), 0)}</div>'
                            f'<div style="font-size:0.79rem;color:var(--muted)">{_tc.get("excerpt","")}</div></div>',
                            unsafe_allow_html=True,
                        )
                _t_trace = _ev_topic_res.get("agent_trace", [])
                if _t_trace:
                    st.markdown("**Agent trace:**")
                    for _ti, _ts in enumerate(_t_trace, 1):
                        st.markdown(
                            f'<div class="source-card"><div class="source-title">{_ti}. {_ts.get("agent","")}</div>'
                            f'<div style="font-size:0.79rem">{_ts.get("summary","")}</div></div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.info(f"Running analysis for **{_ev_sel_topic}** ({_ev_sel_ticker})…")
        else:
            st.markdown("#### 🤖 AI Theme Analysis")
            st.info("Click **🤖 Analyze** on any source card on the left to run a live AI analysis for that sector or theme.")

        st.divider()

        st.markdown("#### Agent Path")
        for idx, step in enumerate(selected["agent_trace"], start=1):
            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-title">{idx}. {step["agent"]}</div>
                  <div>{step["summary"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("#### Topic Lens")
        lens = pd.DataFrame(
            [
                {"lens": "Sentiment", "score": abs(selected.get("sentiment_score", 0))},
                {"lens": "Novelty", "score": selected.get("novelty_score", 0)},
                {"lens": "Source quality", "score": selected.get("source_quality", 0)},
                {"lens": "Confidence", "score": selected.get("confidence", 0)},
            ]
        )
        fig = px.line_polar(lens, r="score", theta="lens", line_close=True, range_r=[0, 1])
        fig.update_traces(fill="toself", line_color="#126c83")
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab_evaluation:
    st.markdown("#### Backtest Diagnostics")

    @st.cache_data(ttl=3600)
    def _compute_real_eval(signal_rows: tuple) -> pd.DataFrame:
        """Compute live forward returns from yfinance when enough future data exists."""
        import yfinance as yf
        from datetime import datetime, timezone

        rows = []
        for tkr, direction, pub_at, horizon, confidence in signal_rows:
            try:
                try:
                    sig_date = datetime.fromisoformat(str(pub_at)).date()
                except Exception:
                    sig_date = datetime.now(timezone.utc).date()

                h = yf.Ticker(tkr).history(period="1y")
                if h.empty or len(h) < 3:
                    continue
                h.index = [d.date() if hasattr(d, "date") else d for d in h.index]
                h = h.sort_index()
                future = h[h.index >= sig_date]
                if future.empty:
                    continue

                entry = float(future["Close"].iloc[0])
                row = {
                    "ticker": tkr,
                    "direction": direction,
                    "confidence": confidence,
                    "horizon_days": horizon,
                }
                for hz in (5, 20):
                    ret_col = f"ret{hz}d"
                    hit_col = f"hit{hz}d"
                    signed_col = f"signed{hz}d"
                    if len(future) >= hz + 1:
                        exit_price = float(future["Close"].iloc[hz])
                        ret = (exit_price - entry) / entry if entry else 0.0
                        hit = (
                            (direction == "Bullish" and ret > 0)
                            or (direction == "Bearish" and ret < 0)
                            or (direction == "Neutral" and abs(ret) < 0.01)
                        )
                        row[ret_col] = ret
                        row[hit_col] = hit
                        row[signed_col] = ret if hit else -ret
                    else:
                        row[ret_col] = None
                        row[hit_col] = None
                        row[signed_col] = None
                rows.append(row)
            except Exception:
                continue
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    _eval_signals = signals[~signals["id"].str.contains("scan", na=False)].copy()
    _eval_signals = _eval_signals.sort_values("confidence", ascending=False).drop_duplicates(subset=["ticker"], keep="first")
    _sig_rows_eval = tuple(
        (r["ticker"], r["direction"], str(r.get("published_at", "")), int(r.get("horizon_days", 5)), float(r.get("confidence", 0.5)))
        for _, r in _eval_signals.iterrows()
    )

    with st.spinner("Computing forward-return evaluation..."):
        _real_eval = _compute_real_eval(_sig_rows_eval)

    _live_has_returns = (
        not _real_eval.empty
        and (
            ("ret5d" in _real_eval.columns and _real_eval["ret5d"].notna().any())
            or ("ret20d" in _real_eval.columns and _real_eval["ret20d"].notna().any())
        )
    )

    def _live_metric_rows(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for hz in (5, 20):
            ret_col = f"ret{hz}d"
            hit_col = f"hit{hz}d"
            signed_col = f"signed{hz}d"
            valid = df[df[ret_col].notna()] if ret_col in df.columns else pd.DataFrame()
            coverage = len(valid) / len(df) if len(df) else 0.0
            hit_rate = float(valid[hit_col].mean()) if len(valid) else 0.0
            avg_signed = float(valid[signed_col].mean()) if len(valid) else 0.0
            avg_raw = float(valid[ret_col].mean()) if len(valid) else 0.0
            rows.append({
                "method": "Multi-Agent RAG",
                "horizon": f"{hz}d",
                "signals": len(valid),
                "hit_rate": hit_rate,
                "avg_signed_return": avg_signed,
                "avg_raw_return": avg_raw,
                "signal_coverage": coverage,
            })
        return pd.DataFrame(rows)

    if _live_has_returns:
        _chart_metrics = _live_metric_rows(_real_eval)
        _eval_source_note = "Live signal evaluation using available realized yfinance forward returns."
        _outcome_rows = pd.DataFrame()
        _n = len(_real_eval)
        _avg_confidence = _eval_signals["confidence"].mean() if len(_eval_signals) else 0.0
    else:
        _chart_metrics = build_dashboard_metrics(eval_sample_evaluated)
        _eval_source_note = (
            "Current live signals are too recent for realized 5d/20d outcomes, so this panel uses "
            "the bundled historical demo evaluation sample aligned with demo_data/prices.csv."
        )
        _outcome_rows = outcome_display_frame(eval_sample_evaluated)
        _n = int(eval_sample_evaluated["id"].nunique()) if "id" in eval_sample_evaluated.columns else len(eval_sample_evaluated)
        _avg_confidence = eval_sample_evaluated["confidence"].mean() if len(eval_sample_evaluated) else 0.0

    def _metric_value(method: str, horizon: str, col: str, default=0.0):
        row = _chart_metrics[(_chart_metrics["method"] == method) & (_chart_metrics["horizon"] == horizon)]
        return row[col].iloc[0] if len(row) else default

    _hr5 = float(_metric_value("Multi-Agent RAG", "5d", "hit_rate"))
    _hr20 = float(_metric_value("Multi-Agent RAG", "20d", "hit_rate"))

    st.caption(_eval_source_note)

    mc = st.columns(4)
    mc[0].metric("RAG 5d hit rate", pct(_hr5, 0), help="Directional accuracy against realized forward returns in the displayed evaluation source.")
    mc[1].metric("RAG 20d hit rate", pct(_hr20, 0))
    mc[2].metric("Avg confidence", pct(_avg_confidence, 0))
    mc[3].metric("Signals evaluated", _n)

    fig_eval = px.bar(
        _chart_metrics,
        x="method",
        y="hit_rate",
        color="horizon",
        barmode="group",
        range_y=[0, 1],
        color_discrete_sequence=["#126c83", "#7a5c00"],
        labels={"method": "", "hit_rate": "Directional hit rate"},
    )
    fig_eval.update_layout(height=310, margin=dict(l=8, r=8, t=20, b=8), legend_title_text="")
    st.plotly_chart(fig_eval, use_container_width=True)

    _metrics_df = _chart_metrics.copy()
    _metrics_df["hit_rate"] = _metrics_df["hit_rate"].map(lambda x: pct(x, 0))
    _metrics_df["avg_signed_return"] = _metrics_df["avg_signed_return"].map(lambda x: pct(x, 2))
    _metrics_df["avg_raw_return"] = _metrics_df["avg_raw_return"].map(lambda x: pct(x, 2))
    _metrics_df["signal_coverage"] = _metrics_df["signal_coverage"].map(lambda x: pct(x, 0))
    st.dataframe(_metrics_df, use_container_width=True, hide_index=True)

    st.markdown("#### Signal-Level Outcomes")
    if not _outcome_rows.empty:
        st.dataframe(_outcome_rows, use_container_width=True, hide_index=True)
    elif not _real_eval.empty:
        _disp = _real_eval.copy()
        _disp["Confidence"] = _disp["confidence"].map(lambda x: pct(x, 0) if x is not None else "n/a")
        _disp["5d Return"] = _disp["ret5d"].map(lambda x: "" if pd.isna(x) else pct(x, 2))
        _disp["20d Return"] = _disp["ret20d"].map(lambda x: "" if pd.isna(x) else pct(x, 2))
        st.dataframe(_disp[["ticker", "direction", "Confidence", "5d Return", "20d Return"]], use_container_width=True, hide_index=True)
    else:
        st.info("No evaluation rows are available yet.")
