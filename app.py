from __future__ import annotations

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
)


ROOT = Path(__file__).resolve().parent
SIGNALS_PATH = ROOT / "demo_data" / "signals.json"
PRICES_PATH = ROOT / "demo_data" / "prices.csv"


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
        padding-top: 1.15rem;
        padding-bottom: 2rem;
        max-width: 1380px;
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
    }
    .app-shell {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: linear-gradient(180deg, #fbfcfd 0%, #ffffff 42%);
        padding: 0.95rem 1rem;
        margin-bottom: 0.8rem;
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
        font-weight: 750;
        font-size: 0.82rem;
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
    }
    .source-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.62rem 0.68rem;
        margin-bottom: 0.5rem;
        background: #fff;
    }
    .source-title {
        font-size: 0.84rem;
        font-weight: 750;
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
    return signals, prices, evaluated


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


signals, prices, evaluated = load_demo_data()
metrics = build_metric_table(evaluated)

with st.sidebar:
    st.markdown("### FinSight RAG")
    st.caption("Research console")
    tickers = ["All"] + sorted(signals["ticker"].unique())
    ticker_filter = st.selectbox("Ticker universe", tickers)
    horizon_filter = st.radio("Horizon", ["All", "5d", "20d"], horizontal=True)
    direction_filter = st.multiselect(
        "Signal",
        ["Bullish", "Bearish", "Neutral"],
        default=["Bullish", "Bearish", "Neutral"],
    )
    selected_id = st.selectbox(
        "Research packet",
        options=signals["id"],
        format_func=lambda sid: f"{signals.loc[signals['id'] == sid, 'ticker'].iloc[0]} | {sid}",
    )

filtered = signals.copy()
if ticker_filter != "All":
    filtered = filtered[filtered["ticker"] == ticker_filter]
if horizon_filter != "All":
    filtered = filtered[filtered["horizon_days"] == int(horizon_filter.replace("d", ""))]
filtered = filtered[filtered["direction"].isin(direction_filter)]
selected = signals.loc[signals["id"] == selected_id].iloc[0].to_dict()
selected_eval = evaluated[evaluated["id"] == selected_id]

avg_confidence = filtered["confidence"].mean() if len(filtered) else 0
avg_quality = filtered["source_quality"].mean() if "source_quality" in filtered and len(filtered) else 0
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

tab_monitor, tab_thesis, tab_evidence, tab_evaluation = st.tabs(
    ["Market Monitor", "Thesis Workspace", "Evidence Audit", "Evaluation Lab"]
)

with tab_monitor:
    left, right = st.columns([1.25, 0.75], gap="medium")
    with left:
        st.markdown("#### Signal Queue")
        st.markdown(
            "<div class='section-note'>Compact ranking view for generated ideas, confidence, novelty, and disagreement checks.</div>",
            unsafe_allow_html=True,
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
        pulse = signals[["ticker", "direction", "confidence", "sentiment_score", "source_quality"]].copy()
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
        st.plotly_chart(fig, width="stretch")

        st.markdown("#### Disagreement Flags")
        divergence = signals[signals["direction"] != signals["baseline_sentiment"]]
        if divergence.empty:
            st.caption("No divergence in current sample.")
        else:
            for row in divergence.to_dict("records"):
                st.markdown(
                    f"""
                    <div class="source-card">
                      <div class="source-title">{row["ticker"]}: RAG {render_pill(row["direction"])} vs sentiment {render_pill(row["baseline_sentiment"])}</div>
                      <div class="source-meta">{row["catalyst"]}</div>
                      <div>{row["reasoning"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tab_thesis:
    snapshot = selected.get("market_snapshot", {}) or {}
    top = st.columns([0.95, 1.05, 1.1], gap="medium")
    with top[0]:
        st.markdown(
            f"""
            <div class="signal-card">
              <div class="ticker">{selected["ticker"]} {render_pill(selected["direction"])}</div>
              <div class="company">{selected["company"]} | {selected.get("sector", "n/a")}</div>
              <div class="compact-grid">
                <div class="mini-stat"><div class="mini-label">Horizon</div><div class="mini-value">{selected["horizon_days"]}d</div></div>
                <div class="mini-stat"><div class="mini-label">Confidence</div><div class="mini-value">{pct(selected["confidence"], 0)}</div></div>
                <div class="mini-stat"><div class="mini-label">Source score</div><div class="mini-value">{pct(selected.get("source_quality"), 0)}</div></div>
              </div>
              <div class="analysis-box">{selected["reasoning"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top[1]:
        st.markdown("#### Quant Snapshot")
        quant_rows = [
            ("Last price", num(snapshot.get("last_price"))),
            ("Day change", pct(snapshot.get("day_change"), 1)),
            (f"{selected.get('benchmark', 'Benchmark')} change", pct(snapshot.get("benchmark_change"), 1)),
            ("Relative strength", pct(snapshot.get("relative_strength"), 1)),
            ("Volume / avg", num(snapshot.get("volume_vs_average"), "x")),
        ]
        st.dataframe(pd.DataFrame(quant_rows, columns=["Metric", "Value"]), width="stretch", hide_index=True)
        st.caption(snapshot.get("valuation_note", ""))
    with top[2]:
        ticker_prices = selected_prices(prices, selected["ticker"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=ticker_prices["date"],
                y=ticker_prices["close"],
                mode="lines+markers",
                line=dict(color="#126c83", width=2),
                marker=dict(size=5),
                name=selected["ticker"],
            )
        )
        fig.update_layout(
            height=250,
            margin=dict(l=8, r=8, t=10, b=8),
            yaxis_title="Close",
            xaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    thesis_col, risk_col, watch_col = st.columns(3, gap="medium")
    with thesis_col:
        st.markdown("#### Thesis Drivers")
        st.markdown(list_html(selected.get("thesis_bullets", [])), unsafe_allow_html=True)
    with risk_col:
        st.markdown("#### Risks and Counter-Evidence")
        combined_risk = selected.get("risk_factors", []) + selected.get("counter_evidence", [])
        st.markdown(list_html(combined_risk), unsafe_allow_html=True)
    with watch_col:
        st.markdown("#### Watch List")
        st.markdown(list_html(selected.get("watch_items", [])), unsafe_allow_html=True)

with tab_evidence:
    left, right = st.columns([1.05, 0.95], gap="medium")
    with left:
        st.markdown("#### Source Audit")
        for citation in selected["citations"]:
            st.markdown(
                f"""
                <div class="source-card">
                  <div class="source-title">{citation["source"]}</div>
                  <div class="source-meta">{citation["title"]} | credibility {pct(float(citation["credibility_weight"]), 0)}</div>
                  <div>{citation["excerpt"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        source_df = pd.DataFrame(selected["citations"])
        fig = px.bar(
            source_df,
            x="source",
            y="credibility_weight",
            color="source",
            range_y=[0, 1],
            labels={"credibility_weight": "Credibility weight", "source": ""},
        )
        fig.update_layout(height=230, margin=dict(l=8, r=8, t=10, b=8), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with right:
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
        st.plotly_chart(fig, width="stretch")

with tab_evaluation:
    st.markdown("#### Backtest Diagnostics")
    metric_cards = st.columns(4)
    rag_metrics = metrics[metrics["method"] == "Multi-Agent RAG"].copy()
    metric_cards[0].metric("RAG 5d hit rate", pct(rag_metrics[rag_metrics["horizon"] == "5d"]["hit_rate"].iloc[0], 0))
    metric_cards[1].metric("RAG 20d hit rate", pct(rag_metrics[rag_metrics["horizon"] == "20d"]["hit_rate"].iloc[0], 0))
    metric_cards[2].metric("Avg confidence", pct(signals["confidence"].mean(), 0))
    metric_cards[3].metric("Signals evaluated", len(evaluated))

    chart_metrics = metrics.copy()
    chart_metrics["horizon"] = chart_metrics["horizon"].astype(str)
    fig = px.bar(
        chart_metrics,
        x="method",
        y="hit_rate",
        color="horizon",
        barmode="group",
        range_y=[0, 1],
        color_discrete_sequence=["#126c83", "#7a5c00"],
        labels={"method": "", "hit_rate": "Directional hit rate"},
    )
    fig.update_layout(height=310, margin=dict(l=8, r=8, t=20, b=8), legend_title_text="")
    st.plotly_chart(fig, width="stretch")

    display_metrics = metrics.copy()
    display_metrics["hit_rate"] = display_metrics["hit_rate"].map(lambda x: pct(x, 0))
    display_metrics["avg_signed_return"] = display_metrics["avg_signed_return"].map(lambda x: pct(x, 2))
    display_metrics["avg_raw_return"] = display_metrics["avg_raw_return"].map(lambda x: pct(x, 2))
    st.dataframe(display_metrics, width="stretch", hide_index=True)

    st.markdown("#### Signal-Level Outcomes")
    cols = [
        "ticker",
        "direction",
        "baseline_sentiment",
        "baseline_random",
        "return_5d",
        "return_20d",
    ]
    table = evaluated[[col for col in cols if col in evaluated.columns]].copy()
    for col in ["return_5d", "return_20d"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else pct(x, 2))
    st.dataframe(table, width="stretch", hide_index=True)
