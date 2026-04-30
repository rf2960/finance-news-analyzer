from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
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


@st.cache_data
def load_demo_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = load_signals(SIGNALS_PATH)
    prices = load_prices(PRICES_PATH)
    evaluated = attach_forward_returns(signals, prices)
    return signals, prices, evaluated


signals, prices, evaluated = load_demo_data()
metrics = build_metric_table(evaluated)

st.title("FinSight RAG")
st.caption("Evidence-grounded market signals from a multi-agent RAG pipeline")

with st.sidebar:
    st.header("Controls")
    tickers = ["All"] + sorted(signals["ticker"].unique())
    ticker_filter = st.selectbox("Ticker", tickers)
    horizon_filter = st.radio("Horizon", ["All", "5d", "20d"], horizontal=True)

filtered = signals.copy()
if ticker_filter != "All":
    filtered = filtered[filtered["ticker"] == ticker_filter]
if horizon_filter != "All":
    filtered = filtered[filtered["horizon_days"] == int(horizon_filter.replace("d", ""))]

tab_signals, tab_evidence, tab_metrics = st.tabs(
    ["Signal Desk", "Evidence Path", "Evaluation"]
)

with tab_signals:
    st.subheader("Generated Investment Ideas")
    cols = st.columns(4)
    cols[0].metric("Signals", len(filtered))
    cols[1].metric("Avg confidence", f"{filtered['confidence'].mean():.0%}" if len(filtered) else "0%")
    cols[2].metric("Bullish", int((filtered["direction"] == "Bullish").sum()))
    cols[3].metric("Bearish", int((filtered["direction"] == "Bearish").sum()))

    for signal in filtered.to_dict("records"):
        with st.container(border=True):
            top = st.columns([1.1, 1.2, 1, 1])
            top[0].markdown(f"### {signal['ticker']}")
            top[0].caption(signal["company"])
            top[1].metric("Signal", signal["direction"])
            top[2].metric("Horizon", f"{signal['horizon_days']} trading days")
            top[3].metric("Confidence", f"{signal['confidence']:.0%}")
            st.markdown(f"**Catalyst:** {signal['catalyst']}")
            st.write(signal["reasoning"])

with tab_evidence:
    st.subheader("Citations and Agent Trace")
    selected_id = st.selectbox(
        "Signal",
        options=signals["id"],
        format_func=lambda sid: f"{signals.loc[signals['id'] == sid, 'ticker'].iloc[0]} - {sid}",
    )
    selected = signals.loc[signals["id"] == selected_id].iloc[0].to_dict()

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown(f"### {selected['ticker']} {selected['direction']}")
        for citation in selected["citations"]:
            with st.container(border=True):
                st.markdown(f"**{citation['source']}**")
                st.write(citation["title"])
                st.caption(citation["excerpt"])
                st.progress(float(citation["credibility_weight"]))
    with right:
        st.markdown("### Reasoning Path")
        for idx, step in enumerate(selected["agent_trace"], start=1):
            st.markdown(f"**{idx}. {step['agent']}**")
            st.write(step["summary"])

with tab_metrics:
    st.subheader("Backtest Summary")
    display_metrics = metrics.copy()
    display_metrics["hit_rate"] = display_metrics["hit_rate"].map(lambda x: f"{x:.0%}")
    display_metrics["avg_signed_return"] = display_metrics["avg_signed_return"].map(lambda x: f"{x:.2%}")
    display_metrics["avg_raw_return"] = display_metrics["avg_raw_return"].map(lambda x: f"{x:.2%}")
    st.dataframe(display_metrics, width="stretch", hide_index=True)

    chart_metrics = metrics.copy()
    chart_metrics["horizon"] = chart_metrics["horizon"].astype(str)
    fig = px.bar(
        chart_metrics,
        x="method",
        y="hit_rate",
        color="horizon",
        barmode="group",
        range_y=[0, 1],
        labels={"method": "Method", "hit_rate": "Directional hit rate"},
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Evaluated Signals")
    cols = [
        "ticker",
        "direction",
        "baseline_sentiment",
        "baseline_random",
        "return_5d",
        "return_20d",
    ]
    available_cols = [col for col in cols if col in evaluated.columns]
    table = evaluated[available_cols].copy()
    for col in ["return_5d", "return_20d"]:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    st.dataframe(table, width="stretch", hide_index=True)
