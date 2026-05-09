from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DIRECTION_SIGN = {
    "Bullish": 1,
    "Bearish": -1,
    "Neutral": 0,
}


@dataclass(frozen=True)
class MetricSummary:
    method: str
    horizon_days: int
    signals: int
    hit_rate: float
    avg_signed_return: float
    avg_raw_return: float


def load_signals(path: str | Path) -> pd.DataFrame:
    signals = pd.read_json(path)
    signals["published_at"] = pd.to_datetime(signals["published_at"], utc=True)
    return signals


def load_prices(path: str | Path) -> pd.DataFrame:
    prices = pd.read_csv(path, parse_dates=["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    return prices


def attach_forward_returns(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, signal in signals.iterrows():
        ticker_prices = prices[prices["ticker"] == signal["ticker"]].sort_values("date")
        trade_date = signal["published_at"].tz_convert(None).normalize()
        eligible = ticker_prices[ticker_prices["date"] >= trade_date]
        if eligible.empty:
            continue

        entry = eligible.iloc[0]
        result = signal.to_dict()
        result["entry_date"] = entry["date"]
        result["entry_close"] = entry["close"]

        for horizon in (5, 20):
            future = ticker_prices[ticker_prices["date"] > entry["date"]].head(horizon)
            if len(future) < horizon:
                result[f"return_{horizon}d"] = pd.NA
                result[f"exit_date_{horizon}d"] = pd.NaT
                continue
            exit_row = future.iloc[-1]
            result[f"return_{horizon}d"] = (exit_row["close"] / entry["close"]) - 1
            result[f"exit_date_{horizon}d"] = exit_row["date"]

        rows.append(result)

    return pd.DataFrame(rows)


def score_direction(direction: str, raw_return: float) -> bool:
    sign = DIRECTION_SIGN.get(direction, 0)
    if sign == 0:
        return abs(raw_return) < 0.01
    return sign * raw_return > 0


def summarize_method(evaluated: pd.DataFrame, method_column: str, horizon_days: int) -> MetricSummary:
    return_col = f"return_{horizon_days}d"
    data = evaluated.dropna(subset=[return_col]).copy()
    if data.empty:
        return MetricSummary(method_column, horizon_days, 0, 0.0, 0.0, 0.0)

    signed = data[method_column].map(DIRECTION_SIGN).fillna(0) * data[return_col]
    hits = [
        score_direction(direction, ret)
        for direction, ret in zip(data[method_column], data[return_col], strict=False)
    ]

    return MetricSummary(
        method=method_column,
        horizon_days=horizon_days,
        signals=len(data),
        hit_rate=sum(hits) / len(hits),
        avg_signed_return=float(signed.mean()),
        avg_raw_return=float(data[return_col].mean()),
    )


def build_metric_table(evaluated: pd.DataFrame) -> pd.DataFrame:
    method_map = {
        "direction": "Multi-Agent RAG",
        "baseline_sentiment": "Sentiment Baseline",
        "baseline_random": "Random Baseline",
    }
    rows = []
    for horizon in (5, 20):
        for column, label in method_map.items():
            summary = summarize_method(evaluated, column, horizon)
            rows.append(
                {
                    "method": label,
                    "horizon": f"{horizon}d",
                    "signals": summary.signals,
                    "hit_rate": summary.hit_rate,
                    "avg_signed_return": summary.avg_signed_return,
                    "avg_raw_return": summary.avg_raw_return,
                }
            )
    return pd.DataFrame(rows)

