"""
technical_factors.py
--------------------
Computes momentum and other technical factors from yfinance price history
and returns a structured dict + a formatted string suitable for injection
into LLM agent prompts.

Factors computed
----------------
Price Momentum
  • 1-week return (5 trading days)
  • 1-month return (21 trading days)
  • 3-month return (63 trading days)
  • 6-month return (126 trading days)

Trend / Moving Averages
  • SMA-20 / SMA-50 / SMA-200
  • SMA crossover signal (bullish/bearish/flat)
  • Price vs SMA-50 deviation %

Momentum Oscillators
  • RSI-14 (overbought > 70, oversold < 30)
  • MACD line, signal line, histogram (12/26/9)

Volatility
  • ATR-14 as % of price (normalised)
  • Bollinger Band width & price position (% of band)
  • Historical volatility (20-day annualised)

Volume
  • 5-day vs 20-day average volume ratio
  • On-Balance Volume trend (rising / falling)

Price extremes
  • 52-week high / low
  • % below 52-week high
  • % above 52-week low
"""
from __future__ import annotations

from typing import Any


def compute_technical_factors(ticker: str, period: str = "1y") -> dict[str, Any]:
    """
    Fetch price history and compute a comprehensive set of technical factors.

    Parameters
    ----------
    ticker : str   Ticker symbol (e.g. "NVDA").
    period : str   yfinance period string (default "1y").

    Returns
    -------
    dict with keys:
        "factors"  – flat dict of named factor values
        "summary"  – human-readable string for prompt injection
        "error"    – None on success, error message string on failure
    """
    empty = {"factors": {}, "summary": "Technical data unavailable.", "error": None}

    try:
        import numpy as np
        import pandas as pd
        import yfinance as yf
    except ImportError as e:
        empty["error"] = str(e)
        return empty

    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or hist.empty or len(hist) < 30:
            empty["error"] = "Insufficient price history."
            return empty

        close  = hist["Close"].dropna()
        volume = hist["Volume"].dropna()
        high   = hist["High"].dropna()
        low    = hist["Low"].dropna()

        n = len(close)

        # ── Price Momentum ────────────────────────────────────────────────
        def _ret(days: int) -> float | None:
            if n > days:
                return round(float((close.iloc[-1] - close.iloc[-(days+1)]) / close.iloc[-(days+1)]), 4)
            return None

        mom_5d   = _ret(5)
        mom_21d  = _ret(21)
        mom_63d  = _ret(63)
        mom_126d = _ret(126)

        # ── Moving Averages ───────────────────────────────────────────────
        sma20  = float(close.tail(20).mean())  if n >= 20  else None
        sma50  = float(close.tail(50).mean())  if n >= 50  else None
        sma200 = float(close.tail(200).mean()) if n >= 200 else None
        last   = float(close.iloc[-1])

        sma_cross = "flat"
        if sma20 and sma50:
            if sma20 > sma50 * 1.005:
                sma_cross = "bullish (20 SMA > 50 SMA)"
            elif sma20 < sma50 * 0.995:
                sma_cross = "bearish (20 SMA < 50 SMA)"

        price_vs_sma50 = round((last - sma50) / sma50, 4) if sma50 else None

        # ── RSI-14 ────────────────────────────────────────────────────────
        rsi = None
        if n >= 15:
            delta = close.diff()
            gain  = delta.clip(lower=0)
            loss  = (-delta).clip(lower=0)
            avg_g = gain.ewm(com=13, adjust=False).mean()
            avg_l = loss.ewm(com=13, adjust=False).mean()
            rs    = avg_g / avg_l.replace(0, float("nan"))
            rsi_s = 100 - (100 / (1 + rs))
            rsi   = round(float(rsi_s.iloc[-1]), 1)

        # ── MACD (12/26/9) ────────────────────────────────────────────────
        macd_line = macd_signal = macd_hist = None
        if n >= 35:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_s = ema12 - ema26
            signal_s = macd_s.ewm(span=9, adjust=False).mean()
            macd_line   = round(float(macd_s.iloc[-1]),  4)
            macd_signal = round(float(signal_s.iloc[-1]), 4)
            macd_hist   = round(macd_line - macd_signal, 4)

        # ── Bollinger Bands (20, 2σ) ──────────────────────────────────────
        bb_pos = bb_width = None
        if n >= 20:
            mid   = close.tail(20).mean()
            std   = close.tail(20).std()
            upper = mid + 2 * std
            lower = mid - 2 * std
            bb_width = round(float((upper - lower) / mid), 4)  # normalised
            bb_pos   = round(float((last - lower) / (upper - lower)), 4)  # 0=low, 1=high

        # ── ATR-14 ────────────────────────────────────────────────────────
        atr_pct = None
        if n >= 15:
            h14 = high.tail(15).reset_index(drop=True)
            l14 = low.tail(15).reset_index(drop=True)
            c14 = close.tail(15).reset_index(drop=True)
            c14_prev = c14.shift(1)
            hl = h14 - l14
            hc = (h14 - c14_prev).abs()
            lc = (l14 - c14_prev).abs()
            tr_series = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            atr_pct = round(float(tr_series.tail(14).mean() / last), 4)

        # ── Historical Volatility (20-day annualised) ─────────────────────
        hist_vol = None
        if n >= 21:
            log_ret  = np.log(close / close.shift(1)).dropna()
            hist_vol = round(float(log_ret.tail(20).std() * (252 ** 0.5)), 4)

        # ── Volume ────────────────────────────────────────────────────────
        vol_ratio = None
        if len(volume) >= 20:
            v5  = float(volume.tail(5).mean())
            v20 = float(volume.tail(20).mean())
            vol_ratio = round(v5 / v20, 2) if v20 else None

        # OBV trend (rising / falling / flat)
        obv_trend = "flat"
        if len(volume) >= 10 and len(close) >= 10:
            obv = (volume * ((close.diff() > 0).astype(int) - (close.diff() < 0).astype(int))).cumsum()
            obv5  = float(obv.tail(5).mean())
            obv20 = float(obv.tail(20).mean())
            if obv5 > obv20 * 1.02:
                obv_trend = "rising"
            elif obv5 < obv20 * 0.98:
                obv_trend = "falling"

        # ── 52-week high/low ──────────────────────────────────────────────
        w52_high = w52_low = pct_from_high = pct_from_low = None
        if n >= 252:
            window = close.tail(252)
            w52_high = round(float(window.max()), 2)
            w52_low  = round(float(window.min()), 2)
        else:
            w52_high = round(float(close.max()), 2)
            w52_low  = round(float(close.min()), 2)
        if w52_high:
            pct_from_high = round((last - w52_high) / w52_high, 4)
        if w52_low:
            pct_from_low  = round((last - w52_low)  / w52_low,  4)

        # ── Assemble factors dict ─────────────────────────────────────────
        factors = {
            # Momentum
            "mom_1w":          mom_5d,
            "mom_1m":          mom_21d,
            "mom_3m":          mom_63d,
            "mom_6m":          mom_126d,
            # Moving averages
            "sma20":           round(sma20,  2) if sma20  else None,
            "sma50":           round(sma50,  2) if sma50  else None,
            "sma200":          round(sma200, 2) if sma200 else None,
            "sma_cross":       sma_cross,
            "price_vs_sma50":  price_vs_sma50,
            # RSI / MACD
            "rsi14":           rsi,
            "macd_line":       macd_line,
            "macd_signal":     macd_signal,
            "macd_histogram":  macd_hist,
            # Bollinger
            "bb_position":     bb_pos,   # 0=at lower band, 1=at upper band
            "bb_width":        bb_width,
            # Volatility
            "atr_pct":         atr_pct,
            "hist_vol_20d":    hist_vol,
            # Volume
            "vol_5d_vs_20d":   vol_ratio,
            "obv_trend":       obv_trend,
            # Price extremes
            "w52_high":        w52_high,
            "w52_low":         w52_low,
            "pct_from_52w_high": pct_from_high,
            "pct_from_52w_low":  pct_from_low,
            "last_price":      round(last, 2),
        }

        # ── Format summary string for prompt injection ─────────────────────
        summary = _format_summary(ticker, factors)
        return {"factors": factors, "summary": summary, "error": None}

    except Exception as exc:
        return {"factors": {}, "summary": "Technical data computation failed.", "error": str(exc)}


def _pct_str(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.2%}"


def _fmt(v: float | None, decimals: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{decimals}f}"


def _format_summary(ticker: str, f: dict) -> str:
    """Return a structured multi-line string ready for LLM prompt injection."""

    # RSI interpretation — overbought threshold raised to 80 to avoid
    # mislabelling strong trending momentum as a reversal signal
    rsi = f.get("rsi14")
    if rsi is None:
        rsi_note = "n/a"
    elif rsi >= 80:
        rsi_note = f"{rsi:.1f} (strongly overbought — high reversal risk)"
    elif rsi >= 70:
        rsi_note = f"{rsi:.1f} (momentum zone — high but trend may continue)"
    elif rsi < 30:
        rsi_note = f"{rsi:.1f} (oversold — potential reversal opportunity)"
    elif rsi < 40:
        rsi_note = f"{rsi:.1f} (approaching oversold — recovering)"
    else:
        rsi_note = f"{rsi:.1f} (neutral range)"

    # MACD interpretation
    mh = f.get("macd_histogram")
    if mh is None:
        macd_note = "n/a"
    elif mh > 0:
        macd_note = f"histogram +{mh:.4f} (bullish momentum — MACD above signal)"
    else:
        macd_note = f"histogram {mh:.4f} (bearish momentum — MACD below signal)"

    # Bollinger interpretation
    bp = f.get("bb_position")
    if bp is None:
        bb_note = "n/a"
    elif bp > 0.85:
        bb_note = f"{bp:.2f} (near upper band — extended / potential mean-reversion risk)"
    elif bp < 0.15:
        bb_note = f"{bp:.2f} (near lower band — potentially oversold)"
    else:
        bb_note = f"{bp:.2f} (mid-band — neutral positioning)"

    # Momentum interpretation
    def _mom_note(v, label):
        if v is None:
            return f"  {label}: n/a"
        arrow = "▲" if v > 0 else "▼"
        return f"  {label}: {arrow} {v:+.2%}"

    lines = [
        f"=== TECHNICAL FACTORS: {ticker} ===",
        "",
        "PRICE MOMENTUM",
        _mom_note(f.get("mom_1w"),  "1-week  (5d)"),
        _mom_note(f.get("mom_1m"),  "1-month (21d)"),
        _mom_note(f.get("mom_3m"),  "3-month (63d)"),
        _mom_note(f.get("mom_6m"),  "6-month (126d)"),
        "",
        "TREND / MOVING AVERAGES",
        f"  SMA-20:  ${_fmt(f.get('sma20'))}  |  SMA-50: ${_fmt(f.get('sma50'))}  |  SMA-200: ${_fmt(f.get('sma200'))}",
        f"  Crossover signal: {f.get('sma_cross', 'n/a')}",
        f"  Price vs SMA-50: {_pct_str(f.get('price_vs_sma50'))} ({'above' if (f.get('price_vs_sma50') or 0) > 0 else 'below'})",
        "",
        "MOMENTUM OSCILLATORS",
        f"  RSI-14: {rsi_note}",
        f"  MACD (12/26/9): {macd_note}",
        "",
        "VOLATILITY",
        f"  Bollinger Band position (0=low, 1=high): {bb_note}",
        f"  BB width (normalised): {_fmt(f.get('bb_width'), 4)}",
        f"  ATR-14 as % of price: {_pct_str(f.get('atr_pct'))}",
        f"  Historical vol (20d ann.): {_pct_str(f.get('hist_vol_20d'))}",
        "",
        "VOLUME",
        f"  5d avg vol vs 20d avg: {_fmt(f.get('vol_5d_vs_20d'))}x",
        f"  OBV trend: {f.get('obv_trend', 'n/a')}",
        "",
        "52-WEEK RANGE",
        f"  52w High: ${_fmt(f.get('w52_high'))}  |  52w Low: ${_fmt(f.get('w52_low'))}",
        f"  % below 52w high: {_pct_str(f.get('pct_from_52w_high'))}",
        f"  % above 52w low:  {_pct_str(f.get('pct_from_52w_low'))}",
        "=================================",
    ]
    return "\n".join(lines)
