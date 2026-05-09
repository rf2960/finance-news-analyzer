#!/usr/bin/env python3
"""
FinSight RAG – Command-line runner.

Usage examples:
    # Heuristic mode (no API key needed) – positional or --ticker flag
    python run_analysis.py NVDA
    python run_analysis.py --ticker NVDA

    # Verbose output
    python run_analysis.py --ticker AAPL --verbose

    # GPT-4o-mini mode
    python run_analysis.py --ticker NVDA --openai-key sk-...

    # Save result to demo_data/signals.json
    python run_analysis.py --ticker AAPL --save-signal

    # Skip RSS feeds (faster, Yahoo Finance only)
    python run_analysis.py --ticker MSFT --no-rss

    # Pretty-print the raw JSON packet
    python run_analysis.py --ticker TSLA --json

    # Set investment horizon (5 or 20 days)
    python run_analysis.py --ticker AAPL --horizon 20
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

# Force UTF-8 output on Windows so emoji / Unicode prints correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure FinSight_RAG/src is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_analysis",
        description="FinSight RAG – run the full multi-agent pipeline for a ticker.",
    )
    # Accept ticker as either a positional argument OR --ticker flag
    parser.add_argument(
        "ticker_pos",
        nargs="?",
        default=None,
        metavar="TICKER",
        help="Stock ticker symbol as positional arg, e.g. NVDA (alternative to --ticker)",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        metavar="TICKER",
        help="Stock ticker symbol, e.g. NVDA",
    )
    parser.add_argument(
        "--openai-key",
        default=None,
        metavar="KEY",
        help="OpenAI API key (enables GPT-4o-mini agents). Omit for heuristic mode.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of RAG chunks to retrieve (default: 8)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=[5, 20],
        default=5,
        help="Investment horizon in days: 5 (short-term) or 20 (medium-term). Default: 5",
    )
    parser.add_argument(
        "--no-rss",
        action="store_true",
        help="Skip RSS feeds and use Yahoo Finance only (faster)",
    )
    parser.add_argument(
        "--save-signal",
        "--save",
        action="store_true",
        dest="save_signal",
        help="Append the result to demo_data/signals.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="raw_json",
        help="Print the full raw JSON packet instead of the summary",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed/verbose output",
    )
    args = parser.parse_args()

    # Resolve ticker from --ticker flag or positional argument
    raw_ticker = args.ticker or args.ticker_pos
    if not raw_ticker:
        parser.error(
            "A ticker symbol is required. Use:\n"
            "  python run_analysis.py AAPL\n"
            "  python run_analysis.py --ticker AAPL"
        )

    ticker = raw_ticker.upper().strip()
    print(f"\n{'='*60}")
    print(f"  FinSight RAG – Analyzing {ticker}")
    print(f"  Mode: {'GPT-4o-mini (LLM)' if args.openai_key else 'Heuristic (local)'}")
    print(f"  RSS feeds: {'disabled' if args.no_rss else 'enabled'}")
    print(f"  Horizon: {args.horizon} days")
    if args.verbose:
        print(f"  Verbose: enabled")
        print(f"  Top-K chunks: {args.top_k}")
        if args.openai_key:
            print(f"  Model: {args.model}")
    print(f"{'='*60}\n")

    try:
        from src.finance_news_analyzer.agent_runner import run_full_pipeline
    except ImportError as exc:
        print(f"[ERROR] Could not import agent_runner: {exc}")
        print("Ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

    try:
        print("⏳ Fetching news and running agent pipeline…")
        packet = run_full_pipeline(
            ticker=ticker,
            openai_api_key=args.openai_key,
            openai_model=args.model,
            top_k=args.top_k,
            include_rss=not args.no_rss,
        )
        if args.verbose:
            print(f"   [verbose] Pipeline call completed successfully")
    except ValueError as ve:
        print(f"\n[ERROR] {ve}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}")
        raise

    meta = packet.pop("_pipeline_meta", {})

    # ── Pipeline stats ────────────────────────────────────────────────────
    print(f"✅ Pipeline complete")
    print(f"   Articles fetched : {meta.get('articles_fetched', '?')}")
    print(f"   Chunks indexed   : {meta.get('chunks_indexed', '?')}")
    print(f"   Chunks retrieved : {meta.get('chunks_retrieved', '?')}")
    if meta.get("sources"):
        print(f"   Sources          : {', '.join(meta['sources'])}")
    if args.verbose and meta:
        print(f"   [verbose] Full meta: {json.dumps(meta, default=str)}")
    print()

    if args.raw_json:
        print(json.dumps(packet, indent=2, default=str))
        return

    # ── Human-readable summary ────────────────────────────────────────────
    direction_emoji = {"Bullish": "📈", "Bearish": "📉", "Neutral": "➡️"}.get(
        packet.get("direction", "Neutral"), "➡️"
    )
    print(f"  {direction_emoji}  {packet['ticker']} ({packet.get('company','')})  [{packet.get('direction','?').upper()}]")
    print(f"  Horizon     : {packet.get('horizon_days', args.horizon)} days")
    print(f"  Confidence  : {packet.get('confidence', 0):.0%}")
    print(f"  Src quality : {packet.get('source_quality', 0):.0%}")
    print(f"  Sector      : {packet.get('sector','?')}")
    print()
    print(f"  Reasoning:\n  {packet.get('reasoning','')}")
    print()

    snap = packet.get("market_snapshot") or {}
    if snap.get("last_price"):
        print(f"  Last price  : ${snap['last_price']:.2f}")
        day_chg = snap.get("day_change")
        if day_chg is not None:
            sign = "+" if day_chg >= 0 else ""
            print(f"  Day change  : {sign}{day_chg:.1%}")

    print()
    if packet.get("thesis_bullets"):
        print("  Thesis drivers:")
        for b in packet["thesis_bullets"]:
            print(f"    • {b}")
        print()

    if packet.get("risk_factors"):
        print("  Risks:")
        for r in packet["risk_factors"]:
            print(f"    ⚠ {r}")
        print()

    citations = packet.get("citations", [])
    if citations:
        print("  Sources used:")
        for cit in citations[:4]:
            cred = cit.get("credibility_weight", 0)
            marker = "⭐" if "bloomberg" in cit.get("source", "").lower() else "  "
            print(f"    {marker} {cit['source']} (credibility {cred:.0%})")
        print()

    print(f"  Agent trace:")
    for step in packet.get("agent_trace", []):
        print(f"    [{step.get('agent','')}] {step.get('summary','')[:120]}")
    print()

    # ── Save ──────────────────────────────────────────────────────────────
    if args.save_signal:
        signals_path = ROOT / "demo_data" / "signals.json"
        try:
            existing = json.loads(signals_path.read_text(encoding="utf-8"))
            existing_ids = {s.get("id") for s in existing}
            if packet.get("id") not in existing_ids:
                existing.append(packet)
                signals_path.write_text(
                    json.dumps(existing, indent=2, default=str),
                    encoding="utf-8",
                )
                print(f"✅ Saved to {signals_path}")
            else:
                print(f"ℹ  Signal ID already exists in {signals_path}; not saved.")
        except Exception as save_err:
            print(f"[WARN] Could not save: {save_err}")

    if args.verbose:
        print(f"  [verbose] Horizon used: {args.horizon} days")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
