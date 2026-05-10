"""
FinSight RAG Quality Evaluation Suite

Evaluates RAG quality in two independent dimensions:

  1. RETRIEVAL QUALITY — Did the system retrieve the right evidence?
     Metrics: Recall@K, Precision@K, MRR, Context Relevance Score (0-100)
     A chunk is "relevant" if it mentions the ticker AND contains
     financial vocabulary matching the query intent.

  2. GENERATION QUALITY — Given the retrieved context, did the pipeline
     produce a well-grounded answer?
     Metrics: Groundedness, Completeness, Correctness, Coherence (0-100)
     Evaluated from the heuristic pipeline output (no API key required).

Usage:
    python test_rag_quality.py --ticker AAPL --verbose
    python test_rag_quality.py --run-all-tests
    python test_rag_quality.py --run-all-tests --save-results demo_data/rag_eval_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.finance_news_analyzer.rag_pipeline import RAGPipeline, RAGChunk
from src.finance_news_analyzer.agent_runner import run_full_pipeline


# ---------------------------------------------------------------------------
# Vocabulary for relevance judgement
# ---------------------------------------------------------------------------

FINANCIAL_TERMS = {
    "earnings", "revenue", "profit", "loss", "guidance", "forecast",
    "analyst", "price", "target", "upgrade", "downgrade", "beat", "miss",
    "quarterly", "annual", "growth", "margin", "eps", "pe", "valuation",
    "stock", "shares", "market", "cap", "buy", "sell", "hold", "neutral",
    "bullish", "bearish", "rally", "decline", "surge", "drop", "acquisition",
    "merger", "dividend", "buyback", "debt", "cash", "revenue", "sales",
    "outlook", "results", "report", "q1", "q2", "q3", "q4", "fiscal",
    "inflation", "rate", "fed", "interest", "gdp", "macro", "sector",
    "competition", "product", "launch", "regulation", "risk", "volatility",
}

COMPANY_MAP = {
    "AAPL": "Apple",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "SPY": "S&P 500 index fund",
    "QQQ": "Nasdaq ETF technology",
}

DEFAULT_TICKERS = list(COMPANY_MAP.keys())


# ---------------------------------------------------------------------------
# Relevance scoring (no LLM — pure lexical)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _chunk_relevance(chunk: RAGChunk, ticker: str, query_tokens: set[str]) -> float:
    """
    Score a single chunk's relevance to the ticker query.

    A chunk is considered relevant when it:
    - Mentions the ticker symbol or company name    (weight 0.40)
    - Shares financial vocabulary with the query    (weight 0.40)
    - Contains at least one specific financial term (weight 0.20)

    Returns a float in [0, 1].
    """
    text = chunk.text.lower()
    text_tokens = _tokenize(chunk.text)

    # 1. Ticker / company mention
    ticker_names = COMPANY_MAP.get(ticker.upper(), ticker).lower().split()
    mention_score = 1.0 if ticker.lower() in text else 0.0
    if not mention_score:
        mention_score = max(1.0 if n in text else 0.0 for n in ticker_names)

    # 2. Query token overlap (Jaccard)
    if query_tokens:
        intersection = text_tokens & query_tokens
        union = text_tokens | query_tokens
        overlap = len(intersection) / max(len(union), 1)
    else:
        overlap = 0.0

    # 3. Financial vocabulary density
    fin_hits = text_tokens & FINANCIAL_TERMS
    fin_score = min(len(fin_hits) / 5.0, 1.0)  # cap at 5 terms

    score = mention_score * 0.40 + overlap * 0.40 + fin_score * 0.20
    return round(score, 4)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

RELEVANT_THRESHOLD = 0.25  # chunk is "relevant" if score >= this


def compute_retrieval_metrics(
    chunks: List[RAGChunk],
    ticker: str,
    query_tokens: set[str],
) -> dict:
    """
    Compute Recall@K, Precision@K, MRR, and Context Relevance Score.
    All scores are in [0, 100].
    """
    if not chunks:
        return {
            "recall_at_k": 0,
            "precision_at_k": 0,
            "mrr": 0,
            "context_relevance": 0,
            "retrieval_score": 0,
            "relevant_count": 0,
            "chunk_scores": [],
        }

    scores = [_chunk_relevance(c, ticker, query_tokens) for c in chunks]
    relevant_flags = [s >= RELEVANT_THRESHOLD for s in scores]
    k = len(chunks)

    # Recall@K: fraction of retrieved chunks that are relevant
    recall_at_k = sum(relevant_flags) / k

    # Precision@K: average relevance score (not binary — continuous)
    precision_at_k = sum(scores) / k

    # MRR: reciprocal rank of first relevant result
    mrr = 0.0
    for rank, is_rel in enumerate(relevant_flags, start=1):
        if is_rel:
            mrr = 1.0 / rank
            break

    # Context Relevance: avg score across all chunks
    context_relevance = sum(scores) / k

    # Combined retrieval score (0-100)
    retrieval_score = round(
        (recall_at_k * 0.35 + precision_at_k * 0.35 + mrr * 0.30) * 100
    )

    return {
        "recall_at_k": round(recall_at_k * 100, 1),
        "precision_at_k": round(precision_at_k * 100, 1),
        "mrr": round(mrr * 100, 1),
        "context_relevance": round(context_relevance * 100, 1),
        "retrieval_score": retrieval_score,
        "relevant_count": sum(relevant_flags),
        "chunk_scores": [round(s, 3) for s in scores],
    }


# ---------------------------------------------------------------------------
# Generation metrics
# ---------------------------------------------------------------------------

def compute_generation_metrics(
    packet: dict,
    chunks: List[RAGChunk],
    ticker: str,
) -> dict:
    """
    Evaluate generation quality from the pipeline output packet.

    Dimensions:
      Completeness   — are all required fields present and non-empty?
      Correctness    — is the direction valid, confidence in [0,1]?
      Groundedness   — does the reasoning reference content from the chunks?
      Coherence      — is the reasoning long and specific (not boilerplate)?
    """
    # ---- Completeness ----
    # Packet fields: direction, confidence, reasoning, thesis_bullets
    direction_val = packet.get("direction", "")
    conf_val = packet.get("confidence")  # "confidence" key (not confidence_score)
    reasoning_val = str(packet.get("reasoning", ""))
    bullets_val = packet.get("thesis_bullets", [])

    completeness = sum([
        bool(direction_val),
        conf_val is not None,
        len(reasoning_val) >= 30,
        len(bullets_val) >= 1,
    ]) / 4.0

    # ---- Correctness ----
    direction_valid = str(direction_val).upper() in {
        "BUY", "SELL", "HOLD", "NEUTRAL", "BULLISH", "BEARISH",
        "STRONG_BUY", "STRONG_SELL"
    }
    conf = conf_val if conf_val is not None else -1
    conf_valid = isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0
    correctness = (int(direction_valid) + int(conf_valid)) / 2.0

    # ---- Groundedness ----
    # Check how much of the reasoning's financial vocabulary overlaps with chunks
    reasoning_text = str(packet.get("reasoning", ""))
    bullets = " ".join(str(b) for b in packet.get("thesis_bullets", []))
    generated_tokens = _tokenize(reasoning_text + " " + bullets)
    generated_fin = generated_tokens & FINANCIAL_TERMS

    chunk_text_all = " ".join(c.text for c in chunks)
    chunk_tokens = _tokenize(chunk_text_all)
    chunk_fin = chunk_tokens & FINANCIAL_TERMS

    if chunk_fin:
        # What fraction of financial terms in generated output also appear in chunks?
        overlap = generated_fin & chunk_fin
        groundedness = len(overlap) / max(len(generated_fin), 1)
        groundedness = min(groundedness, 1.0)
    else:
        groundedness = 0.0

    # Penalize very short reasoning (likely boilerplate)
    if len(reasoning_text) < 80:
        groundedness *= 0.5

    # ---- Coherence ----
    # Is the reasoning specific (long enough, ticker mentioned, financial terms)?
    ticker_in_reasoning = ticker.lower() in reasoning_text.lower()
    reasoning_fin_terms = len(generated_tokens & FINANCIAL_TERMS)
    reasoning_length_ok = len(reasoning_text) >= 100

    coherence = (
        int(ticker_in_reasoning) * 0.35
        + min(reasoning_fin_terms / 5.0, 1.0) * 0.35
        + int(reasoning_length_ok) * 0.30
    )

    # Combined generation score (0-100)
    generation_score = round(
        (completeness * 0.25 + correctness * 0.20 + groundedness * 0.30 + coherence * 0.25) * 100
    )

    # Label
    if generation_score >= 70 and groundedness >= 0.5:
        label = "correct"
    elif generation_score >= 40:
        label = "partial"
    else:
        label = "hallucination"

    return {
        "completeness": round(completeness * 100, 1),
        "correctness": round(correctness * 100, 1),
        "groundedness": round(groundedness * 100, 1),
        "coherence": round(coherence * 100, 1),
        "generation_score": generation_score,
        "label": label,
        "direction": packet.get("direction", "N/A"),
        "confidence": conf,
    }


# ---------------------------------------------------------------------------
# Per-ticker evaluation
# ---------------------------------------------------------------------------

def evaluate_ticker(
    ticker: str,
    top_k: int = 8,
    verbose: bool = False,
) -> dict:
    """
    Full evaluation for a single ticker:
    Retrieval quality + Generation quality.
    """
    company = COMPANY_MAP.get(ticker.upper(), ticker)
    query = f"{ticker} {company} stock news earnings price analyst"
    query_tokens = _tokenize(query)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluating: {ticker} ({company})")
        print(f"Query: {query[:80]}")
        print(f"{'='*60}")

    # ---- RETRIEVAL ----
    t0 = time.time()
    try:
        rag = RAGPipeline(
            ticker=ticker,
            company=company,
            retrieval_query=query,
            top_k=top_k,
        )
        rag.ingest()
        chunks = rag.get_chunks(top_k=top_k)
        retrieval_ms = round((time.time() - t0) * 1000, 1)
        total_indexed = rag.chunk_count
    except Exception as e:
        return {
            "ticker": ticker,
            "error": str(e),
            "retrieval": {},
            "generation": {},
            "retrieval_pass": False,
            "generation_pass": False,
            "overall_pass": False,
        }

    ret = compute_retrieval_metrics(chunks, ticker, query_tokens)
    ret["retrieval_time_ms"] = retrieval_ms
    ret["chunks_indexed"] = total_indexed
    ret["chunks_retrieved"] = len(chunks)

    # ---- GENERATION ----
    t1 = time.time()
    try:
        packet = run_full_pipeline(
            ticker=ticker,
            openai_api_key=None,   # heuristic mode — no LLM, no API call
            top_k=top_k,
            include_rss=True,
        )
        generation_ms = round((time.time() - t1) * 1000, 1)
    except Exception as e:
        packet = {}
        generation_ms = 0

    gen = compute_generation_metrics(packet, chunks, ticker)
    gen["generation_time_ms"] = generation_ms

    # ---- PASS / FAIL thresholds ----
    retrieval_pass = (
        ret["retrieval_score"] >= 40           # Recall@K + Precision@K + MRR >= 40
        and ret["relevant_count"] >= 1          # At least 1 relevant chunk
    )
    generation_pass = (
        gen["generation_score"] >= 50          # Overall generation >= 50
        and gen["completeness"] >= 75           # All key fields present
    )
    overall_pass = retrieval_pass and generation_pass

    if verbose:
        _print_verbose(ticker, ret, gen, chunks)

    return {
        "ticker": ticker,
        "retrieval": ret,
        "generation": gen,
        "retrieval_pass": retrieval_pass,
        "generation_pass": generation_pass,
        "overall_pass": overall_pass,
    }


def _print_verbose(ticker: str, ret: dict, gen: dict, chunks: list):
    """Pretty-print detailed results for a single ticker."""
    print(f"\n--- RETRIEVAL QUALITY (top-{ret['chunks_retrieved']} chunks) ---")
    print(f"  Chunks indexed:      {ret['chunks_indexed']}")
    print(f"  Chunks retrieved:    {ret['chunks_retrieved']}")
    print(f"  Relevant chunks:     {ret['relevant_count']}/{ret['chunks_retrieved']} "
          f"(threshold >= {RELEVANT_THRESHOLD})")
    print(f"  Recall@{ret['chunks_retrieved']}:         {ret['recall_at_k']:.1f}/100")
    print(f"  Precision@{ret['chunks_retrieved']}:      {ret['precision_at_k']:.1f}/100")
    print(f"  MRR:                 {ret['mrr']:.1f}/100")
    print(f"  Context Relevance:   {ret['context_relevance']:.1f}/100")
    print(f"  Retrieval Score:     {ret['retrieval_score']}/100")
    print(f"  Retrieval time:      {ret['retrieval_time_ms']}ms")

    print(f"\n  Chunk-level relevance scores:")
    for i, (chunk, score) in enumerate(zip(chunks, ret['chunk_scores']), 1):
        relevant = "RELEVANT" if score >= RELEVANT_THRESHOLD else "not relevant"
        src = chunk.source[:20].ljust(20)
        print(f"    [{i:2d}] {src}  score={score:.3f}  ({relevant})")
        if score >= RELEVANT_THRESHOLD:
            print(f"         {chunk.title[:70]}...")

    print(f"\n--- GENERATION QUALITY ---")
    print(f"  Direction:           {gen['direction']}")
    print(f"  Confidence:          {gen['confidence']}")
    print(f"  Completeness:        {gen['completeness']:.1f}/100")
    print(f"  Correctness:         {gen['correctness']:.1f}/100")
    print(f"  Groundedness:        {gen['groundedness']:.1f}/100")
    print(f"  Coherence:           {gen['coherence']:.1f}/100")
    print(f"  Generation Score:    {gen['generation_score']}/100")
    print(f"  Label:               {gen['label'].upper()}")
    print(f"  Generation time:     {gen['generation_time_ms']}ms")


# ---------------------------------------------------------------------------
# Multi-ticker summary
# ---------------------------------------------------------------------------

def run_all_tests(top_k: int = 8, verbose: bool = False) -> List[dict]:
    print(f"\n{'='*60}")
    print(f"FinSight RAG Evaluation — {len(DEFAULT_TICKERS)} Tickers")
    print(f"  Retrieval:  Recall@K, Precision@K, MRR, Context Relevance")
    print(f"  Generation: Groundedness, Completeness, Correctness, Coherence")
    print(f"{'='*60}")

    results = []
    for ticker in DEFAULT_TICKERS:
        r = evaluate_ticker(ticker, top_k=top_k, verbose=verbose)
        results.append(r)
        if not verbose:
            ret_s = r["retrieval"].get("retrieval_score", 0)
            gen_s = r["generation"].get("generation_score", 0)
            label = r["generation"].get("label", "N/A")
            status = "PASS" if r["overall_pass"] else "FAIL"
            print(f"  {ticker:<6}  Retrieval={ret_s:3d}  Generation={gen_s:3d}  "
                  f"Label={label:<13} {status}")

    _print_summary(results)
    return results


def _print_summary(results: List[dict]):
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("\nNo valid results.")
        return

    total = len(valid)
    passed = sum(1 for r in valid if r["overall_pass"])
    ret_pass = sum(1 for r in valid if r["retrieval_pass"])
    gen_pass = sum(1 for r in valid if r["generation_pass"])

    avg_ret = sum(r["retrieval"].get("retrieval_score", 0) for r in valid) / total
    avg_recall = sum(r["retrieval"].get("recall_at_k", 0) for r in valid) / total
    avg_prec = sum(r["retrieval"].get("precision_at_k", 0) for r in valid) / total
    avg_mrr = sum(r["retrieval"].get("mrr", 0) for r in valid) / total
    avg_ctx = sum(r["retrieval"].get("context_relevance", 0) for r in valid) / total

    avg_gen = sum(r["generation"].get("generation_score", 0) for r in valid) / total
    avg_ground = sum(r["generation"].get("groundedness", 0) for r in valid) / total
    avg_complete = sum(r["generation"].get("completeness", 0) for r in valid) / total
    avg_correct = sum(r["generation"].get("correctness", 0) for r in valid) / total
    avg_coher = sum(r["generation"].get("coherence", 0) for r in valid) / total

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Overall PASS:        {passed}/{total}")
    print(f"  Retrieval PASS:      {ret_pass}/{total}")
    print(f"  Generation PASS:     {gen_pass}/{total}")

    print(f"\n  --- Retrieval Averages ---")
    print(f"  Recall@K:            {avg_recall:.1f}/100")
    print(f"  Precision@K:         {avg_prec:.1f}/100")
    print(f"  MRR:                 {avg_mrr:.1f}/100")
    print(f"  Context Relevance:   {avg_ctx:.1f}/100")
    print(f"  Retrieval Score:     {avg_ret:.1f}/100")

    print(f"\n  --- Generation Averages ---")
    print(f"  Groundedness:        {avg_ground:.1f}/100")
    print(f"  Completeness:        {avg_complete:.1f}/100")
    print(f"  Correctness:         {avg_correct:.1f}/100")
    print(f"  Coherence:           {avg_coher:.1f}/100")
    print(f"  Generation Score:    {avg_gen:.1f}/100")

    label_counts = {}
    for r in valid:
        lbl = r["generation"].get("label", "N/A")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    print(f"\n  --- Labels ---")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"  {lbl:<15}: {cnt}/{total}")

    print(f"\n{'='*60}")
    header = f"{'Ticker':<6}  {'Ret Score':>9}  {'Recall@K':>8}  {'Prec@K':>6}  {'MRR':>5}  {'Ground':>6}  {'Complt':>6}  {'GenScr':>6}  {'Label':<13}  {'Status'}"
    print(header)
    print("-" * len(header))
    for r in valid:
        ret = r["retrieval"]
        gen = r["generation"]
        status = "PASS" if r["overall_pass"] else "FAIL"
        print(
            f"{r['ticker']:<6}  "
            f"{ret.get('retrieval_score',0):>9}  "
            f"{ret.get('recall_at_k',0):>8.1f}  "
            f"{ret.get('precision_at_k',0):>6.1f}  "
            f"{ret.get('mrr',0):>5.1f}  "
            f"{gen.get('groundedness',0):>6.1f}  "
            f"{gen.get('completeness',0):>6.1f}  "
            f"{gen.get('generation_score',0):>6}  "
            f"{gen.get('label','N/A'):<13}  "
            f"{status}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FinSight RAG Quality Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluates retrieval quality and generation quality separately.

Examples:
  python test_rag_quality.py --ticker AAPL --verbose
  python test_rag_quality.py --run-all-tests
  python test_rag_quality.py --run-all-tests --save-results demo_data/rag_eval_results.json
        """,
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker to test (default: AAPL)")
    parser.add_argument("--run-all-tests", action="store_true", help="Test all 9 default tickers")
    parser.add_argument("--top-k", type=int, default=8, help="Chunks to retrieve (default: 8)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed chunk-level scores")
    parser.add_argument("--save-results", metavar="FILE", help="Save results to JSON")

    # Legacy compatibility flags (accepted but treated as no-ops or redirected)
    parser.add_argument("--benchmark", metavar="TICKER", help="Run single-ticker verbose evaluation")
    parser.add_argument("--test-credibility", metavar="TICKER", help="Alias for --ticker TICKER --verbose")

    args = parser.parse_args()

    # Handle legacy flags
    if args.benchmark:
        args.ticker = args.benchmark
        args.verbose = True
    if args.test_credibility:
        args.ticker = args.test_credibility
        args.verbose = True

    results = []
    if args.run_all_tests:
        results = run_all_tests(top_k=args.top_k, verbose=args.verbose)
    else:
        r = evaluate_ticker(args.ticker, top_k=args.top_k, verbose=args.verbose)
        results = [r]
        if not args.verbose:
            ret_s = r["retrieval"].get("retrieval_score", 0)
            gen_s = r["generation"].get("generation_score", 0)
            label = r["generation"].get("label", "N/A")
            status = "PASS" if r["overall_pass"] else "FAIL"
            print(f"\n{args.ticker}: Retrieval={ret_s}/100  Generation={gen_s}/100  "
                  f"Label={label}  {status}")

    if args.save_results:
        # Build a clean summary for JSON
        output = {
            "tickers": results,
            "summary": {
                "total": len(results),
                "overall_pass": sum(1 for r in results if r.get("overall_pass")),
                "retrieval_pass": sum(1 for r in results if r.get("retrieval_pass")),
                "generation_pass": sum(1 for r in results if r.get("generation_pass")),
            }
        }
        with open(args.save_results, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.save_results}")

    # Exit code
    all_passed = all(r.get("overall_pass", False) for r in results)
    if all_passed:
        print(f"\nAll tests passed!")
        return 0
    else:
        failed = [r["ticker"] for r in results if not r.get("overall_pass")]
        print(f"\nFailed tickers: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
