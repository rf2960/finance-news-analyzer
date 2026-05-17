#!/usr/bin/env python3
"""
Explainable retrieval case studies for FinSight RAG.

This is a deterministic sanity/regression harness, not a production benchmark.
It uses small labeled finance-news fixtures to compare how keyword, TF-IDF,
BM25, and hybrid lexical retrieval rank evidence for common analyst queries.

Use it when changing retrieval logic:

    python retrieval_case_studies.py
    python retrieval_case_studies.py --method hybrid --verbose
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finance_news_analyzer.news_ingester import NewsArticle
from src.finance_news_analyzer.rag_pipeline import chunk_articles, retrieve


@dataclass(frozen=True)
class RetrievalCase:
    name: str
    ticker: str
    company: str
    query: str
    relevant_ids: set[str]
    articles: list[NewsArticle]


def _article(
    cid: str,
    source: str,
    title: str,
    text: str,
    ticker: str = "",
    credibility: float = 0.72,
    day: str = "2026-05-16",
) -> NewsArticle:
    return NewsArticle(
        citation_id=cid,
        source=source,
        title=title,
        url=f"https://example.com/{cid.lower()}",
        published_at=f"{day}T12:00:00+00:00",
        text=text,
        ticker=ticker,
        credibility_weight=credibility,
    )


def build_cases() -> list[RetrievalCase]:
    return [
        RetrievalCase(
            name="AI guidance catalyst",
            ticker="NVDA",
            company="NVIDIA",
            query="NVDA NVIDIA AI chip earnings guidance data center demand",
            relevant_ids={"NVDA-1", "NVDA-3"},
            articles=[
                _article(
                    "NVDA-1",
                    "Reuters",
                    "NVIDIA raises guidance as AI chip demand accelerates",
                    "NVIDIA lifted revenue guidance after data center demand for AI accelerators exceeded expectations.",
                    "NVDA",
                    0.85,
                ),
                _article(
                    "NVDA-2",
                    "MarketWatch",
                    "Mega-cap technology stocks trade mixed",
                    "Broad technology shares were mixed as investors debated interest rates and valuation risk.",
                    "",
                    0.70,
                ),
                _article(
                    "NVDA-3",
                    "CNBC",
                    "Analysts lift NVIDIA price targets after earnings beat",
                    "Analysts raised targets for NVDA following an earnings beat and stronger AI server demand.",
                    "NVDA",
                    0.78,
                ),
            ],
        ),
        RetrievalCase(
            name="Regulatory downside risk",
            ticker="TSLA",
            company="Tesla",
            query="TSLA Tesla regulatory investigation recall margin risk",
            relevant_ids={"TSLA-2", "TSLA-3"},
            articles=[
                _article(
                    "TSLA-1",
                    "Yahoo Finance",
                    "Tesla launches refreshed Model Y in Europe",
                    "Tesla announced a product refresh and new delivery options in several European markets.",
                    "TSLA",
                    0.72,
                ),
                _article(
                    "TSLA-2",
                    "Reuters",
                    "Tesla faces regulator questions over driver assistance recall",
                    "U.S. regulators opened a review into Tesla driver assistance software after recall concerns.",
                    "TSLA",
                    0.85,
                ),
                _article(
                    "TSLA-3",
                    "CNBC",
                    "Tesla margins pressured as price cuts continue",
                    "Analysts warned that price cuts could pressure Tesla auto margins and near-term earnings.",
                    "TSLA",
                    0.78,
                ),
            ],
        ),
        RetrievalCase(
            name="Ticker ambiguity filter",
            ticker="META",
            company="Meta Platforms",
            query="META Meta Platforms advertising revenue AI capex risk",
            relevant_ids={"META-1", "META-2"},
            articles=[
                _article(
                    "META-1",
                    "Financial Times",
                    "Meta ad revenue beats as AI ranking improves engagement",
                    "Meta Platforms reported stronger ad revenue and said AI ranking improved user engagement.",
                    "META",
                    0.88,
                ),
                _article(
                    "META-2",
                    "Reuters",
                    "Meta capex plan worries investors despite revenue beat",
                    "Meta shares fell as investors weighed AI infrastructure capex against advertising growth.",
                    "META",
                    0.85,
                ),
                _article(
                    "META-3",
                    "Benzinga",
                    "Metaverse tokens rally after crypto conference",
                    "Crypto assets connected to metaverse projects rallied after a digital asset conference.",
                    "",
                    0.60,
                ),
            ],
        ),
    ]


def evaluate_case(case: RetrievalCase, method: str, top_k: int = 3) -> dict:
    chunks = chunk_articles(case.articles, case.ticker, case.company)
    ranked = retrieve(
        ticker=case.ticker,
        retrieval_query=case.query,
        chunks=chunks,
        top_k=top_k,
        always_include_bloomberg=False,
        retrieval_method=method,
    )
    ranked_ids = [chunk.article_id or chunk.citation_id for chunk in ranked]
    hits = [rid in case.relevant_ids for rid in ranked_ids]
    precision = sum(hits) / max(len(hits), 1)
    recall = len(set(ranked_ids) & case.relevant_ids) / max(len(case.relevant_ids), 1)
    mrr = 0.0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / rank
            break
    return {
        "case": case.name,
        "method": method,
        "actual_methods": sorted({chunk.retrieval_method for chunk in ranked if chunk.retrieval_method}),
        "ranked_ids": ranked_ids,
        "precision_at_k": precision,
        "recall_at_k": recall,
        "mrr": mrr,
        "chunks": ranked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare FinSight retrieval methods on labeled sanity fixtures.")
    parser.add_argument("--method", choices=["keyword", "tfidf", "bm25", "hybrid", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    methods = ["keyword", "tfidf", "bm25", "hybrid"] if args.method == "all" else [args.method]
    cases = build_cases()
    print("FinSight retrieval case studies")
    print("Note: synthetic labeled fixtures for regression sanity, not production performance metrics.\n")

    for method in methods:
        rows = [evaluate_case(case, method, top_k=args.top_k) for case in cases]
        avg_precision = sum(r["precision_at_k"] for r in rows) / len(rows)
        avg_recall = sum(r["recall_at_k"] for r in rows) / len(rows)
        avg_mrr = sum(r["mrr"] for r in rows) / len(rows)
        print(f"{method.upper():<8} precision@{args.top_k}={avg_precision:.2f}  recall@{args.top_k}={avg_recall:.2f}  mrr={avg_mrr:.2f}")
        for row in rows:
            actual = "/".join(row["actual_methods"]) or method
            fallback_note = f" (actual: {actual})" if actual != method else ""
            print(f"  - {row['case']}{fallback_note}: {', '.join(row['ranked_ids'])}")
            if args.verbose:
                for chunk in row["chunks"]:
                    print(
                        "      "
                        f"rank={chunk.retrieval_rank} id={chunk.article_id} "
                        f"score={chunk.retrieval_score:.3f} breakdown={chunk.score_breakdown}"
                    )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
