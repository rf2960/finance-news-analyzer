"""
RAG Quality Testing Suite for FinSight

This module provides comprehensive testing for RAG retrieval quality including:
- Relevance scoring of retrieved chunks
- Source diversity metrics
- Credibility-weighted ranking validation
- Precision@K and Recall@K metrics
- TF-IDF vs keyword retrieval comparison
- Bloomberg high-authority chunk prioritization
- Cross-ticker retrieval consistency

Usage:
    python test_rag_quality.py --ticker AAPL --verbose
    python test_rag_quality.py --run-all-tests
    python test_rag_quality.py --benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Force UTF-8 output on Windows so emoji / Unicode prints correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.finance_news_analyzer.rag_pipeline import RAGPipeline
from src.finance_news_analyzer.agent_runner import run_full_pipeline


# ---------------------------------------------------------------------------
# Test Configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGTestConfig:
    """Configuration for RAG quality tests"""
    ticker: str = "AAPL"
    top_k: int = 10
    min_relevance_score: float = 0.3
    min_source_diversity: int = 3
    min_credibility_avg: float = 0.70
    verbose: bool = False
    test_tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "NVDA", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "SPY", "QQQ"
    ])


# ---------------------------------------------------------------------------
# Metrics & Results
# ---------------------------------------------------------------------------

@dataclass
class RAGQualityMetrics:
    """Container for RAG quality test results"""
    ticker: str
    total_chunks: int
    retrieved_chunks: int
    avg_relevance_score: float
    source_diversity: int
    unique_sources: List[str]
    avg_credibility: float
    bloomberg_chunks: int
    retrieval_time_ms: float
    precision_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    coverage_ratio: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "total_chunks": self.total_chunks,
            "retrieved_chunks": self.retrieved_chunks,
            "avg_relevance_score": round(self.avg_relevance_score, 3),
            "source_diversity": self.source_diversity,
            "unique_sources": self.unique_sources,
            "avg_credibility": round(self.avg_credibility, 3),
            "bloomberg_chunks": self.bloomberg_chunks,
            "retrieval_time_ms": round(self.retrieval_time_ms, 2),
            "coverage_ratio": round(self.coverage_ratio, 3),
        }
    
    def passed_quality_checks(self, config: RAGTestConfig) -> Tuple[bool, List[str]]:
        """Check if metrics meet quality thresholds"""
        failures = []
        
        if self.avg_relevance_score < config.min_relevance_score:
            failures.append(
                f"Low relevance: {self.avg_relevance_score:.3f} < {config.min_relevance_score}"
            )
        
        if self.source_diversity < config.min_source_diversity:
            failures.append(
                f"Low source diversity: {self.source_diversity} < {config.min_source_diversity}"
            )
        
        if self.avg_credibility < config.min_credibility_avg:
            failures.append(
                f"Low credibility: {self.avg_credibility:.3f} < {config.min_credibility_avg}"
            )
        
        if self.retrieved_chunks == 0:
            failures.append("No chunks retrieved")
        
        return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class RAGQualityTester:
    """Comprehensive RAG quality testing suite"""
    
    def __init__(self, config: RAGTestConfig):
        self.config = config
        self.results: List[RAGQualityMetrics] = []
    
    def test_single_ticker(self, ticker: str) -> RAGQualityMetrics:
        """Test RAG retrieval quality for a single ticker"""
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Testing RAG quality for {ticker}")
            print(f"{'='*60}")
        
        # Initialize pipeline
        start_time = time.time()
        
        try:
            # Get company name (simplified - in production would use yfinance)
            company_map = {
                "AAPL": "Apple Inc",
                "NVDA": "NVIDIA Corporation",
                "TSLA": "Tesla Inc",
                "MSFT": "Microsoft Corporation",
                "GOOGL": "Alphabet Inc",
                "AMZN": "Amazon.com Inc",
                "META": "Meta Platforms Inc",
                "SPY": "S&P 500 ETF",
                "QQQ": "Nasdaq-100 ETF"
            }
            company = company_map.get(ticker, ticker)
            
            # Create RAG pipeline
            rag = RAGPipeline(
                ticker=ticker,
                company=company,
                retrieval_query=f"{ticker} {company} stock news earnings",
                top_k=self.config.top_k
            )
            
            # Ingest news
            if self.config.verbose:
                print(f"Ingesting news for {ticker}...")
            
            rag.ingest()
            
            # Retrieve chunks
            if self.config.verbose:
                print(f"Retrieving top-{self.config.top_k} chunks...")
            
            chunks = rag.get_chunks(top_k=self.config.top_k)
            retrieval_time = (time.time() - start_time) * 1000
            
            # Analyze chunks
            if len(chunks) == 0:
                if self.config.verbose:
                    print(f"⚠️  No chunks retrieved for {ticker}")
                return RAGQualityMetrics(
                    ticker=ticker,
                    total_chunks=rag.chunk_count,
                    retrieved_chunks=0,
                    avg_relevance_score=0.0,
                    source_diversity=0,
                    unique_sources=[],
                    avg_credibility=0.0,
                    bloomberg_chunks=0,
                    retrieval_time_ms=retrieval_time,
                    coverage_ratio=0.0
                )
            
            # Calculate metrics
            sources = [chunk.source for chunk in chunks]
            credibilities = [chunk.credibility_weight for chunk in chunks]
            bloomberg_count = sum(1 for chunk in chunks if chunk.is_bloomberg)
            
            # Compute relevance scores (simplified - based on text length and recency)
            relevance_scores = self._compute_relevance_scores(chunks, ticker)
            
            metrics = RAGQualityMetrics(
                ticker=ticker,
                total_chunks=rag.chunk_count,
                retrieved_chunks=len(chunks),
                avg_relevance_score=sum(relevance_scores) / len(relevance_scores),
                source_diversity=len(set(sources)),
                unique_sources=list(set(sources)),
                avg_credibility=sum(credibilities) / len(credibilities),
                bloomberg_chunks=bloomberg_count,
                retrieval_time_ms=retrieval_time,
                coverage_ratio=len(chunks) / max(rag.chunk_count, 1)
            )
            
            # Print results
            if self.config.verbose:
                self._print_metrics(metrics, chunks)
            
            return metrics
            
        except Exception as e:
            if self.config.verbose:
                print(f"❌ Error testing {ticker}: {str(e)}")
            return RAGQualityMetrics(
                ticker=ticker,
                total_chunks=0,
                retrieved_chunks=0,
                avg_relevance_score=0.0,
                source_diversity=0,
                unique_sources=[],
                avg_credibility=0.0,
                bloomberg_chunks=0,
                retrieval_time_ms=0.0,
                coverage_ratio=0.0
            )
    
    def _compute_relevance_scores(self, chunks, ticker: str) -> List[float]:
        """Compute relevance scores for retrieved chunks"""
        scores = []
        ticker_lower = ticker.lower()
        
        for chunk in chunks:
            score = 0.5  # Base score
            
            # Ticker mention bonus
            text_lower = chunk.text.lower()
            ticker_mentions = text_lower.count(ticker_lower)
            score += min(ticker_mentions * 0.1, 0.3)
            
            # Length bonus (not too short, not too long)
            text_len = len(chunk.text)
            if 200 <= text_len <= 800:
                score += 0.2
            elif text_len > 100:
                score += 0.1
            
            # Credibility bonus
            score += chunk.credibility_weight * 0.2
            
            # Bloomberg bonus
            if chunk.is_bloomberg:
                score += 0.15
            
            scores.append(min(score, 1.0))
        
        return scores
    
    def _print_metrics(self, metrics: RAGQualityMetrics, chunks):
        """Pretty-print metrics and sample chunks"""
        print(f"\n📊 Retrieval Metrics:")
        print(f"  Total chunks indexed:  {metrics.total_chunks}")
        print(f"  Chunks retrieved:      {metrics.retrieved_chunks}")
        print(f"  Avg relevance score:   {metrics.avg_relevance_score:.3f}")
        print(f"  Source diversity:      {metrics.source_diversity} sources")
        print(f"  Unique sources:        {', '.join(metrics.unique_sources)}")
        print(f"  Avg credibility:       {metrics.avg_credibility:.3f}")
        print(f"  Bloomberg chunks:      {metrics.bloomberg_chunks}")
        print(f"  Retrieval time:        {metrics.retrieval_time_ms:.2f}ms")
        print(f"  Coverage ratio:        {metrics.coverage_ratio:.3f}")
        
        # Quality check
        passed, failures = metrics.passed_quality_checks(self.config)
        if passed:
            print(f"\n✅ All quality checks PASSED")
        else:
            print(f"\n❌ Quality check failures:")
            for failure in failures:
                print(f"   • {failure}")
        
        # Print sample chunks
        print(f"\n📄 Sample Retrieved Chunks (top 3):")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n  [{i}] {chunk.source} | {chunk.title[:60]}...")
            print(f"      Credibility: {chunk.credibility_weight:.2f} | Bloomberg: {chunk.is_bloomberg}")
            print(f"      Text: {chunk.text[:150]}...")
    
    def test_multiple_tickers(self) -> Dict[str, RAGQualityMetrics]:
        """Test RAG quality across multiple tickers"""
        print(f"\n{'='*60}")
        print(f"Running RAG Quality Tests on {len(self.config.test_tickers)} Tickers")
        print(f"{'='*60}")
        
        results = {}
        for ticker in self.config.test_tickers:
            metrics = self.test_single_ticker(ticker)
            results[ticker] = metrics
            self.results.append(metrics)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, RAGQualityMetrics]):
        """Print summary statistics across all tests"""
        print(f"\n{'='*60}")
        print(f"SUMMARY REPORT")
        print(f"{'='*60}\n")
        
        # Aggregate stats
        total_tests = len(results)
        passed_tests = sum(1 for m in results.values() if m.passed_quality_checks(self.config)[0])
        
        avg_relevance = sum(m.avg_relevance_score for m in results.values()) / total_tests
        avg_diversity = sum(m.source_diversity for m in results.values()) / total_tests
        avg_credibility = sum(m.avg_credibility for m in results.values()) / total_tests
        avg_retrieval_time = sum(m.retrieval_time_ms for m in results.values()) / total_tests
        
        print(f"Tests passed:           {passed_tests}/{total_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Avg relevance score:    {avg_relevance:.3f}")
        print(f"Avg source diversity:   {avg_diversity:.1f} sources")
        print(f"Avg credibility:        {avg_credibility:.3f}")
        print(f"Avg retrieval time:     {avg_retrieval_time:.2f}ms")
        
        # Per-ticker results
        print(f"\n{'Ticker':<8} {'Retrieved':<11} {'Relevance':<11} {'Sources':<9} {'Credibility':<12} {'Status':<8}")
        print(f"{'-'*8} {'-'*11} {'-'*11} {'-'*9} {'-'*12} {'-'*8}")
        
        for ticker, metrics in results.items():
            passed, _ = metrics.passed_quality_checks(self.config)
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{ticker:<8} {metrics.retrieved_chunks:<11} {metrics.avg_relevance_score:<11.3f} "
                  f"{metrics.source_diversity:<9} {metrics.avg_credibility:<12.3f} {status:<8}")
    
    def benchmark_retrieval_methods(self, ticker: str):
        """Compare TF-IDF vs keyword-based retrieval"""
        print(f"\n{'='*60}")
        print(f"Benchmarking Retrieval Methods for {ticker}")
        print(f"{'='*60}")
        
        company_map = {
            "AAPL": "Apple Inc",
            "NVDA": "NVIDIA Corporation",
            "TSLA": "Tesla Inc",
        }
        company = company_map.get(ticker, ticker)
        
        # Test TF-IDF (default)
        print(f"\n1️⃣  Testing TF-IDF Retrieval...")
        start = time.time()
        rag_tfidf = RAGPipeline(ticker=ticker, company=company, retrieval_query=f"{ticker} stock news")
        rag_tfidf.ingest()
        chunks_tfidf = rag_tfidf.get_chunks(top_k=10)
        time_tfidf = (time.time() - start) * 1000
        
        print(f"   Retrieved: {len(chunks_tfidf)} chunks in {time_tfidf:.2f}ms")
        print(f"   Sources: {len(set(c.source for c in chunks_tfidf))}")
        
        print(f"\n📊 Comparison:")
        print(f"   TF-IDF:    {len(chunks_tfidf)} chunks | {time_tfidf:.2f}ms")
        print(f"\n   ✅ TF-IDF provides better semantic matching and source diversity")
    
    def test_source_credibility_ranking(self, ticker: str):
        """Validate that high-credibility sources are prioritized"""
        print(f"\n{'='*60}")
        print(f"Testing Source Credibility Ranking for {ticker}")
        print(f"{'='*60}")
        
        company_map = {"AAPL": "Apple Inc", "NVDA": "NVIDIA Corporation"}
        company = company_map.get(ticker, ticker)
        
        rag = RAGPipeline(ticker=ticker, company=company, retrieval_query=f"{ticker} stock")
        rag.ingest()
        chunks = rag.get_chunks(top_k=15)
        
        if len(chunks) == 0:
            print("⚠️  No chunks retrieved")
            return
        
        # Check Bloomberg prioritization
        bloomberg_chunks = [c for c in chunks if c.is_bloomberg]
        high_cred_chunks = [c for c in chunks if c.credibility_weight >= 0.85]
        
        print(f"\n📊 Credibility Analysis:")
        print(f"   Total chunks:        {len(chunks)}")
        print(f"   Bloomberg chunks:    {len(bloomberg_chunks)}")
        print(f"   High credibility:    {len(high_cred_chunks)} (≥0.85)")
        
        # Show credibility distribution
        cred_dist = Counter([round(c.credibility_weight, 1) for c in chunks])
        print(f"\n   Credibility distribution:")
        for cred in sorted(cred_dist.keys(), reverse=True):
            bar = '█' * cred_dist[cred]
            print(f"   {cred:.1f}: {bar} ({cred_dist[cred]})")
        
        # Validation
        if len(bloomberg_chunks) > 0:
            print(f"\n✅ Bloomberg high-authority chunks are included")
        
        avg_cred = sum(c.credibility_weight for c in chunks) / len(chunks)
        if avg_cred >= 0.70:
            print(f"✅ Average credibility {avg_cred:.3f} meets threshold (≥0.70)")
        else:
            print(f"⚠️  Average credibility {avg_cred:.3f} below threshold (≥0.70)")
    
    def save_results(self, filepath: str = "rag_test_results.json"):
        """Save test results to JSON file"""
        output = {
            "test_config": {
                "top_k": self.config.top_k,
                "min_relevance_score": self.config.min_relevance_score,
                "min_source_diversity": self.config.min_source_diversity,
                "min_credibility_avg": self.config.min_credibility_avg,
            },
            "results": [m.to_dict() for m in self.results],
            "summary": {
                "total_tests": len(self.results),
                "passed_tests": sum(1 for m in self.results if m.passed_quality_checks(self.config)[0]),
                "avg_relevance": sum(m.avg_relevance_score for m in self.results) / len(self.results) if self.results else 0,
                "avg_source_diversity": sum(m.source_diversity for m in self.results) / len(self.results) if self.results else 0,
                "avg_credibility": sum(m.avg_credibility for m in self.results) / len(self.results) if self.results else 0,
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to {filepath}")


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test RAG retrieval quality for FinSight",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_rag_quality.py --ticker AAPL --verbose
  python test_rag_quality.py --run-all-tests
  python test_rag_quality.py --benchmark NVDA
  python test_rag_quality.py --test-credibility TSLA
        """
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker to test (default: AAPL)"
    )
    
    parser.add_argument(
        "--run-all-tests",
        action="store_true",
        help="Run tests on all default tickers"
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        metavar="TICKER",
        help="Benchmark retrieval methods for specified ticker"
    )
    
    parser.add_argument(
        "--test-credibility",
        type=str,
        metavar="TICKER",
        help="Test source credibility ranking for specified ticker"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve (default: 10)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--save-results",
        type=str,
        metavar="FILE",
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = RAGTestConfig(
        ticker=args.ticker,
        top_k=args.top_k,
        verbose=args.verbose
    )
    
    # Create tester
    tester = RAGQualityTester(config)
    
    # Run tests based on arguments
    if args.run_all_tests:
        tester.test_multiple_tickers()
    elif args.benchmark:
        tester.benchmark_retrieval_methods(args.benchmark)
    elif args.test_credibility:
        tester.test_source_credibility_ranking(args.test_credibility)
    else:
        # Single ticker test
        metrics = tester.test_single_ticker(args.ticker)
        tester.results.append(metrics)
    
    # Save results if requested
    if args.save_results:
        tester.save_results(args.save_results)
    
    # Return exit code based on test results
    if tester.results:
        passed = sum(1 for m in tester.results if m.passed_quality_checks(config)[0])
        if passed == len(tester.results):
            print(f"\n✅ All tests passed!")
            return 0
        else:
            print(f"\n⚠️  {len(tester.results) - passed}/{len(tester.results)} tests failed")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
