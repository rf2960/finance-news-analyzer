"""
RAG Pipeline for FinSight.

Responsibilities:
  1. Chunk ingested news articles into retrievable segments.
  2. Build a TF-IDF retrieval index over all chunks.
  3. Return the top-k most relevant RetrievedChunk objects for a given
     ticker / query, ready to be consumed by the agent workflow.

The pipeline deliberately avoids heavy ML dependencies:
  • scikit-learn (TF-IDF + cosine similarity) is preferred.
  • A pure-stdlib keyword frequency fallback is used when scikit-learn
    is not installed, so the system always runs.

Bloomberg-sourced chunks are tagged with a ``high_authority`` flag and
are always included in the top results alongside TF-IDF winners, to
implement the "Source-Aware" retrieval strategy from the proposal.
"""
from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from src.finance_news_analyzer.news_ingester import NewsArticle

# ---------------------------------------------------------------------------
# Chunk dataclass  (maps directly onto the agent system's RetrievedChunk)
# ---------------------------------------------------------------------------

@dataclass
class RAGChunk:
    citation_id: str
    source: str
    title: str
    published_at: str
    ticker: str
    text: str
    url: str = ""
    chunk_index: int = 0
    article_id: str = ""
    word_count: int = 0
    ticker_match: bool = False
    company_match: bool = False
    recency_days: float | None = None
    credibility_weight: float = 0.65
    is_bloomberg: bool = False   # High-authority flag
    retrieval_rank: int | None = None
    retrieval_score: float = 0.0
    retrieval_method: str = ""
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_retrieved_chunk_dict(self) -> dict:
        """Return kwargs suitable for constructing a RetrievedChunk."""
        return {
            "citation_id": self.citation_id,
            "source": self.source,
            "title": self.title,
            "published_at": self.published_at,
            "ticker": self.ticker,
            "text": self.text,
        }


# ---------------------------------------------------------------------------
# Article chunker
# ---------------------------------------------------------------------------

_WORD_LIMIT = 180   # target words per chunk
_FINANCIAL_TERMS = {
    "earnings", "revenue", "profit", "margin", "guidance", "forecast",
    "analyst", "price", "target", "upgrade", "downgrade", "valuation",
    "stock", "shares", "market", "buy", "sell", "hold", "bullish",
    "bearish", "demand", "growth", "risk", "regulation", "tariff",
    "cash", "debt", "capex", "ai", "cloud", "chip", "semiconductor",
    "inflation", "rates", "fed", "oil", "macro", "sector",
}


def _split_into_chunks(text: str, max_words: int = _WORD_LIMIT) -> list[str]:
    """Split text into overlapping word-windows."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks: list[str] = []
    step = max(max_words // 2, 50)
    for start in range(0, len(words), step):
        chunk_words = words[start: start + max_words]
        chunks.append(" ".join(chunk_words))
        if start + max_words >= len(words):
            break
    return chunks or [text]


def _parse_published_at(raw: str) -> datetime | None:
    """Parse common ISO-ish dates from feeds into timezone-aware UTC datetimes."""
    if not raw:
        return None
    cleaned = raw.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def _recency_days(published_at: str) -> float | None:
    dt = _parse_published_at(published_at)
    if dt is None:
        return None
    delta = datetime.now(timezone.utc) - dt
    return max(delta.total_seconds() / 86400.0, 0.0)


def _recency_score(days: float | None) -> float:
    """Decay from 1.0 today to roughly 0.2 after one month."""
    if days is None:
        return 0.45
    return max(0.20, min(1.0, 1.0 - days / 38.0))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9]{2,}\b", text.lower()))


def _token_list(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z][a-zA-Z0-9]{2,}\b", text.lower())


def _ticker_mentioned(text: str, ticker: str) -> bool:
    return bool(re.search(rf"(?<![A-Z0-9]){re.escape(ticker.upper())}(?![A-Z0-9])", text.upper()))


def _company_terms(company: str) -> list[str]:
    stop = {"inc", "corp", "corporation", "company", "limited", "plc", "holdings", "class"}
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", company) if len(w) >= 4]
    return [w for w in words if w not in stop][:3]


def _company_mentioned(text: str, company: str) -> bool:
    if not company:
        return False
    lowered = text.lower()
    return any(term in lowered for term in _company_terms(company))


def _source_authority(chunk: RAGChunk) -> float:
    source = chunk.source.lower()
    if chunk.is_bloomberg:
        return 1.0
    if any(name in source for name in ("reuters", "wall street journal", "financial times", "barron")):
        return 0.90
    if any(name in source for name in ("cnbc", "marketwatch", "yahoo finance")):
        return 0.72
    return 0.55


def _intent_overlap(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    intent = (query_tokens | _FINANCIAL_TERMS) & chunk_tokens
    return min(len(intent) / 8.0, 1.0)


def chunk_articles(
    articles: list[NewsArticle],
    ticker: str,
    company: str = "",
    max_words: int = _WORD_LIMIT,
) -> list[RAGChunk]:
    """
    Convert a list of :class:`NewsArticle` objects into :class:`RAGChunk`
    objects, splitting long articles when necessary.
    """
    chunks: list[RAGChunk] = []
    seen_texts: set[str] = set()

    for art in articles:
        full_text = f"{art.title}. {art.text}".strip()
        segments = _split_into_chunks(full_text, max_words)
        for idx, segment in enumerate(segments):
            # Deduplicate near-identical segments
            norm = re.sub(r"\s+", " ", segment.lower())[:120]
            if norm in seen_texts:
                continue
            seen_texts.add(norm)

            chunk_id = f"{art.citation_id}-{idx:02d}" if idx else art.citation_id
            chunks.append(
                RAGChunk(
                    citation_id=chunk_id,
                    source=art.source,
                    title=art.title,
                    published_at=art.published_at,
                    ticker=art.ticker or ticker.upper(),
                    text=segment,
                    url=art.url,
                    chunk_index=idx,
                    article_id=art.citation_id,
                    word_count=len(segment.split()),
                    ticker_match=_ticker_mentioned(segment, ticker),
                    company_match=_company_mentioned(segment, company),
                    recency_days=_recency_days(art.published_at),
                    credibility_weight=art.credibility_weight,
                    is_bloomberg="bloomberg" in art.source.lower(),
                )
            )
    return chunks


# ---------------------------------------------------------------------------
# Retrieval: TF-IDF (preferred) or keyword fallback
# ---------------------------------------------------------------------------

def _build_query(ticker: str, retrieval_query: str) -> str:
    """Combine ticker and free-text query into a single search string."""
    parts = [ticker.upper()]
    if retrieval_query:
        parts.append(retrieval_query)
    return " ".join(parts)


def _tfidf_retrieve(
    query: str,
    chunks: list[RAGChunk],
    top_k: int,
) -> list[tuple[RAGChunk, float]]:
    """Rank chunks by TF-IDF cosine similarity (requires scikit-learn)."""
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    corpus = [c.text for c in chunks]
    all_texts = corpus + [query]
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=8000,
        sublinear_tf=True,
        min_df=1,
    )
    tfidf_matrix = vec.fit_transform(all_texts)
    query_vec = tfidf_matrix[-1]
    chunk_vecs = tfidf_matrix[:-1]
    sims = cosine_similarity(query_vec, chunk_vecs).flatten()

    return [(chunks[i], float(sims[i])) for i in range(len(chunks))]


def _keyword_retrieve(
    query: str,
    chunks: list[RAGChunk],
    top_k: int,
) -> list[tuple[RAGChunk, float]]:
    """
    Fallback: rank by normalised keyword hit count + credibility weight.
    No external dependencies.
    """
    keywords = set(re.findall(r"\b\w{3,}\b", query.lower()))

    def _score(chunk: RAGChunk) -> float:
        lowered = chunk.text.lower()
        hits = sum(1 for kw in keywords if kw in lowered)
        return hits / max(len(keywords), 1)

    return [(_c, _score(_c)) for _c in chunks]


def _bm25_retrieve(
    query: str,
    chunks: list[RAGChunk],
    top_k: int,
    k1: float = 1.45,
    b: float = 0.72,
) -> list[tuple[RAGChunk, float]]:
    """
    Pure-Python BM25 retrieval.

    BM25 is a strong fit for finance news because ticker symbols, company
    names, product names, and event terms are high-signal lexical anchors.
    """
    query_terms = _token_list(query)
    if not query_terms or not chunks:
        return [(c, 0.0) for c in chunks]

    docs = [_token_list(c.title + " " + c.text) for c in chunks]
    avg_len = sum(len(doc) for doc in docs) / max(len(docs), 1)
    doc_freq: dict[str, int] = {}
    for doc in docs:
        for term in set(doc):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    n_docs = len(docs)
    raw_scores: list[float] = []
    for doc in docs:
        term_counts: dict[str, int] = {}
        for term in doc:
            term_counts[term] = term_counts.get(term, 0) + 1
        doc_len = len(doc) or 1
        score = 0.0
        for term in query_terms:
            tf = term_counts.get(term, 0)
            if tf == 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / max(avg_len, 1.0))
            score += idf * (tf * (k1 + 1)) / denom
        raw_scores.append(score)

    max_score = max(raw_scores) if raw_scores else 0.0
    if max_score <= 0:
        return [(chunks[i], 0.0) for i in range(len(chunks))]
    return [(chunks[i], raw_scores[i] / max_score) for i in range(len(chunks))]


def _minmax_scores(scores: list[tuple[RAGChunk, float]]) -> dict[str, float]:
    if not scores:
        return {}
    vals = [score for _, score in scores]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        equal_score = 1.0 if hi > 0 else 0.0
        return {chunk.citation_id: equal_score for chunk, _ in scores}
    return {chunk.citation_id: (score - lo) / (hi - lo) for chunk, score in scores}


def _hybrid_lexical_retrieve(
    query: str,
    chunks: list[RAGChunk],
    top_k: int,
) -> list[tuple[RAGChunk, float]]:
    """
    Blend TF-IDF and BM25 lexical signals before metadata reranking.

    TF-IDF helps broad query/document overlap; BM25 is sharper for sparse,
    exact financial event terms. The blend remains fully inspectable.
    """
    bm25_scores = _bm25_retrieve(query, chunks, top_k)
    try:
        tfidf_scores = _tfidf_retrieve(query, chunks, top_k)
    except Exception:
        tfidf_scores = _keyword_retrieve(query, chunks, top_k)

    bm25_norm = _minmax_scores(bm25_scores)
    tfidf_norm = _minmax_scores(tfidf_scores)
    chunk_by_id = {c.citation_id: c for c in chunks}
    combined: list[tuple[RAGChunk, float]] = []
    for cid, chunk in chunk_by_id.items():
        score = bm25_norm.get(cid, 0.0) * 0.55 + tfidf_norm.get(cid, 0.0) * 0.45
        combined.append((chunk, score))
    return combined


def _metadata_score(chunk: RAGChunk, query_tokens: set[str]) -> tuple[float, dict[str, float]]:
    chunk_tokens = _tokens(chunk.title + " " + chunk.text)
    ticker_company = 1.0 if chunk.ticker_match else (0.75 if chunk.company_match else 0.0)
    recency = _recency_score(chunk.recency_days)
    authority = _source_authority(chunk)
    intent = _intent_overlap(query_tokens, chunk_tokens)
    breakdown = {
        "ticker_company": round(ticker_company, 3),
        "source_credibility": round(float(chunk.credibility_weight), 3),
        "source_authority": round(authority, 3),
        "recency": round(recency, 3),
        "financial_intent": round(intent, 3),
    }
    metadata = (
        ticker_company * 0.30
        + float(chunk.credibility_weight) * 0.25
        + authority * 0.15
        + recency * 0.15
        + intent * 0.15
    )
    return metadata, breakdown


def _blend_ranked_results(
    base_scores: Iterable[tuple[RAGChunk, float]],
    query: str,
    method: str,
) -> list[tuple[RAGChunk, float]]:
    query_tokens = _tokens(query)
    ranked: list[tuple[RAGChunk, float]] = []
    for chunk, base_score in base_scores:
        metadata, breakdown = _metadata_score(chunk, query_tokens)
        blended = float(base_score) * 0.58 + metadata * 0.42
        chunk.retrieval_score = round(blended, 4)
        chunk.retrieval_method = method
        chunk.score_breakdown = {"semantic": round(float(base_score), 3), **breakdown}
        ranked.append((chunk, blended))
    ranked.sort(key=lambda item: item[1], reverse=True)
    for rank, (chunk, _) in enumerate(ranked, start=1):
        chunk.retrieval_rank = rank
    return ranked


def retrieve(
    ticker: str,
    retrieval_query: str,
    chunks: list[RAGChunk],
    top_k: int = 8,
    always_include_bloomberg: bool = True,
    source_allowlist: list[str] | None = None,
    min_credibility: float = 0.0,
    max_age_days: int | None = None,
    retrieval_method: str = "hybrid",
) -> list[RAGChunk]:
    """
    Return the top-k most relevant chunks for the given ticker and query.

    Bloomberg chunks are always guaranteed a slot in the results (up to
    their actual count), implementing the "Source-Aware" retrieval from
    the project proposal.

    Args:
        ticker:                    Stock ticker symbol.
        retrieval_query:           Free-text retrieval query.
        chunks:                    Pool of candidate RAGChunk objects.
        top_k:                     Maximum number of chunks to return.
        always_include_bloomberg:  Pin Bloomberg chunks into results.

    Returns:
        Ordered list of :class:`RAGChunk` objects (most relevant first).
    """
    if not chunks:
        return []

    query = _build_query(ticker, retrieval_query)
    filtered_chunks = list(chunks)
    if source_allowlist:
        allowed = {s.lower() for s in source_allowlist}
        filtered_chunks = [c for c in filtered_chunks if c.source.lower() in allowed]
    if min_credibility > 0:
        filtered_chunks = [c for c in filtered_chunks if c.credibility_weight >= min_credibility]
    if max_age_days is not None:
        filtered_chunks = [
            c for c in filtered_chunks
            if c.recency_days is None or c.recency_days <= max_age_days
        ]
    if not filtered_chunks:
        return []

    method = (retrieval_method or "hybrid").lower()
    if method not in {"hybrid", "tfidf", "bm25", "keyword"}:
        method = "hybrid"

    if method == "bm25":
        base_scores = _bm25_retrieve(query, filtered_chunks, top_k * 4)
    elif method == "keyword":
        base_scores = _keyword_retrieve(query, filtered_chunks, top_k * 4)
    elif method == "tfidf":
        try:
            base_scores = _tfidf_retrieve(query, filtered_chunks, top_k * 4)
        except Exception:
            method = "keyword"
            base_scores = _keyword_retrieve(query, filtered_chunks, top_k * 4)
    else:
        base_scores = _hybrid_lexical_retrieve(query, filtered_chunks, top_k * 4)
    ranked_with_scores = _blend_ranked_results(base_scores, query, method)

    selected_ids: set[str] = set()
    result: list[RAGChunk] = []

    # 1. Always include Bloomberg chunks (up to 3)
    if always_include_bloomberg:
        bloomberg_chunks = [c for c, _ in ranked_with_scores if c.is_bloomberg]
        for bc in bloomberg_chunks[:3]:
            if bc.citation_id not in selected_ids:
                result.append(bc)
                selected_ids.add(bc.citation_id)

    # 2. Fill remaining slots with TF-IDF winners
    for chunk, _score in ranked_with_scores:
        if len(result) >= top_k:
            break
        if chunk.citation_id not in selected_ids:
            result.append(chunk)
            selected_ids.add(chunk.citation_id)

    for rank, chunk in enumerate(result[:top_k], start=1):
        chunk.retrieval_rank = rank
    return result[:top_k]


# ---------------------------------------------------------------------------
# High-level pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end RAG pipeline: ingest → chunk → retrieve.

    Usage::

        pipeline = RAGPipeline(ticker="NVDA", company="NVIDIA")
        pipeline.ingest()
        chunks = pipeline.get_chunks(top_k=8)
    """

    def __init__(
        self,
        ticker: str,
        company: str = "",
        retrieval_query: str = "",
        top_k: int = 8,
        include_rss: bool = True,
        rss_sources: list[str] | None = None,
        source_allowlist: list[str] | None = None,
        min_credibility: float = 0.0,
        max_age_days: int | None = None,
        retrieval_method: str = "hybrid",
        bloomberg_config=None,  # BloombergConfig | None
    ) -> None:
        self.ticker = ticker.upper()
        self.company = company
        self.retrieval_query = retrieval_query or f"{ticker} investment catalyst news analysis"
        self.top_k = top_k
        self.include_rss = include_rss
        self.rss_sources = rss_sources
        self.source_allowlist = source_allowlist
        self.min_credibility = min_credibility
        self.max_age_days = max_age_days
        self.retrieval_method = retrieval_method
        self.bloomberg_config = bloomberg_config
        self._articles: list[NewsArticle] = []
        self._chunks: list[RAGChunk] = []

    def ingest(self) -> "RAGPipeline":
        """Fetch and chunk articles. Returns self for chaining."""
        from src.finance_news_analyzer.news_ingester import ingest_news
        self._articles = ingest_news(
            ticker=self.ticker,
            company=self.company,
            include_rss=self.include_rss,
            rss_sources=self.rss_sources,
            bloomberg_config=self.bloomberg_config,
        )
        self._chunks = chunk_articles(self._articles, self.ticker, self.company)
        return self

    @property
    def article_count(self) -> int:
        return len(self._articles)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def source_names(self) -> list[str]:
        return sorted({a.source for a in self._articles})

    def get_chunks(self, top_k: int | None = None) -> list[RAGChunk]:
        """Return the top-k most relevant chunks for the pipeline query."""
        k = top_k or self.top_k
        return retrieve(
            ticker=self.ticker,
            retrieval_query=self.retrieval_query,
            chunks=self._chunks,
            top_k=k,
            source_allowlist=self.source_allowlist,
            min_credibility=self.min_credibility,
            max_age_days=self.max_age_days,
            retrieval_method=self.retrieval_method,
        )

    def get_all_chunks(self) -> list[RAGChunk]:
        """Return all chunks without ranking (useful for inspection)."""
        return list(self._chunks)
