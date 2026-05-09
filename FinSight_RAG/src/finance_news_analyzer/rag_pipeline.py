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

import math
import re
from dataclasses import dataclass
from typing import List, Optional

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
    credibility_weight: float = 0.65
    is_bloomberg: bool = False   # High-authority flag

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


def chunk_articles(
    articles: list[NewsArticle],
    ticker: str,
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
    import numpy as np  # type: ignore

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

    # Blend TF-IDF similarity with credibility weight
    blended = [
        (chunks[i], float(sims[i]) * 0.75 + chunks[i].credibility_weight * 0.25)
        for i in range(len(chunks))
    ]
    blended.sort(key=lambda x: x[1], reverse=True)
    return blended[:top_k]


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
        norm_hits = hits / max(len(keywords), 1)
        return norm_hits * 0.75 + chunk.credibility_weight * 0.25

    ranked = sorted(chunks, key=_score, reverse=True)
    scored = [(_c, _score(_c)) for _c in ranked[:top_k]]
    return scored


def retrieve(
    ticker: str,
    retrieval_query: str,
    chunks: list[RAGChunk],
    top_k: int = 8,
    always_include_bloomberg: bool = True,
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

    # --- Rank all chunks ---
    try:
        ranked_with_scores = _tfidf_retrieve(query, chunks, top_k * 2)
    except Exception:
        ranked_with_scores = _keyword_retrieve(query, chunks, top_k * 2)

    selected_ids: set[str] = set()
    result: list[RAGChunk] = []

    # 1. Always include Bloomberg chunks (up to 3)
    if always_include_bloomberg:
        bloomberg_chunks = [c for c in chunks if c.is_bloomberg]
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
        bloomberg_config=None,  # BloombergConfig | None
    ) -> None:
        self.ticker = ticker.upper()
        self.company = company
        self.retrieval_query = retrieval_query or f"{ticker} investment catalyst news analysis"
        self.top_k = top_k
        self.include_rss = include_rss
        self.rss_sources = rss_sources
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
        self._chunks = chunk_articles(self._articles, self.ticker)
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
        )

    def get_all_chunks(self) -> list[RAGChunk]:
        """Return all chunks without ranking (useful for inspection)."""
        return list(self._chunks)
