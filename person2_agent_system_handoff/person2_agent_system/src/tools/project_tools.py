from __future__ import annotations

from dataclasses import asdict

from src.models.data_packet import RetrievedChunk, WorkflowInput
from src.orchestration.citation_validator import collect_known_citations
from src.orchestration.memory_store import PatternMemoryStore


def retrieval_tool(workflow_input: WorkflowInput) -> list[dict]:
    """Expose retrieved chunks in a tool-friendly shape."""
    return [asdict(chunk) for chunk in workflow_input.chunks]


def citation_audit_tool(chunks: list[RetrievedChunk]) -> dict:
    citations = sorted(collect_known_citations(chunks))
    sources = sorted({chunk.source for chunk in chunks})
    stale_candidates = [
        chunk.citation_id
        for chunk in chunks
        if any(token in chunk.text.lower() for token in ["last year", "months ago", "earlier this quarter"])
    ]
    return {
        "known_citations": citations,
        "citation_count": len(citations),
        "source_count": len(sources),
        "sources": sources,
        "stale_candidates": stale_candidates,
    }


def memory_lookup_tool(memory_store: PatternMemoryStore, ticker: str) -> list[str]:
    return memory_store.lookup_notes(ticker)


def market_context_tool(workflow_input: WorkflowInput) -> dict:
    """Derive lightweight market context from retrieved evidence without owning retrieval."""
    positive_tokens = {"beat", "growth", "strong", "surge", "demand", "accelerate", "upbeat"}
    negative_tokens = {"risk", "weak", "decline", "cut", "investigation", "valuation", "downside"}
    macro_tokens = {"fed", "rate", "inflation", "sector", "economy", "macro", "benchmark"}

    positive_hits = 0
    negative_hits = 0
    macro_hits = set()

    for chunk in workflow_input.chunks:
        lowered = chunk.text.lower()
        positive_hits += sum(token in lowered for token in positive_tokens)
        negative_hits += sum(token in lowered for token in negative_tokens)
        for token in macro_tokens:
            if token in lowered:
                macro_hits.add(token)

    if positive_hits > negative_hits:
        market_tone = "supportive"
    elif negative_hits > positive_hits:
        market_tone = "cautious"
    else:
        market_tone = "mixed"

    notes = [
        f"Derived market tone from retrieved evidence is {market_tone}.",
        f"Positive signal hits: {positive_hits}. Negative signal hits: {negative_hits}.",
    ]
    if macro_hits:
        notes.append(f"Macro or sector context detected: {', '.join(sorted(macro_hits))}.")

    return {
        "market_tone": market_tone,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "macro_keywords": sorted(macro_hits),
        "notes": notes,
    }
