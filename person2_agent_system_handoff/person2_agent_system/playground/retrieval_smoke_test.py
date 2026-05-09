from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from src.models.data_packet import RetrievedChunk, WorkflowInput
from src.orchestration.memory_store import PatternMemoryStore
from src.orchestration.workflow import MultiAgentWorkflow


SAMPLE_CORPUS = [
    {
        "citation_id": "DOC100",
        "source": "Reuters",
        "title": "NVIDIA supplier demand expands",
        "published_at": "2026-05-04T14:00:00Z",
        "ticker": "NVDA",
        "text": "NVIDIA suppliers reported strong AI server demand and improving enterprise orders.",
    },
    {
        "citation_id": "DOC101",
        "source": "Yahoo Finance",
        "title": "Valuation worries remain",
        "published_at": "2026-05-04T15:00:00Z",
        "ticker": "NVDA",
        "text": "Some investors warned that valuation risk remains elevated after the recent surge.",
    },
    {
        "citation_id": "DOC102",
        "source": "CNBC",
        "title": "Microsoft cloud growth accelerates",
        "published_at": "2026-05-04T14:20:00Z",
        "ticker": "MSFT",
        "text": "Microsoft posted strong cloud growth and upbeat enterprise commentary.",
    },
]


def retrieve_chunks(query: str, ticker: str, top_k: int = 3) -> list[RetrievedChunk]:
    query_terms = set(query.lower().split())
    scored = []
    for row in SAMPLE_CORPUS:
        if row["ticker"] != ticker:
            continue
        text_terms = set((row["title"] + " " + row["text"]).lower().split())
        overlap = len(query_terms & text_terms)
        scored.append((overlap, row))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [RetrievedChunk(**row) for score, row in scored[:top_k] if score > 0]


def build_workflow_input() -> WorkflowInput:
    query = "NVDA AI demand valuation risk"
    chunks = retrieve_chunks(query=query, ticker="NVDA")
    return WorkflowInput(
        ticker="NVDA",
        query_date="2026-05-04",
        chunks=chunks,
        sector="Semiconductors",
        retrieval_query=query,
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow_input = build_workflow_input()
    memory_store = PatternMemoryStore(root / "outputs" / "memory" / "signal_memory.json")
    workflow = MultiAgentWorkflow(memory_store=memory_store)
    result = workflow.run(workflow_input, output_dir=root / "outputs")
    print("=== Retrieved Chunks ===")
    print(json.dumps([asdict(chunk) for chunk in workflow_input.chunks], indent=2))
    print("\n=== Final Decision ===")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
