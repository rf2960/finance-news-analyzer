from __future__ import annotations

import json
from pathlib import Path

from src.llm.factory import build_llm_client_from_env
from src.models.data_packet import RetrievedChunk, WorkflowInput
from src.orchestration.memory_store import PatternMemoryStore
from src.orchestration.workflow import MultiAgentWorkflow


def build_demo_input() -> WorkflowInput:
    chunks = [
        RetrievedChunk(
            citation_id="DOC1",
            source="Yahoo Finance",
            title="NVDA demand remains strong",
            published_at="2026-05-01T14:00:00Z",
            ticker="NVDA",
            text="AI chip demand remains strong and management highlighted healthy enterprise orders.",
        ),
        RetrievedChunk(
            citation_id="DOC2",
            source="Reuters",
            title="Investors watch valuation risk",
            published_at="2026-05-01T15:30:00Z",
            ticker="NVDA",
            text="Some analysts warned that valuation risk could limit upside after the recent surge.",
        ),
    ]
    return WorkflowInput(
        ticker="NVDA",
        query_date="2026-05-01",
        chunks=chunks,
        sector="Semiconductors",
        retrieval_query="NVDA AI demand and valuation risk",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    llm_client = build_llm_client_from_env(schema_root=root / "schemas")
    memory_store = PatternMemoryStore(root / "outputs" / "memory" / "signal_memory.json")
    workflow = MultiAgentWorkflow(llm_client=llm_client, memory_store=memory_store)
    workflow_input = build_demo_input()
    result = workflow.run(workflow_input, output_dir=root / "outputs")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
