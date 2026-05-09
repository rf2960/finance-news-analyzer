from __future__ import annotations

import json
from pathlib import Path

from src.llm.factory import build_llm_client_from_env
from src.models.data_packet import RetrievedChunk, WorkflowInput
from src.orchestration.memory_store import PatternMemoryStore
from src.orchestration.workflow import MultiAgentWorkflow


def load_cases(path: Path) -> list[WorkflowInput]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[WorkflowInput] = []
    for item in raw:
        chunks = [RetrievedChunk(**chunk) for chunk in item["chunks"]]
        cases.append(
            WorkflowInput(
                ticker=item["ticker"],
                query_date=item["query_date"],
                chunks=chunks,
                sector=item.get("sector", ""),
                retrieval_query=item.get("retrieval_query", ""),
            )
        )
    return cases


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    cases_path = root / "examples" / "live_test_cases.json"
    llm_client = build_llm_client_from_env(schema_root=root / "schemas")
    memory_store = PatternMemoryStore(root / "outputs" / "memory" / "signal_memory.json")
    workflow = MultiAgentWorkflow(llm_client=llm_client, memory_store=memory_store)

    cases = load_cases(cases_path)
    print(f"Running {len(cases)} cases from {cases_path}")
    for case in cases:
        result = workflow.run(case, output_dir=root / "outputs")
        print(f"\n=== {case.ticker} | {case.query_date} ===")
        print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
