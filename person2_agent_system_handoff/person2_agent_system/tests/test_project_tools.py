import unittest
from pathlib import Path

from src.models.data_packet import RetrievedChunk, WorkflowInput
from src.orchestration.memory_store import PatternMemoryStore
from src.tools.project_tools import citation_audit_tool, market_context_tool, memory_lookup_tool, retrieval_tool


class ProjectToolsTests(unittest.TestCase):
    def test_retrieval_tool_returns_chunk_dicts(self) -> None:
        workflow_input = WorkflowInput(
            ticker="NVDA",
            query_date="2026-05-05",
            chunks=[
                RetrievedChunk(
                    citation_id="DOC1",
                    source="Reuters",
                    title="Demand",
                    published_at="2026-05-05T12:00:00Z",
                    ticker="NVDA",
                    text="Strong demand persists.",
                )
            ],
        )
        result = retrieval_tool(workflow_input)
        self.assertEqual(result[0]["citation_id"], "DOC1")

    def test_citation_audit_tool_counts_citations(self) -> None:
        chunks = [
            RetrievedChunk(
                citation_id="DOC1",
                source="Reuters",
                title="Demand",
                published_at="2026-05-05T12:00:00Z",
                ticker="NVDA",
                text="Strong demand persists.",
            )
        ]
        result = citation_audit_tool(chunks)
        self.assertEqual(result["citation_count"], 1)
        self.assertEqual(result["source_count"], 1)

    def test_memory_lookup_tool_reads_notes(self) -> None:
        memory_path = Path(__file__).resolve().parents[1] / "outputs" / "memory" / "tool_memory.json"
        if memory_path.exists():
            memory_path.unlink()
        store = PatternMemoryStore(memory_path)
        store.record_preliminary_signal("NVDA", "2026-05-05", "bullish", 0.7, ["DOC1"])
        notes = memory_lookup_tool(store, "NVDA")
        self.assertTrue(notes)

    def test_market_context_tool_summarizes_tone(self) -> None:
        workflow_input = WorkflowInput(
            ticker="NVDA",
            query_date="2026-05-05",
            chunks=[
                RetrievedChunk(
                    citation_id="DOC1",
                    source="Reuters",
                    title="Demand",
                    published_at="2026-05-05T12:00:00Z",
                    ticker="NVDA",
                    text="Strong demand persists but valuation risk remains elevated.",
                )
            ],
        )
        result = market_context_tool(workflow_input)
        self.assertIn(result["market_tone"], {"supportive", "cautious", "mixed"})
        self.assertTrue(result["notes"])


if __name__ == "__main__":
    unittest.main()
