import unittest
from pathlib import Path

from src.models.data_packet import RetrievedChunk, WorkflowInput
from src.orchestration.workflow import MultiAgentWorkflow


class WorkflowTests(unittest.TestCase):
    def test_workflow_runs_end_to_end(self) -> None:
        workflow = MultiAgentWorkflow()
        workflow_input = WorkflowInput(
            ticker="NVDA",
            query_date="2026-05-01",
            chunks=[
                RetrievedChunk(
                    citation_id="DOC1",
                    source="Yahoo Finance",
                    title="Strong demand",
                    published_at="2026-05-01T14:00:00Z",
                    ticker="NVDA",
                    text="Demand remained strong and enterprise orders improved.",
                )
            ],
        )
        output_dir = Path(__file__).resolve().parents[1] / "outputs"
        result = workflow.run(workflow_input, output_dir=output_dir)
        self.assertEqual(result.ticker, "NVDA")
        self.assertIn("DOC1", result.citations)


if __name__ == "__main__":
    unittest.main()
