import unittest

from src.agents.analyst_agent import AnalystAgent
from src.models.data_packet import RetrievedChunk, WorkflowInput


class AnalystAgentTests(unittest.TestCase):
    def test_analyst_extracts_supporting_evidence(self) -> None:
        agent = AnalystAgent()
        workflow_input = WorkflowInput(
            ticker="MSFT",
            query_date="2026-05-01",
            chunks=[
                RetrievedChunk(
                    citation_id="DOC1",
                    source="Reuters",
                    title="Cloud growth strong",
                    published_at="2026-05-01T10:00:00Z",
                    ticker="MSFT",
                    text="Cloud growth remained strong and enterprise demand improved.",
                )
            ],
        )

        result = agent.run(workflow_input)
        self.assertEqual(result.ticker, "MSFT")
        self.assertEqual(len(result.supporting_evidence), 1)


if __name__ == "__main__":
    unittest.main()
