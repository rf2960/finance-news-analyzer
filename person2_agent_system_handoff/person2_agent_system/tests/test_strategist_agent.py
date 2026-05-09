import unittest

from src.agents.strategist_agent import StrategistAgent
from src.models.data_packet import AnalystOutput, EvidenceItem


class StrategistAgentTests(unittest.TestCase):
    def test_strategist_returns_direction_and_citations(self) -> None:
        agent = StrategistAgent()
        analyst_output = AnalystOutput(
            ticker="AAPL",
            event_summary="Positive catalyst mix.",
            supporting_evidence=[EvidenceItem(claim="Demand improved", citation_ids=["DOC1"])],
            contradicting_evidence=[],
            uncertainties=[],
        )

        result = agent.run(analyst_output)
        self.assertEqual(result.direction, "bullish")
        self.assertIn("DOC1", result.citations)


if __name__ == "__main__":
    unittest.main()
