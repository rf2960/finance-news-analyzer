import unittest

from src.agents.decision_agent import DecisionAgent
from src.models.data_packet import AnalystOutput, EvidenceItem, StrategistOutput


class DecisionAgentTests(unittest.TestCase):
    def test_decision_generates_confidence(self) -> None:
        agent = DecisionAgent()
        analyst_output = AnalystOutput(
            ticker="TSLA",
            event_summary="Evidence leans positive.",
            supporting_evidence=[EvidenceItem(claim="Orders increased", citation_ids=["DOC1"])],
            contradicting_evidence=[],
            uncertainties=[],
        )
        strategist_output = StrategistOutput(
            ticker="TSLA",
            direction="bullish",
            horizon="5d",
            thesis="Positive evidence suggests upside.",
            causal_chain="Orders can improve revenue expectations.",
            risks=[],
            citations=["DOC1"],
        )

        result = agent.run(analyst_output, strategist_output)
        self.assertGreaterEqual(result.confidence, 0.5)
        self.assertFalse(result.abstain)


if __name__ == "__main__":
    unittest.main()
