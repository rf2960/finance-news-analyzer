import unittest

from src.models.data_packet import AnalystOutput, EvidenceItem, StrategistOutput
from src.orchestration.disagreement_detector import detect_disagreement


class DisagreementDetectorTests(unittest.TestCase):
    def test_detects_mixed_evidence_directional_call(self) -> None:
        analyst_output = AnalystOutput(
            ticker="NVDA",
            event_summary="Mixed evidence.",
            supporting_evidence=[EvidenceItem(claim="Demand is strong", citation_ids=["DOC1"])],
            contradicting_evidence=[EvidenceItem(claim="Valuation risk is high", citation_ids=["DOC2"])],
            evidence_balance="mixed",
        )
        strategist_output = StrategistOutput(
            ticker="NVDA",
            direction="bullish",
            horizon="5d",
            thesis="Upside remains likely.",
            causal_chain="Demand may outweigh valuation concerns.",
            citations=["DOC1", "DOC2"],
        )

        disagreement, reason = detect_disagreement(analyst_output, strategist_output)
        self.assertTrue(disagreement)
        self.assertTrue(reason)


if __name__ == "__main__":
    unittest.main()
