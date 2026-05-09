from __future__ import annotations

from src.models.data_packet import AnalystOutput, StrategistOutput


def detect_disagreement(analyst_output: AnalystOutput, strategist_output: StrategistOutput) -> tuple[bool, str]:
    contradiction_count = len(analyst_output.contradicting_evidence)
    support_count = len(analyst_output.supporting_evidence)

    if contradiction_count == 0:
        return False, ""
    if strategist_output.direction == "bullish" and contradiction_count >= support_count:
        return True, "Bullish thesis despite equal or greater contradicting evidence."
    if strategist_output.direction == "bearish" and support_count >= contradiction_count:
        return True, "Bearish thesis despite equal or greater supporting evidence."
    if strategist_output.direction != "neutral" and analyst_output.evidence_balance == "mixed":
        return True, "Directional thesis issued even though evidence balance is mixed."
    return False, ""
