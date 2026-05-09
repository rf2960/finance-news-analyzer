from __future__ import annotations

from src.models.data_packet import AnalystOutput, FinalDecision, StrategistOutput


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, round(score, 2)))


def score_decision(analyst: AnalystOutput, strategist: StrategistOutput) -> float:
    score = 0.35
    score += min(len(analyst.supporting_evidence), 3) * 0.12
    score -= min(len(analyst.contradicting_evidence), 2) * 0.10
    if strategist.citations:
        score += 0.08
    if not strategist.causal_chain.strip():
        score -= 0.15
    if strategist.thesis_strength == "high":
        score += 0.05
    if analyst.evidence_balance == "mixed":
        score -= 0.05
    return clamp_score(score)


def should_abstain(decision: FinalDecision) -> bool:
    if decision.confidence < 0.45:
        return True
    if len(decision.citations) == 0:
        return True
    if len(decision.reasoning.strip()) < 20:
        return True
    return False
