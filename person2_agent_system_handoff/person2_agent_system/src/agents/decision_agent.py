from __future__ import annotations

import json

from src.llm.base import StructuredLLMClient
from src.models.data_packet import AnalystOutput, FinalDecision, StrategistOutput
from src.orchestration.disagreement_detector import detect_disagreement
from src.utils.confidence_scoring import score_decision, should_abstain
from src.utils.prompt_loader import load_prompt


class DecisionAgent:
    """Validates and formats the final grounded investment packet."""

    def __init__(self, llm_client: StructuredLLMClient | None = None, prompt_path: str | None = None) -> None:
        self.llm_client = llm_client
        self.prompt_path = prompt_path

    def run(
        self,
        analyst_output: AnalystOutput,
        strategist_output: StrategistOutput,
        memory_notes: list[str] | None = None,
        citation_audit: dict | None = None,
    ) -> FinalDecision:
        if self.llm_client and self.prompt_path:
            return self._run_with_llm(
                analyst_output,
                strategist_output,
                memory_notes or [],
                citation_audit or {},
            )
        return self._run_heuristic(
            analyst_output,
            strategist_output,
            memory_notes or [],
            citation_audit or {},
        )

    def _run_heuristic(
        self,
        analyst_output: AnalystOutput,
        strategist_output: StrategistOutput,
        memory_notes: list[str],
        citation_audit: dict,
    ) -> FinalDecision:
        confidence = score_decision(analyst_output, strategist_output)
        reasoning = (
            f"{strategist_output.thesis} {strategist_output.causal_chain} "
            f"Analyst summary: {analyst_output.event_summary}"
        ).strip()
        disagreement_signal, disagreement_reason = detect_disagreement(analyst_output, strategist_output)
        validation_notes = [
            "Final decision references only retrieved citations.",
            "Confidence is reduced when contradictory evidence is present.",
        ]
        if citation_audit:
            validation_notes.append(
                f"Citation audit confirmed {citation_audit.get('citation_count', 0)} citations across "
                f"{citation_audit.get('source_count', 0)} sources."
            )
            if citation_audit.get("stale_candidates"):
                validation_notes.append("Citation audit flagged potentially stale evidence for caution.")
        if strategist_output.counterarguments:
            validation_notes.append("Counterarguments were preserved in the strategist output.")

        decision = FinalDecision(
            ticker=strategist_output.ticker,
            direction=strategist_output.direction,
            horizon=strategist_output.horizon,
            confidence=confidence,
            reasoning=reasoning,
            risks=strategist_output.risks,
            citations=strategist_output.citations,
            abstain=False,
            validation_notes=validation_notes,
            disagreement_signal=disagreement_signal,
            disagreement_reason=disagreement_reason,
            memory_notes=memory_notes,
            market_context_notes=strategist_output.market_context_notes,
        )
        decision.abstain = should_abstain(decision)
        return decision

    def _run_with_llm(
        self,
        analyst_output: AnalystOutput,
        strategist_output: StrategistOutput,
        memory_notes: list[str],
        citation_audit: dict,
    ) -> FinalDecision:
        system_prompt = load_prompt(self.prompt_path)
        user_prompt = self._build_user_prompt(
            analyst_output,
            strategist_output,
            memory_notes,
            citation_audit,
        )
        raw = self.llm_client.invoke_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="final_decision_schema",
        )
        disagreement_signal, disagreement_reason = detect_disagreement(analyst_output, strategist_output)
        decision = FinalDecision(
            ticker=raw["ticker"],
            direction=raw["direction"],
            horizon=raw["horizon"],
            confidence=raw["confidence"],
            reasoning=raw["reasoning"],
            risks=raw.get("risks", []),
            citations=raw.get("citations", []),
            abstain=raw.get("abstain", False),
            validation_notes=raw.get("validation_notes", []),
            disagreement_signal=raw.get("disagreement_signal", disagreement_signal),
            disagreement_reason=raw.get("disagreement_reason", disagreement_reason),
            memory_notes=raw.get("memory_notes", memory_notes),
            market_context_notes=raw.get("market_context_notes", strategist_output.market_context_notes),
        )
        decision.abstain = should_abstain(decision)
        return decision

    def _build_user_prompt(
        self,
        analyst_output: AnalystOutput,
        strategist_output: StrategistOutput,
        memory_notes: list[str],
        citation_audit: dict,
    ) -> str:
        payload = {
            "analyst_output": {
                "ticker": analyst_output.ticker,
                "event_summary": analyst_output.event_summary,
                "supporting_evidence": [
                    {"claim": item.claim, "citation_ids": item.citation_ids}
                    for item in analyst_output.supporting_evidence
                ],
                "contradicting_evidence": [
                    {"claim": item.claim, "citation_ids": item.citation_ids}
                    for item in analyst_output.contradicting_evidence
                ],
                "uncertainties": analyst_output.uncertainties,
                "primary_catalysts": analyst_output.primary_catalysts,
                "macro_context": analyst_output.macro_context,
                "staleness_flags": analyst_output.staleness_flags,
                "evidence_balance": analyst_output.evidence_balance,
            },
            "strategist_output": {
                "ticker": strategist_output.ticker,
                "direction": strategist_output.direction,
                "horizon": strategist_output.horizon,
                "thesis": strategist_output.thesis,
                "causal_chain": strategist_output.causal_chain,
                "risks": strategist_output.risks,
                "citations": strategist_output.citations,
                "counterarguments": strategist_output.counterarguments,
                "invalidation_conditions": strategist_output.invalidation_conditions,
                "thesis_strength": strategist_output.thesis_strength,
                "market_context_notes": strategist_output.market_context_notes,
            },
            "memory_notes": memory_notes,
            "citation_audit": citation_audit,
        }
        return (
            "Produce only a JSON object that matches the final decision schema.\n"
            "Validate the strategist output against the analyst evidence package:\n"
            f"{json.dumps(payload, indent=2)}"
        )
