from __future__ import annotations

import json

from src.llm.base import StructuredLLMClient
from src.models.data_packet import AnalystOutput, StrategistOutput
from src.utils.prompt_loader import load_prompt


class StrategistAgent:
    """Converts extracted evidence into an investment thesis."""

    def __init__(self, llm_client: StructuredLLMClient | None = None, prompt_path: str | None = None) -> None:
        self.llm_client = llm_client
        self.prompt_path = prompt_path

    def run(
        self,
        analyst_output: AnalystOutput,
        memory_notes: list[str] | None = None,
        market_context: dict | None = None,
    ) -> StrategistOutput:
        if self.llm_client and self.prompt_path:
            return self._run_with_llm(analyst_output, memory_notes or [], market_context or {})
        return self._run_heuristic(analyst_output, market_context or {})

    def _run_heuristic(self, analyst_output: AnalystOutput, market_context: dict) -> StrategistOutput:
        support_count = len(analyst_output.supporting_evidence)
        contradiction_count = len(analyst_output.contradicting_evidence)
        market_tone = market_context.get("market_tone", "mixed")

        if support_count > contradiction_count:
            direction = "bullish"
            thesis = "Recent retrieved evidence suggests a favorable near-term catalyst profile."
            thesis_strength = "medium"
        elif contradiction_count > support_count:
            direction = "bearish"
            thesis = "Recent retrieved evidence suggests downside pressure or elevated near-term risk."
            thesis_strength = "medium"
        else:
            direction = "neutral"
            thesis = "Retrieved evidence is balanced and does not support a strong directional view."
            thesis_strength = "low"

        if market_tone == "supportive" and direction == "neutral":
            thesis = "Evidence is balanced, but broader retrieved context leans supportive."
        elif market_tone == "cautious" and direction == "bullish":
            thesis = "Company-specific evidence is favorable, though broader retrieved context remains cautious."

        citations = []
        for item in analyst_output.supporting_evidence + analyst_output.contradicting_evidence:
            citations.extend(item.citation_ids)

        causal_chain = (
            "The thesis is derived from extracted news catalysts, then translated into a "
            "short-horizon market view while accounting for contradicting evidence."
        )

        risks = list(analyst_output.uncertainties)
        if contradiction_count:
            risks.append("Conflicting evidence may weaken the signal.")
        if analyst_output.staleness_flags:
            risks.append("Some supporting context may be stale relative to the query date.")
        if market_tone == "cautious" and direction == "bullish":
            risks.append("Broader retrieved market context is cautious despite positive company-specific signals.")

        counterarguments = [item.claim for item in analyst_output.contradicting_evidence]
        invalidation_conditions = [
            "New company-specific news contradicts the current catalyst view.",
            "Market reaction diverges materially from the cited catalyst narrative.",
        ]
        market_context_notes = market_context.get("notes", [])

        return StrategistOutput(
            ticker=analyst_output.ticker,
            direction=direction,
            horizon="5d",
            thesis=thesis,
            causal_chain=causal_chain,
            risks=risks,
            citations=sorted(set(citations)),
            counterarguments=counterarguments,
            invalidation_conditions=invalidation_conditions,
            thesis_strength=thesis_strength,
            market_context_notes=market_context_notes,
        )

    def _run_with_llm(
        self,
        analyst_output: AnalystOutput,
        memory_notes: list[str],
        market_context: dict,
    ) -> StrategistOutput:
        system_prompt = load_prompt(self.prompt_path)
        user_prompt = self._build_user_prompt(analyst_output, memory_notes, market_context)
        raw = self.llm_client.invoke_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name="strategist_output_schema",
        )
        return StrategistOutput(
            ticker=raw["ticker"],
            direction=raw["direction"],
            horizon=raw["horizon"],
            thesis=raw["thesis"],
            causal_chain=raw["causal_chain"],
            risks=raw.get("risks", []),
            citations=raw.get("citations", []),
            counterarguments=raw.get("counterarguments", []),
            invalidation_conditions=raw.get("invalidation_conditions", []),
            thesis_strength=raw.get("thesis_strength", "medium"),
            market_context_notes=raw.get("market_context_notes", market_context.get("notes", [])),
        )

    def _build_user_prompt(self, analyst_output: AnalystOutput, memory_notes: list[str], market_context: dict) -> str:
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
            "memory_notes": memory_notes,
            "market_context": market_context,
        }
        return (
            "Produce only a JSON object that matches the strategist schema.\n"
            "Analyst evidence package:\n"
            f"{json.dumps(payload, indent=2)}"
        )
