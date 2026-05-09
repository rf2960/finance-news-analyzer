from __future__ import annotations

import json

from src.llm.base import StructuredLLMClient
from src.models.data_packet import AnalystOutput, EvidenceItem, WorkflowInput
from src.utils.prompt_loader import load_prompt


class AnalystAgent:
    """Extracts factual evidence from retrieved chunks."""

    def __init__(self, llm_client: StructuredLLMClient | None = None, prompt_path: str | None = None) -> None:
        self.llm_client = llm_client
        self.prompt_path = prompt_path

    def run(self, workflow_input: WorkflowInput) -> AnalystOutput:
        if self.llm_client and self.prompt_path:
            return self._run_with_llm(workflow_input)
        return self._run_heuristic(workflow_input)

    def _run_heuristic(self, workflow_input: WorkflowInput) -> AnalystOutput:
        supporting: list[EvidenceItem] = []
        contradicting: list[EvidenceItem] = []
        primary_catalysts: list[str] = []
        macro_context: list[str] = []
        staleness_flags: list[str] = []

        for chunk in workflow_input.chunks:
            lowered = chunk.text.lower()
            claim = chunk.text.strip().split(".")[0].strip()
            if not claim:
                continue
            item = EvidenceItem(claim=claim, citation_ids=[chunk.citation_id])
            if any(token in lowered for token in ["beat", "growth", "surge", "strong", "demand", "partnership"]):
                supporting.append(item)
                primary_catalysts.append(claim)
            elif any(token in lowered for token in ["miss", "cut", "risk", "weak", "decline", "investigation"]):
                contradicting.append(item)
            else:
                supporting.append(item)
                primary_catalysts.append(claim)

            if any(token in lowered for token in ["fed", "rate", "inflation", "macro", "sector", "economy"]):
                macro_context.append(claim)
            if any(token in lowered for token in ["last year", "months ago", "earlier this quarter"]):
                staleness_flags.append(chunk.citation_id)

        summary = "Mixed signals extracted from retrieved financial news."
        evidence_balance = "mixed"
        if len(supporting) > len(contradicting):
            summary = "Evidence leans positive based on retrieved catalyst mentions."
            evidence_balance = "supporting"
        elif len(contradicting) > len(supporting):
            summary = "Evidence leans negative based on retrieved risk mentions."
            evidence_balance = "contradicting"

        uncertainties = []
        if not workflow_input.chunks:
            uncertainties.append("No retrieved evidence was supplied.")
        if len(workflow_input.chunks) < 2:
            uncertainties.append("Small evidence set may reduce robustness.")

        return AnalystOutput(
            ticker=workflow_input.ticker,
            event_summary=summary,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            uncertainties=uncertainties,
            primary_catalysts=sorted(set(primary_catalysts)),
            macro_context=sorted(set(macro_context)),
            staleness_flags=sorted(set(staleness_flags)),
            evidence_balance=evidence_balance,
        )

    def _run_with_llm(self, workflow_input: WorkflowInput) -> AnalystOutput:
        system_prompt = load_prompt(self.prompt_path)
        user_prompt = self._build_user_prompt(workflow_input)
        schema_name = "analyst_output_schema"
        raw = self.llm_client.invoke_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema_name=schema_name,
        )
        return AnalystOutput(
            ticker=raw["ticker"],
            event_summary=raw["event_summary"],
            supporting_evidence=[
                EvidenceItem(claim=item["claim"], citation_ids=item["citation_ids"])
                for item in raw.get("supporting_evidence", [])
            ],
            contradicting_evidence=[
                EvidenceItem(claim=item["claim"], citation_ids=item["citation_ids"])
                for item in raw.get("contradicting_evidence", [])
            ],
            uncertainties=raw.get("uncertainties", []),
            primary_catalysts=raw.get("primary_catalysts", []),
            macro_context=raw.get("macro_context", []),
            staleness_flags=raw.get("staleness_flags", []),
            evidence_balance=raw.get("evidence_balance", "mixed"),
        )

    def _build_user_prompt(self, workflow_input: WorkflowInput) -> str:
        payload = {
            "ticker": workflow_input.ticker,
            "query_date": workflow_input.query_date,
            "sector": workflow_input.sector,
            "retrieval_query": workflow_input.retrieval_query,
            "chunks": [
                {
                    "citation_id": chunk.citation_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "published_at": chunk.published_at,
                    "ticker": chunk.ticker,
                    "text": chunk.text,
                }
                for chunk in workflow_input.chunks
            ],
        }
        return (
            "Produce only a JSON object that matches the analyst schema.\n"
            "Retrieved context:\n"
            f"{json.dumps(payload, indent=2)}"
        )
