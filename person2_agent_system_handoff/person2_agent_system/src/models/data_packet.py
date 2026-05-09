from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class RetrievedChunk:
    citation_id: str
    source: str
    title: str
    published_at: str
    ticker: str
    text: str


@dataclass
class WorkflowInput:
    ticker: str
    query_date: str
    chunks: List[RetrievedChunk]
    sector: str = ""
    retrieval_query: str = ""


@dataclass
class EvidenceItem:
    claim: str
    citation_ids: List[str]


@dataclass
class AnalystOutput:
    ticker: str
    event_summary: str
    supporting_evidence: List[EvidenceItem] = field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    primary_catalysts: List[str] = field(default_factory=list)
    macro_context: List[str] = field(default_factory=list)
    staleness_flags: List[str] = field(default_factory=list)
    evidence_balance: str = "mixed"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrategistOutput:
    ticker: str
    direction: str
    horizon: str
    thesis: str
    causal_chain: str
    risks: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    counterarguments: List[str] = field(default_factory=list)
    invalidation_conditions: List[str] = field(default_factory=list)
    thesis_strength: str = "medium"
    market_context_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FinalDecision:
    ticker: str
    direction: str
    horizon: str
    confidence: float
    reasoning: str
    risks: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    abstain: bool = False
    validation_notes: List[str] = field(default_factory=list)
    disagreement_signal: bool = False
    disagreement_reason: str = ""
    memory_notes: List[str] = field(default_factory=list)
    market_context_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
