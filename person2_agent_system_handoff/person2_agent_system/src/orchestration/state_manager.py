from __future__ import annotations

from dataclasses import asdict
from typing import Any, TypedDict

from src.models.data_packet import WorkflowInput


class GraphState(TypedDict, total=False):
    workflow_input: WorkflowInput
    ticker: str
    analyst_output: dict[str, Any]
    strategist_output: dict[str, Any]
    final_output: dict[str, Any]
    market_context: dict[str, Any]
    citation_audit: dict[str, Any]
    known_citations: list[str]
    memory_notes: list[str]


def initialize_graph_state(workflow_input: WorkflowInput, memory_notes: list[str], known_citations: list[str]) -> GraphState:
    return {
        "workflow_input": workflow_input,
        "ticker": workflow_input.ticker,
        "analyst_output": {},
        "strategist_output": {},
        "final_output": {},
        "market_context": {},
        "citation_audit": {},
        "known_citations": known_citations,
        "memory_notes": memory_notes,
    }


def snapshot_graph_state(state: GraphState) -> dict[str, Any]:
    workflow_input = state["workflow_input"]
    return {
        "ticker": state["ticker"],
        "workflow_input": asdict(workflow_input),
        "analyst_output": state.get("analyst_output", {}),
        "strategist_output": state.get("strategist_output", {}),
        "final_output": state.get("final_output", {}),
        "market_context": state.get("market_context", {}),
        "citation_audit": state.get("citation_audit", {}),
        "known_citations": state.get("known_citations", []),
        "memory_notes": state.get("memory_notes", []),
    }
