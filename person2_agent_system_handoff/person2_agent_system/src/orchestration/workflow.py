from __future__ import annotations

from pathlib import Path

from langgraph.graph import END, START, StateGraph

from src.agents.analyst_agent import AnalystAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.strategist_agent import StrategistAgent
from src.llm.base import StructuredLLMClient
from src.models.data_packet import FinalDecision, WorkflowInput
from src.orchestration.citation_validator import collect_known_citations, citations_are_valid
from src.orchestration.memory_store import PatternMemoryStore
from src.orchestration.state_manager import GraphState, initialize_graph_state, snapshot_graph_state
from src.tools.project_tools import citation_audit_tool, market_context_tool, memory_lookup_tool
from src.utils.json_parser import write_json


class MultiAgentWorkflow:
    """LangGraph-based orchestration wrapper for the three-agent pipeline."""

    def __init__(
        self,
        llm_client: StructuredLLMClient | None = None,
        prompt_root: str | Path | None = None,
        memory_store: PatternMemoryStore | None = None,
    ) -> None:
        prompt_base = Path(prompt_root) if prompt_root else Path(__file__).resolve().parents[2] / "prompts"
        self.analyst_agent = AnalystAgent(
            llm_client=llm_client,
            prompt_path=str(prompt_base / "analyst_prompt.txt"),
        )
        self.strategist_agent = StrategistAgent(
            llm_client=llm_client,
            prompt_path=str(prompt_base / "strategist_prompt.txt"),
        )
        self.decision_agent = DecisionAgent(
            llm_client=llm_client,
            prompt_path=str(prompt_base / "decision_prompt.txt"),
        )
        self.memory_store = memory_store
        self.graph = self._build_graph()

    @staticmethod
    def _build_analyst_output(payload: dict):
        from src.models.data_packet import AnalystOutput, EvidenceItem

        return AnalystOutput(
            ticker=payload["ticker"],
            event_summary=payload["event_summary"],
            supporting_evidence=[
                EvidenceItem(claim=item["claim"], citation_ids=item["citation_ids"])
                for item in payload.get("supporting_evidence", [])
            ],
            contradicting_evidence=[
                EvidenceItem(claim=item["claim"], citation_ids=item["citation_ids"])
                for item in payload.get("contradicting_evidence", [])
            ],
            uncertainties=payload.get("uncertainties", []),
            primary_catalysts=payload.get("primary_catalysts", []),
            macro_context=payload.get("macro_context", []),
            staleness_flags=payload.get("staleness_flags", []),
            evidence_balance=payload.get("evidence_balance", "mixed"),
        )

    @staticmethod
    def _build_strategist_output(payload: dict):
        from src.models.data_packet import StrategistOutput

        return StrategistOutput(
            ticker=payload["ticker"],
            direction=payload["direction"],
            horizon=payload["horizon"],
            thesis=payload["thesis"],
            causal_chain=payload["causal_chain"],
            risks=payload.get("risks", []),
            citations=payload.get("citations", []),
            counterarguments=payload.get("counterarguments", []),
            invalidation_conditions=payload.get("invalidation_conditions", []),
            thesis_strength=payload.get("thesis_strength", "medium"),
            market_context_notes=payload.get("market_context_notes", []),
        )

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("analyst", self._analyst_node)
        graph.add_node("market_context", self._market_context_node)
        graph.add_node("strategist", self._strategist_node)
        graph.add_node("citation_audit", self._citation_audit_node)
        graph.add_node("decision", self._decision_node)
        graph.add_node("persist", self._persist_node)
        graph.add_edge(START, "analyst")
        graph.add_edge("analyst", "market_context")
        graph.add_edge("market_context", "strategist")
        graph.add_edge("strategist", "citation_audit")
        graph.add_edge("citation_audit", "decision")
        graph.add_edge("decision", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    def _analyst_node(self, state: GraphState) -> GraphState:
        analyst_output = self.analyst_agent.run(state["workflow_input"])
        return {"analyst_output": analyst_output.to_dict()}

    def _market_context_node(self, state: GraphState) -> GraphState:
        return {"market_context": market_context_tool(state["workflow_input"])}

    def _strategist_node(self, state: GraphState) -> GraphState:
        analyst_payload = state["analyst_output"]
        analyst_output = self._build_analyst_output(analyst_payload)
        strategist_output = self.strategist_agent.run(
            analyst_output,
            memory_notes=state.get("memory_notes", []),
            market_context=state.get("market_context", {}),
        )
        if not citations_are_valid(strategist_output.citations, set(state["known_citations"])):
            raise ValueError("Strategist produced citations that are not in retrieved evidence.")
        return {"strategist_output": strategist_output.to_dict()}

    def _citation_audit_node(self, state: GraphState) -> GraphState:
        return {"citation_audit": citation_audit_tool(state["workflow_input"].chunks)}

    def _decision_node(self, state: GraphState) -> GraphState:
        analyst_payload = state["analyst_output"]
        strategist_payload = state["strategist_output"]

        analyst_output = self._build_analyst_output(analyst_payload)
        strategist_output = self._build_strategist_output(strategist_payload)
        final_output = self.decision_agent.run(
            analyst_output,
            strategist_output,
            memory_notes=state.get("memory_notes", []),
            citation_audit=state.get("citation_audit", {}),
        )
        if not citations_are_valid(final_output.citations, set(state["known_citations"])):
            raise ValueError("Decision output contains invalid citations.")
        return {"final_output": final_output.to_dict()}

    def _persist_node(self, state: GraphState) -> GraphState:
        return state

    def run(self, workflow_input: WorkflowInput, output_dir: str | Path | None = None) -> FinalDecision:
        known_citations = sorted(collect_known_citations(workflow_input.chunks))
        memory_notes = memory_lookup_tool(self.memory_store, workflow_input.ticker) if self.memory_store else []
        initial_state = initialize_graph_state(
            workflow_input=workflow_input,
            memory_notes=memory_notes,
            known_citations=known_citations,
        )
        final_state = self.graph.invoke(initial_state)

        if output_dir is not None:
            output_path = Path(output_dir)
            write_json(output_path / "traces" / f"{workflow_input.ticker}_trace.json", snapshot_graph_state(final_state))
            write_json(output_path / "final_packets" / f"{workflow_input.ticker}_packet.json", final_state["final_output"])
            if self.memory_store:
                final_output = final_state["final_output"]
                self.memory_store.record_preliminary_signal(
                    ticker=final_output["ticker"],
                    query_date=workflow_input.query_date,
                    direction=final_output["direction"],
                    confidence=final_output["confidence"],
                    citations=final_output["citations"],
                )

        final_payload = final_state["final_output"]
        return FinalDecision(**final_payload)
