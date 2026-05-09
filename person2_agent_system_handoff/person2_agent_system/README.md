# Person 2 Agent System

This folder contains the Person 2 multi-agent reasoning layer for the project.

## Goal

Transform retrieved financial news evidence into a grounded investment idea with:

- ticker
- direction
- horizon
- confidence
- reasoning
- citations

## What This Owns

This folder owns the agent-side reasoning system:

- LangGraph workflow
- Analyst, Strategist, and Decision agents
- citation audit, memory lookup, and market context tools
- prompt templates
- JSON output schemas
- traces and final investment packets

This folder does not own:

- news ingestion
- article cleaning
- embeddings
- vector database setup
- retrieval ranking over the full news corpus

Those belong to Person 1's RAG pipeline.

## Agent Design

### 1. Analyst Agent

- reads retrieved articles/chunks
- extracts key events
- separates supporting and contradicting evidence
- does not make the final trade call

### 2. Strategist Agent

- converts evidence into an investment thesis
- proposes direction and horizon
- explains the causal chain and risks

### 3. Decision Agent

- validates consistency
- checks citation coverage
- assigns confidence
- returns the final structured packet

## LangGraph Flow

The default graph is:

- `analyst`
- `market_context`
- `strategist`
- `citation_audit`
- `decision`
- `persist`

## Folder Guide

- `prompts/`: prompt templates for each agent
- `schemas/`: output contracts for each step
- `src/agents/`: individual agent logic
- `src/orchestration/`: workflow and validation
- `src/tools/`: tool logic used by the graph
- `examples/`: sample inputs and outputs
- `playground/`: local integration and live-case runners
- `tests/`: test suite

## Current Status

Current capabilities:

- LangGraph-based multi-agent pipeline
- deterministic mode and live OpenAI-backed mode
- grounded JSON outputs for all 3 agent stages
- citation audit, memory lookup, and market context tools
- disagreement detection
- local `.env` support through `.env.example`
- sample live-case runner and test cases

## Integration Contract For Person 1

Person 1 should connect retrieval output into `WorkflowInput` defined in [data_packet.py](/Users/andrewchen/Desktop/Columbia/Genai with llm/Project/person2_agent_system/src/models/data_packet.py).

Required `WorkflowInput` fields:

- `ticker`
- `query_date`
- `chunks`
- optional: `sector`
- optional: `retrieval_query`

Each retrieved chunk should have:

- `citation_id`
- `source`
- `title`
- `published_at`
- `ticker`
- `text`

This is the clean handoff boundary:

- Person 1 returns retrieved chunks in this format
- Person 2 workflow consumes them and produces structured investment outputs

## Main Entry Points

- single demo run: [src/main.py](/Users/andrewchen/Desktop/Columbia/Genai with llm/Project/person2_agent_system/src/main.py)
- multi-case live runner: [run_live_cases.py](/Users/andrewchen/Desktop/Columbia/Genai with llm/Project/person2_agent_system/playground/run_live_cases.py)
- mock retrieval smoke test: [retrieval_smoke_test.py](/Users/andrewchen/Desktop/Columbia/Genai with llm/Project/person2_agent_system/playground/retrieval_smoke_test.py)
- LangGraph workflow: [workflow.py](/Users/andrewchen/Desktop/Columbia/Genai with llm/Project/person2_agent_system/src/orchestration/workflow.py)

## Run

```bash
python3 -m src.main
```

To try the separate retrieval harness:

```bash
python3 -m playground.retrieval_smoke_test
```

## Test

```bash
python3 -m unittest discover -s tests
```

## Environment Setup

This handoff does not include a real `.env` file.

If Person 1 or another teammate wants to enable live OpenAI-backed runs, create a local `.env` file in this folder.

Example:

```bash
cp .env.example .env
```

Then fill in:

- `AGENT_BACKEND=langchain_openai`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o-mini`

The code now auto-loads `.env` if it exists.

## Handoff Steps For Person 1

1. Keep your own retrieval pipeline separate.
2. Map your retrieval output into `WorkflowInput`.
3. Replace the sample/mock input path with your real retrieved chunks.
4. Run:
   `python3 -m unittest discover -s tests`
5. Run:
   `python3 -m playground.run_live_cases`
6. Inspect:
   - `outputs/traces`
   - `outputs/final_packets`
7. If using live OpenAI mode, create your own local `.env` from `.env.example`.

## Tool Calling Fit

Yes, tool calling can fit this project well. The most natural tools are:

- citation audit tool: verify allowed citation ids before final output
- memory lookup tool: surface prior signal patterns for the ticker
- market context tool: summarize supportive vs cautious evidence context

Starter tool-shaped functions live in [project_tools.py](/Users/andrewchen/Desktop/Columbia/Genai with llm/Project/person2_agent_system/src/tools/project_tools.py).

## Recommended Next Work

- plug in Person 1's real retrieval output
- run live OpenAI cases on real financial news
- tune prompts for confidence, horizon choice, and abstention
- build the final Streamlit demo around traces and final packets
