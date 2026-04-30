# FinSight RAG

**Repo name:** `finance-news-analyzer`  
**Demo name:** **FinSight RAG: Evidence-Grounded Market Signals**

FinSight RAG is a course project for generating and evaluating short-term investment ideas from financial news using a multi-agent RAG pipeline. This repository starts with the evaluation pipeline and Streamlit demo UI so the product/evaluation work can proceed before the data ingestion, retrieval, and agent modules are finished.

## Why the UI can be built now

Yes. The evaluation and demo UI can be prepared independently as long as the team agrees on one stable output contract. The UI does not need the final RAG system yet; it only needs example records with the same fields the final Decision Agent will produce:

- `ticker`
- `direction`
- `horizon_days`
- `confidence`
- `reasoning`
- `citations`
- `agent_trace`
- `published_at`

When the Data/Retrieval Lead and Agent/Reasoning Lead finish their parts, they can replace `demo_data/signals.json` with real generated signal packets using the same schema.

## Project Scope

The system follows the project proposal and final plan:

- Generate bullish/bearish short-term signals from financial news.
- Ground every signal in cited retrieved evidence.
- Evaluate signals against 5-trading-day and 20-trading-day forward returns.
- Compare multi-agent RAG against random and sentiment baselines.
- Show reasoning, citations, and aggregate metrics in a Streamlit dashboard.

## Repository Layout

```text
.
|-- app.py
|-- demo_data/
|   |-- prices.csv
|   `-- signals.json
|-- docs/
|   `-- research_notes.md
|-- src/
|   `-- finance_news_analyzer/
|       |-- __init__.py
|       |-- evaluation.py
|       `-- schemas.py
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Run the Demo

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Integration Contract

The final agent layer should write JSON records compatible with `demo_data/signals.json`. The evaluation layer expects market prices in the format shown by `demo_data/prices.csv`.

The current data is intentionally synthetic and only exists to make the UI, metrics, and report visuals ready before full integration.

## Demo Views

- Market Monitor: signal queue, confidence, novelty, sentiment/model disagreement, and market pulse.
- Thesis Workspace: selected signal, quantitative snapshot, price chart, thesis drivers, risks, and watch list.
- Evidence Audit: source cards, credibility weights, and agent handoff path.
- Evaluation Lab: directional hit rate, average signed return, and baseline comparison.

## Suggested Ownership

- Data and Retrieval Lead: news ingestion, article schema, deduplication, embeddings, vector retrieval.
- Agent and Reasoning Lead: analyst/strategist/decision agents, prompts, JSON signal records, citation grounding.
- Evaluation and Product Lead: forward-return labels, baselines, metrics, dashboard, final result interpretation.

