# FinSight RAG Application Guide

This folder contains the runnable Streamlit app, command-line runner, evaluation data, and core Python modules for FinSight RAG.

For the polished project overview, report links, and portfolio-facing description, see the repository-level [README](../README.md).

## Run the App

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Optional Windows launcher:

```bash
start.bat
```

## Environment Variables

Copy `.env.example` to `.env` if you want local environment configuration.

| Variable | Required | Purpose |
|---|---:|---|
| `OPENAI_API_KEY` | No | Enables LLM-backed agents. Leave blank for heuristic mode. |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini`. |
| `AGENT_BACKEND` | No | `heuristic` or `langchain_openai`. |
| `BLOOMBERG_HOST` | No | Bloomberg B-PIPE host, if available. |
| `BLOOMBERG_PORT` | No | Bloomberg B-PIPE port, usually `8194`. |

The project runs without an API key in heuristic mode.

## Main Commands

Run ticker analysis from the CLI:

```bash
python run_analysis.py --ticker NVDA --verbose
```

Run with an OpenAI-backed agent workflow:

```bash
python run_analysis.py --ticker NVDA --openai-key sk-your-key
```

Save a signal packet:

```bash
python run_analysis.py --ticker AAPL --save-signal
```

Run the RAG quality evaluation:

```bash
python test_rag_quality.py --run-all-tests --save-results demo_data/rag_eval_results.json
```

## Data Files

| Path | Purpose |
|---|---|
| `demo_data/signals.json` | Demo and generated signal packets. |
| `demo_data/evaluation_signals.json` | Historical sample used for report/demo evaluation. |
| `demo_data/prices.csv` | Bundled price data for forward-return diagnostics. |
| `demo_data/rag_eval_results.json` | Latest RAG quality test output. |

## Core Modules

| Module | Role |
|---|---|
| `agent_runner.py` | Coordinates retrieval, enrichment, heuristic mode, and optional LLM mode. |
| `rag_pipeline.py` | Builds TF-IDF retrieval over normalized news chunks. |
| `news_ingester.py` | Collects and normalizes news from public sources. |
| `technical_factors.py` | Computes RSI, MACD, moving averages, volatility, and momentum context. |
| `macro_events.py` | Adds macro and geopolitical event context. |
| `stock_screener.py` | Supports market scan and ticker discovery. |
| `evaluation.py` | Attaches forward returns and computes baseline metrics. |
| `schemas.py` | Defines the structured signal packet and citation models. |

## Output Contract

The final signal packet includes:

- ticker, company, sector, benchmark
- direction and horizon
- confidence, novelty, sentiment, and source-quality scores
- reasoning, catalyst, thesis bullets, risk factors, counter-evidence, watch items
- market snapshot
- citations
- agent trace
- random and keyword-sentiment baselines

## Notes for Reviewers

- The Streamlit app is the primary interface.
- The CLI runner is useful for reproducible single-ticker checks.
- The current report metrics use a bundled historical demo sample plus RAG quality tests. A larger live-data evaluation is listed as future work.
