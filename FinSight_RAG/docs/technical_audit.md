# FinSight RAG Technical Audit

This audit summarizes the current system design, gaps found during review, and the portfolio-oriented improvements implemented in this pass.

## Current Architecture

FinSight RAG is a financial-news research application with four major layers:

1. **News ingestion**: pulls ticker-specific Yahoo Finance news, Google News RSS, public financial RSS feeds, and optional Bloomberg B-PIPE headlines.
2. **RAG retrieval**: chunks normalized articles, runs explainable lexical retrieval with BM25, TF-IDF, or hybrid scoring, then applies metadata reranking.
3. **Agent workflow**: runs local heuristic agents by default, with an optional OpenAI/LangGraph path through the bundled `person2_agent_system_handoff` workflow.
4. **Product surface**: Streamlit dashboard with live analysis, market scan, monitor, evidence audit, and evaluation views.

## Weak Points Found

- Retrieval ranked mostly by text similarity plus credibility, but did not expose why a chunk was selected.
- Chunk metadata was thin: no source URL propagation, rank, score breakdown, recency signal, or ticker/company match.
- The UI showed final citations but not an evidence ledger that separated supporting evidence, counter-evidence, and neutral context.
- The agent trace was closer to a three-step class-project workflow than a real analyst pipeline.
- The Evidence Audit tab could not inspect the retrieval internals of a newly generated signal.
- Several user-facing strings still contain encoding artifacts from earlier copy/paste operations.

## Implemented Improvements

### Metadata-Aware Retrieval

`rag_pipeline.py` now enriches every `RAGChunk` with:

- source URL
- article ID and chunk index
- word count
- ticker/company match flags
- approximate article age in days
- retrieval rank
- blended retrieval score
- per-feature score breakdown

Retrieval now blends lexical/semantic relevance with:

- BM25 sparse event matching
- TF-IDF or keyword overlap
- ticker/company match
- source credibility
- source authority
- recency
- financial-intent overlap

It also supports optional `source_allowlist`, `min_credibility`, and `max_age_days` filters for future product controls.

See `docs/retrieval_architecture.md` for the TF-IDF vs BM25 vs embedding tradeoff discussion.

### Evidence Ledger and Verifier

`agent_runner.py` now builds an `evidence_profile` for each signal packet:

- supporting/challenging/context counts
- source count
- average credibility
- average retrieval score
- top evidence rows
- skeptical verifier flags
- grounding summary

The agent trace is expanded into a more realistic research workflow:

1. News Retriever
2. Evidence Selector
3. Market Relevance Analyst
4. Risk / Sentiment Analyst
5. Skeptical Verifier
6. Signal Synthesizer
7. Decision Agent

### Demo/UI Upgrade

`app.py` now renders:

- a workflow rail for each live analysis
- retrieval strategy and feature notes
- an evidence ledger table with rank, stance, source, date, retrieval score, credibility, semantic score, ticker match, and excerpt
- verifier warnings directly in the Live Analysis and Evidence Audit tabs

### Retrieval Case Studies

`retrieval_case_studies.py` provides a deterministic sanity harness for keyword, TF-IDF, BM25, and hybrid retrieval. The cases are synthetic fixtures for regression testing only; they are not presented as production performance metrics.

## Remaining Opportunities

- Build a labeled real-news retrieval benchmark before adding embedding reranking.
- Add a no-RAG LLM baseline in Evaluation Lab.
- Persist live evidence profiles into bundled demo packets so the Evidence Audit tab is populated before a user runs a live query.
- Clean remaining encoding artifacts throughout UI copy.
- Add Playwright screenshot QA for the Streamlit dashboard.
