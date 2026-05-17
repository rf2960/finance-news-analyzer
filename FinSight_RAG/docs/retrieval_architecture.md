# Retrieval Architecture Decision

FinSight does not use a vector database by default just because it sounds more advanced. Financial-news retrieval has a different failure profile than open-domain semantic search: exact tickers, company names, products, guidance terms, regulatory events, dates, and source authority often matter more than broad semantic similarity.

## Retrieval Options Compared

| Approach | Strengths | Weaknesses | Fit for FinSight |
|---|---|---|---|
| Keyword match | Very transparent, fast, dependency-free | Brittle phrasing, weak ranking | Useful fallback only |
| TF-IDF | Explainable term weighting, good latency, easy demo inspection | Can miss paraphrases, vocabulary-sensitive | Strong baseline for ticker/event news |
| BM25 | Better sparse retrieval for short queries and exact finance terms | Still lexical; no paraphrase understanding | Best first-stage candidate retriever |
| Embeddings | Captures paraphrases and thematic similarity | Can retrieve semantically related but ticker-wrong passages; harder to explain | Useful optional reranker, not default |
| Cross-encoder reranker | High precision when trained/generalizes well | Higher latency/cost, extra dependency, less transparent | Future production upgrade after lexical candidate recall is stable |
| Hybrid lexical + metadata | Combines exact event matching with source/date/ticker controls | More knobs to calibrate | Current best portfolio/demo architecture |

## Current Design

The implemented retriever uses a lexical-first hybrid:

```text
User ticker/query
        |
        v
Ticker/company/date/source filters
        |
        v
BM25 + TF-IDF/keyword candidate scoring
        |
        v
Metadata reranking
  - ticker/company match
  - source credibility
  - source authority
  - recency
  - financial-intent overlap
        |
        v
Evidence ledger + agent analysis
```

This design favors grounded, inspectable retrieval over opaque semantic recall. Each retrieved chunk carries a score breakdown so the UI can show why the evidence was selected.

## Why Not Embeddings First?

Embedding-first retrieval can be valuable for earnings-call transcripts, long filings, or broad thematic research. For short financial news, it can also create avoidable risk:

- It may retrieve articles that are semantically related but mention the wrong company.
- It can blur source/date constraints unless metadata filtering is strict.
- The reason a source was selected is harder to explain to a reviewer.
- It adds latency and setup complexity that does not improve the local demo unless evaluated.

The recommended production path is not "replace lexical retrieval"; it is:

```text
BM25/TF-IDF candidate generation -> strict metadata filters -> optional embedding/cross-encoder rerank -> cited synthesis
```

## Evaluation Plan

The repo includes `retrieval_case_studies.py`, a small synthetic sanity harness. It is not a production benchmark and should not be reported as live performance. It is useful for regression checks when retrieval logic changes.

Recommended next evaluation work:

- Build a labeled set of real ticker queries with known relevant article IDs.
- Compare keyword, TF-IDF, BM25, hybrid, and optional embedding reranking on Recall@K, Precision@K, MRR, and citation grounding.
- Track latency and source diversity alongside relevance.
- Add hallucination-risk checks: final claims must map to selected evidence snippets.
