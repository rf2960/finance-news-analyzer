# Research Notes for Demo UI

These notes summarize product patterns from market-intelligence platforms that informed the FinSight RAG demo layout.

## AlphaSense

AlphaSense emphasizes trusted AI insights, a large curated document base, integrated workflows, and sentence-level citations. Its public platform page also frames the workflow as moving from discovery through reasoning to deliverable output in one place.

Relevant UI takeaways:

- Keep source citations visible near the generated claim.
- Combine qualitative evidence with quantitative company context.
- Show workflow state instead of only a final answer.
- Make the interface compact enough for repeated analyst use.

Sources:

- https://prod.alpha-sense.com/
- https://help.alpha-sense.com/hc/en-us/articles/42465162626451-AlphaSense-Financials-Explained

## GNOMI

GNOMI frames its product as a real-time intelligence layer that analyzes news, financial data, filings, and global sources. Its Finance Mode announcement highlights live market coverage, transcripts, conversational insight, and global/multilingual coverage.

Relevant UI takeaways:

- Include a market-monitoring layer, not only a static report.
- Separate raw news sentiment from contextual decision intelligence.
- Make verification and attribution part of the user flow.
- Track evolving signals and watch items.

Sources:

- https://www.gnomi.com/en/blog/gnomi-introduces-a-new-standard-for-real-time-intelligence
- https://www.businesswire.com/news/home/20251124017508/en/GNOMI-Launches-the-Only-Finance-Mode-with-Real-Time-Global-Earnings-Calls-and-Generative-Market-Intelligence

## Changes Reflected in FinSight RAG

- Market Monitor: compact queue with confidence, novelty, relative strength, and disagreement flags.
- Thesis Workspace: signal card, quant snapshot, price chart, thesis drivers, risks, and watch list.
- Evidence Audit: citation cards, source credibility scores, agent path, and topic lens.
- Evaluation Lab: model-vs-baseline hit rate and return diagnostics.
