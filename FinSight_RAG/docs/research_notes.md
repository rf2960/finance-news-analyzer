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

## Quartr

Quartr focuses on earnings calls, live transcripts, filings, reports, presentations, and LLM-compatible event data.

Relevant UI takeaways:

- Event documents need structured metadata so agents can reason across them.
- Transcript and filing evidence should be treated as first-class source material.
- A finance demo should distinguish news-derived signals from earnings-event signals.

Source:

- https://www.quartr.ai/

## Fiscal.ai

Fiscal.ai positions itself as a modern financial data platform with global financial data, AI summaries, dashboards, KPI data, estimates, IR content, and click-through auditability.

Relevant UI takeaways:

- Pair qualitative thesis text with quantitative business context.
- Include watchlists and dashboards rather than only one-off answers.
- Show auditability for both text evidence and numeric data.

Source:

- https://fiscal.ai/

## Koyfin and YCharts

Koyfin and YCharts emphasize customizable dashboards, broad market data, charting, portfolio context, watchlists, AI chat, and report/proposal workflows.

Relevant UI takeaways:

- Analysts need compact monitoring views and customizable slices of the market.
- Charts and tables should support a presentation-ready story.
- AI output is more useful when connected to existing research workflows.

Sources:

- https://www.koyfin.com/
- https://ycharts.com/

## BloombergGPT

BloombergGPT is important as literature rather than UI inspiration. The paper argues for domain-specific financial language modeling and evaluates performance across financial NLP tasks.

Relevant methodology takeaways:

- Finance-specific language and data matter.
- Evaluation should be domain-specific, not only general text quality.
- A credible GenAI finance project needs explicit benchmark comparisons.

Source:

- https://arxiv.org/abs/2303.17564

## Changes Reflected in FinSight RAG

- Market Monitor: compact queue with confidence, novelty, relative strength, and disagreement flags.
- Thesis Workspace: signal card, quant snapshot, price chart, thesis drivers, risks, and watch list.
- Evidence Audit: citation cards, source credibility scores, agent path, and topic lens.
- Evaluation Lab: model-vs-baseline hit rate and return diagnostics.
- Research Brief: benchmark table and methodology ladder.
