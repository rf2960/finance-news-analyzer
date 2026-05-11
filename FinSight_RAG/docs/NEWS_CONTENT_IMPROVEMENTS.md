# News Content Quality Notes

This note records a cleanup made during implementation: early evidence cards sometimes repeated article titles instead of showing useful article content.

## Issue

Many RSS feeds expose only a headline and a short description. Some descriptions are identical to the title, which made the Evidence Audit view look repetitive and less credible.

## Fix

`news_ingester.py` now checks multiple content fields in order of usefulness:

1. rich `content` fields when available
2. `summary_detail`
3. standard `summary` or `description`
4. fallback text when only a title is available

It also detects obvious title repetition and avoids presenting duplicated headline text as evidence.

## Impact

Evidence cards now show more useful excerpts, which improves both readability and citation auditability in the dashboard.
