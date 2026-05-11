# Signal Direction Logic Notes

This note summarizes changes made to reduce overly cautious neutral outputs.

## Issue

The heuristic pipeline sometimes returned `Neutral` even when the retrieved news and technical context had a clear directional lean.

## Fixes

- Lowered the sentiment threshold used to classify bullish/bearish evidence.
- Passed technical-factor bias into the strategist step.
- Allowed strong technical context to move a mixed news signal toward bullish or bearish.
- Added technical notes into the generated thesis when they materially affect the direction.

## Remaining Limitation

The heuristic sentiment score is still rule-based. A domain-adapted financial sentiment model, such as FinBERT, would likely improve this component.
