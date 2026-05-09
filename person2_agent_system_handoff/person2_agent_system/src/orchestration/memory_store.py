from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.utils.json_parser import read_json, write_json


class PatternMemoryStore:
    """Small JSON-backed memory store for repeated ticker patterns."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict:
        if not self.path.exists():
            return {"signals": []}
        return read_json(self.path)

    def _save(self, data: dict) -> None:
        write_json(self.path, data)

    def lookup_notes(self, ticker: str) -> list[str]:
        data = self._load()
        ticker_signals = [row for row in data.get("signals", []) if row.get("ticker") == ticker]
        if not ticker_signals:
            return []

        direction_counts = Counter(row.get("direction") for row in ticker_signals)
        notes = [f"Memory contains {len(ticker_signals)} prior signals for {ticker}."]
        dominant_direction, count = direction_counts.most_common(1)[0]
        notes.append(f"Most common prior direction was {dominant_direction} ({count} instances).")

        evaluated = [row for row in ticker_signals if row.get("outcome_label")]
        if evaluated:
            hit_rate = sum(1 for row in evaluated if row["outcome_label"] == "correct") / len(evaluated)
            notes.append(f"Observed historical hit rate in memory: {hit_rate:.2f}.")
        return notes

    def record_preliminary_signal(
        self,
        ticker: str,
        query_date: str,
        direction: str,
        confidence: float,
        citations: list[str],
    ) -> None:
        data = self._load()
        data.setdefault("signals", []).append(
            {
                "ticker": ticker,
                "query_date": query_date,
                "direction": direction,
                "confidence": confidence,
                "citations": citations,
                "outcome_label": None,
            }
        )
        self._save(data)
