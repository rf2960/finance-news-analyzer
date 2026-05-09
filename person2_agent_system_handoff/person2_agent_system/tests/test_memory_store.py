import unittest
from pathlib import Path

from src.orchestration.memory_store import PatternMemoryStore


class PatternMemoryStoreTests(unittest.TestCase):
    def test_record_and_lookup_notes(self) -> None:
        memory_path = Path(__file__).resolve().parents[1] / "outputs" / "memory" / "test_memory.json"
        if memory_path.exists():
            memory_path.unlink()

        store = PatternMemoryStore(memory_path)
        store.record_preliminary_signal(
            ticker="AAPL",
            query_date="2026-05-04",
            direction="bullish",
            confidence=0.66,
            citations=["DOC1"],
        )
        notes = store.lookup_notes("AAPL")
        self.assertTrue(notes)


if __name__ == "__main__":
    unittest.main()
