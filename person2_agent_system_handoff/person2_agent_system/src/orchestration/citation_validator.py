from __future__ import annotations

from typing import Iterable, Set


def collect_known_citations(chunks: Iterable) -> Set[str]:
    return {chunk.citation_id for chunk in chunks}


def citations_are_valid(citation_ids: list[str], known_citations: set[str]) -> bool:
    return all(citation in known_citations for citation in citation_ids)
