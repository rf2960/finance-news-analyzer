from __future__ import annotations

import os
from pathlib import Path

from src.llm.base import StructuredLLMClient
from src.llm.langchain_openai_client import LangChainOpenAIClient


def load_dotenv_if_present(start_path: str | Path) -> None:
    current = Path(start_path).resolve()
    candidates = [current] + list(current.parents)
    for path in candidates:
        dotenv_path = path / ".env"
        if not dotenv_path.exists():
            continue
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)
        break


def build_llm_client_from_env(schema_root: str | Path) -> StructuredLLMClient | None:
    load_dotenv_if_present(schema_root)
    backend = os.getenv("AGENT_BACKEND", "heuristic").strip().lower()
    if backend in {"", "heuristic", "none"}:
        return None
    if backend == "langchain_openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "AGENT_BACKEND=langchain_openai requires OPENAI_API_KEY. "
                "Add it to your shell environment or to person2_agent_system/.env."
            )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return LangChainOpenAIClient(schema_root=str(schema_root), model=model)
    raise ValueError(f"Unsupported AGENT_BACKEND: {backend}")
