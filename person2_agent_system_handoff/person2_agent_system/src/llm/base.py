from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.utils.json_parser import read_json


class StructuredLLMClient(Protocol):
    def invoke_structured(self, system_prompt: str, user_prompt: str, schema_name: str) -> dict:
        """Return a JSON-compatible dict that matches the named schema."""


class SchemaRegistry:
    def __init__(self, schema_root: str | Path) -> None:
        self.schema_root = Path(schema_root)

    def get_schema(self, schema_name: str) -> dict:
        return read_json(self.schema_root / f"{schema_name}.json")
