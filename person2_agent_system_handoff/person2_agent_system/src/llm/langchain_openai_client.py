from __future__ import annotations

from src.llm.base import SchemaRegistry


class LangChainOpenAIClient:
    """Optional structured-output client using langchain_openai."""

    def __init__(self, schema_root: str, model: str, temperature: float = 0.0) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "langchain_openai is not installed. Install openai, langchain, and langchain-openai "
                "before enabling AGENT_BACKEND=langchain_openai."
            ) from exc

        self.schema_registry = SchemaRegistry(schema_root)
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def invoke_structured(self, system_prompt: str, user_prompt: str, schema_name: str) -> dict:
        schema = self.schema_registry.get_schema(schema_name)
        structured_llm = self.llm.with_structured_output(schema, method="json_schema")
        response = structured_llm.invoke(
            [
                ("system", system_prompt),
                ("human", user_prompt),
            ]
        )
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return dict(response)
