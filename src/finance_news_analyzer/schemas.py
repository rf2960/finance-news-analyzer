from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


Direction = Literal["Bullish", "Bearish", "Neutral"]
Horizon = Literal[5, 20]


class Citation(BaseModel):
    source: str
    title: str
    url: HttpUrl | str
    excerpt: str
    credibility_weight: float = Field(ge=0.0, le=1.0)


class AgentTraceStep(BaseModel):
    agent: str
    summary: str


class SignalPacket(BaseModel):
    id: str
    ticker: str
    company: str
    direction: Direction
    horizon_days: Horizon
    confidence: float = Field(ge=0.0, le=1.0)
    published_at: str
    reasoning: str
    catalyst: str
    citations: list[Citation]
    agent_trace: list[AgentTraceStep]
    baseline_sentiment: Direction
    baseline_random: Direction

