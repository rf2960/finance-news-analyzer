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


class MarketSnapshot(BaseModel):
    last_price: float
    day_change: float
    benchmark_change: float
    relative_strength: float
    volume_vs_average: float
    valuation_note: str


class SignalPacket(BaseModel):
    id: str
    ticker: str
    company: str
    sector: str | None = None
    benchmark: str | None = None
    event_type: str | None = None
    direction: Direction
    horizon_days: Horizon
    confidence: float = Field(ge=0.0, le=1.0)
    novelty_score: float | None = Field(default=None, ge=0.0, le=1.0)
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    source_quality: float | None = Field(default=None, ge=0.0, le=1.0)
    published_at: str
    reasoning: str
    catalyst: str
    thesis_bullets: list[str] = []
    risk_factors: list[str] = []
    counter_evidence: list[str] = []
    watch_items: list[str] = []
    market_snapshot: MarketSnapshot | None = None
    citations: list[Citation]
    agent_trace: list[AgentTraceStep]
    baseline_sentiment: Direction
    baseline_random: Direction
