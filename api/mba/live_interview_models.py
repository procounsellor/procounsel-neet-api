"""Pydantic models for the live AI mock interview — contract: docs/mba-api.md "Live AI interview"."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .gdpi_models import Background, Delivery, FeedbackScores

LiveStatus = Literal["in_progress", "completed"]
Band = Literal["ready", "almost", "needs_work"]


class LiveInterviewInput(BaseModel):
    """What the candidate tells the panel up front. Only used to tailor the questions."""

    background: Background | None = None
    workMonths: int = Field(default=0, ge=0, le=240)
    target: str | None = Field(default=None, max_length=120)  # e.g. "IIM, finance"
    about: str | None = Field(default=None, max_length=1500)  # a few lines about themselves
    questions: int = Field(default=7, ge=4, le=10)

    model_config = {"extra": "forbid"}


class LiveAnswerInput(BaseModel):
    answerText: str = Field(min_length=1, max_length=4000)

    model_config = {"extra": "forbid"}


class LiveTurn(BaseModel):
    number: int
    question: str
    questionType: str
    askedAt: str
    answerText: str | None
    mode: Literal["text", "voice"] | None
    delivery: Delivery | None
    scores: FeedbackScores | None  # filled when the answer is in
    note: str | None  # one-line coaching note, shown in the report
    answeredAt: str | None


class LiveReport(BaseModel):
    overall: float | None
    scores: FeedbackScores | None
    band: Band | None
    summary: str
    strengths: list[str]
    improvements: list[str]
    fallback: bool


class LiveProfile(BaseModel):
    background: Background | None
    workMonths: int
    target: str | None


class LiveInterview(BaseModel):
    id: str
    status: LiveStatus
    profile: LiveProfile
    totalQuestions: int
    turns: list[LiveTurn]
    report: LiveReport | None
    startedAt: str
    completedAt: str | None


class LiveInterviewSummary(BaseModel):
    id: str
    status: LiveStatus
    answered: int
    totalQuestions: int
    overall: float | None
    startedAt: str
    completedAt: str | None


class LiveInterviewsResponse(BaseModel):
    interviews: list[LiveInterviewSummary]


class LiveTurnResponse(BaseModel):
    interview: LiveInterview
    remainingToday: int
