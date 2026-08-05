"""Structured-output schema for triage's handoff decision."""

from typing import Literal

from pydantic import BaseModel, Field

type ResearcherAgent = Literal['sales', 'billing', 'support', 'end']


class HandoffDecision(BaseModel):
    """Triage's decision on which specialist should handle the query, and why."""

    handoff_to: ResearcherAgent = Field(description='Which agent to hand off to')
    reason: str = Field(description='Reason for handoff')
    context: str = Field(description='Key context to pass to the next agent')
