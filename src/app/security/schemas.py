"""Structured-output schema for the security guard's classification."""

from pydantic import BaseModel, Field


class SecurityCheckSchema(BaseModel):
    """The security guard's verdict on a piece of user input."""

    safe: bool = Field(description='Whether the input is safe to process')
    reason: str = Field(description='Explanation if unsafe, empty string if safe')
