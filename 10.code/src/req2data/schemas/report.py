from pydantic import BaseModel, Field

from req2data.schemas.base import LLMOutput


class TraceabilityRow(BaseModel):
    requirement_id: str
    requirement: str
    columns: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    sql_ref: str = ""
    check_status: str = ""


class ReportSummary(LLMOutput):
    headline: str
    highlights: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
