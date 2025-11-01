# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class TechnicalClass(BaseModel):
    """Classification result for whether a job is technical and role type."""

    is_technical: bool = Field(
        description="True if the role is technical (SDE, ML, AI, DevOps, SRE, Data Engineer, etc.)"
    )
    role_category: str = Field(
        description="The specific technical category (e.g., 'ML Engineer', 'Backend SDE', 'DevOps') or 'Non-Technical'"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )


class GenericJobInfo(BaseModel):
    """Generic job information from a job posting."""

    work_mode: Literal["remote", "hybrid", "on-site", "not_specified"] = Field(
        description="Work location type"
    )

    languages_required: List[str] = Field(
        description="Programming languages or human languages required (e.g., ['Python', 'Java', 'English', 'French'])",
        default_factory=list,
    )

    salary_info: Optional[str] = Field(
        description="Salary range if mentioned, otherwise None", default=None
    )

    job_type: Literal[
        "full-time", "part-time", "contract", "internship", "not_specified"
    ] = Field(description="Type of employment")

    requires_local_nationality: bool = Field(
        description="True if the job requires local citizenship/nationality or work authorization"
    )
    model_config = {"extra": "allow"}


class EnrichedJob(BaseModel):
    """Complete job information with classification and extractions."""

    # Original job data
    title: str
    company: str
    location: str
    description: str

    # Classification
    classification: TechnicalClass

    # Generic info (only if technical)
    extraction: Optional[GenericJobInfo] = None
