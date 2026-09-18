from typing import Literal, Optional
from pydantic import BaseModel, Field

# 1. RawSkill — Output of extraction (Step 1)
class RawSkill(BaseModel):
    """Raw skill phrase extracted directly from text."""
    skill_name: str = Field(..., description="The raw phrase as extracted from the document")
    context: Optional[str] = Field(None, description="Section heading where skill was found (resume only)")
    priority: Literal["essential", "preferred", "unspecified"] = Field(
        "unspecified", description="Priority level specified in the job description"
    )

# 2. GroundedSkill — Output of grounding (Step 2)
class GroundedSkill(BaseModel):
    """Skill standardized and verified against the ESCO taxonomy."""
    original_name: str = Field(..., description="The raw phrase before standardization")
    standardized_name: str = Field(..., description="The ESCO-matched preferred name or original if unverified")
    esco_id: Optional[str] = Field(None, description="ESCO concept URI, if matched")
    match_type: Literal["exact", "close", "unverified"] = Field(..., description="Match category")
    match_score: Optional[float] = Field(None, description="Fuzzy match score (0-100), if applicable")
    context: Optional[str] = Field(None, description="Carried forward from RawSkill")
    priority: Literal["essential", "preferred", "unspecified"] = Field(
        "unspecified", description="Carried forward from RawSkill"
    )

# 3. ExtractionResult — Container for Step 1 output
class ExtractionResult(BaseModel):
    """Raw skills extracted from both resume and job description."""
    resume_skills: list[RawSkill] = Field(default_factory=list)
    jd_skills: list[RawSkill] = Field(default_factory=list)

# 4. GroundingResult — Container for Step 2 output
class GroundingResult(BaseModel):
    """Grounded skills for both resume and job description."""
    resume_skills: list[GroundedSkill] = Field(default_factory=list)
    jd_skills: list[GroundedSkill] = Field(default_factory=list)

# 5. ComparisonResult — Output of comparison (Step 3)
class ComparisonResult(BaseModel):
    """Categorized skill alignment and discrepancy sets."""
    matched_skills: list[GroundedSkill] = Field(default_factory=list)
    missing_essential: list[GroundedSkill] = Field(default_factory=list)
    missing_preferred: list[GroundedSkill] = Field(default_factory=list)
    extra_skills: list[GroundedSkill] = Field(default_factory=list)
    unverified_resume: list[GroundedSkill] = Field(default_factory=list)
    unverified_jd: list[GroundedSkill] = Field(default_factory=list)

# 6. SufficiencyResult — Output of gate (Step 4)
class SufficiencyResult(BaseModel):
    """Validation decision on whether enough verified skills exist for synthesis."""
    passed: bool
    reason: Optional[str] = None
    resume_grounded_count: int
    jd_grounded_count: int

# 7. AnalysisOutput — Final output artifact (Step 5 / Pipeline Output)
class AnalysisOutput(BaseModel):
    """Final comprehensive resume vs JD gap analysis."""
    summary: str
    match_score: Optional[float] = None
    matched_skills: list[dict] = Field(default_factory=list)
    missing_essential: list[dict] = Field(default_factory=list)
    missing_preferred: list[dict] = Field(default_factory=list)
    extra_skills: list[dict] = Field(default_factory=list)
    unverified_skills: dict = Field(default_factory=lambda: {"resume": [], "jd": []})
    recommendations: list[str] = Field(default_factory=list)
    generation_ran: bool
    sufficiency_detail: Optional[str] = None


if __name__ == "__main__":
    # Self-validation block for schemas
    sample_raw = RawSkill(skill_name="Python", context="Work Experience", priority="unspecified")
    sample_grounded = GroundedSkill(
        original_name="Python",
        standardized_name="python",
        esco_id="http://data.europa.eu/esco/skill/123",
        match_type="exact",
        match_score=100.0,
        context="Work Experience",
        priority="unspecified"
    )
    print("RawSkill sample:", sample_raw.model_dump())
    print("GroundedSkill sample:", sample_grounded.model_dump())
    print("Schemas module validated successfully.")
