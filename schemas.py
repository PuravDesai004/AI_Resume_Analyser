from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Internal Schemas ──

class GroundedSkill(BaseModel):
    """Internal standardized skill representation after taxonomy grounding."""
    original_name: str
    standardized_name: str
    skill_id: str
    source: Literal["esco", "custom"]
    match_type: Literal["exact", "close"]


# ── External & API Schemas ──

class JDRecord(BaseModel):
    """Stored Job Description record."""
    jd_id: str
    title: str
    company: str
    location: str
    snippet: str
    full_text: str
    essential_skills: list[GroundedSkill] = Field(default_factory=list)
    preferred_skills: list[GroundedSkill] = Field(default_factory=list)
    skills_cached: bool = False


class RankingCard(BaseModel):
    """Ranked job description summary card for Tier 1 results."""
    jd_id: str
    rank: int
    similarity_score: int = Field(..., ge=0, le=100, description="Similarity score between 0 and 100")
    title: str
    company: str
    location: str
    snippet: str


class RankingResult(BaseModel):
    """Tier 1 ranking response."""
    resume_id: str
    results: list[RankingCard] = Field(default_factory=list)


class MatchedSkill(BaseModel):
    """Skill matched between resume and JD."""
    skill_name: str
    match_type: Literal["exact", "close"]


class SkillGap(BaseModel):
    """Grounded skill comparison categories."""
    matched: list[MatchedSkill] = Field(default_factory=list)
    missing_essential: list[str] = Field(default_factory=list)
    missing_preferred: list[str] = Field(default_factory=list)
    extra: list[str] = Field(default_factory=list)


class RequirementMatch(BaseModel):
    """Qualitative match percentage for a requirement category."""
    category: Literal["Skills", "Experience", "Education", "Responsibilities"]
    match_percent: int = Field(..., ge=0, le=100)


class Recommendation(BaseModel):
    """Actionable recommendation for the candidate."""
    skill: str
    reason: str


class DetailAnalysis(BaseModel):
    """Tier 2 comprehensive gap analysis response."""
    resume_id: str
    jd_id: str
    context_sufficient: bool
    insufficient_reason: Optional[str] = None
    match_score: int = Field(..., ge=0, le=100)
    summary: str
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    skill_gap: SkillGap = Field(default_factory=SkillGap)
    requirement_match: list[RequirementMatch] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str
    message: str
    stage: Optional[str] = None


if __name__ == "__main__":
    # Self-validation block
    grounded = GroundedSkill(
        original_name="Python3",
        standardized_name="Python",
        skill_id="custom:python",
        source="custom",
        match_type="exact"
    )
    assert GroundedSkill.model_validate(grounded.model_dump()) == grounded

    jd = JDRecord(
        jd_id="jd_001",
        title="Backend Engineer",
        company="Acme Corp",
        location="Remote",
        snippet="Looking for a Python backend engineer...",
        full_text="Looking for a Python backend engineer with FastAPI experience.",
        essential_skills=[grounded],
        preferred_skills=[],
        skills_cached=True
    )
    assert JDRecord.model_validate(jd.model_dump()) == jd

    # Test empty skill lists with skills_cached=True (valid expected state)
    jd_empty = JDRecord(
        jd_id="jd_002",
        title="General Role",
        company="Acme Corp",
        location="Remote",
        snippet="General duties...",
        full_text="General duties as assigned.",
        essential_skills=[],
        preferred_skills=[],
        skills_cached=True
    )
    assert JDRecord.model_validate(jd_empty.model_dump()) == jd_empty
    assert jd_empty.skills_cached is True
    assert len(jd_empty.essential_skills) == 0

    card = RankingCard(
        jd_id="jd_001",
        rank=1,
        similarity_score=85,
        title="Backend Engineer",
        company="Acme Corp",
        location="Remote",
        snippet="Looking for a Python backend engineer..."
    )
    ranking = RankingResult(resume_id="res_001", results=[card])
    assert RankingResult.model_validate(ranking.model_dump()) == ranking

    grounded = GroundedSkill(
        original_name="Python3",
        standardized_name="Python",
        skill_id="custom:python",
        source="custom",
        match_type="exact"
    )
    assert GroundedSkill.model_validate(grounded.model_dump()) == grounded

    gap = SkillGap(
        matched=[MatchedSkill(skill_name="Python", match_type="exact")],
        missing_essential=["Docker"],
        missing_preferred=["Kubernetes"],
        extra=["Flask"]
    )
    detail = DetailAnalysis(
        resume_id="res_001",
        jd_id="jd_001",
        context_sufficient=True,
        insufficient_reason=None,
        match_score=75,
        summary="Strong candidate for backend role.",
        strengths=["Solid Python background"],
        weaknesses=["Missing containerization skills"],
        skill_gap=gap,
        requirement_match=[
            RequirementMatch(category="Skills", match_percent=75),
            RequirementMatch(category="Experience", match_percent=80)
        ],
        recommendations=[Recommendation(skill="Docker", reason="Required for deployment")]
    )
    assert DetailAnalysis.model_validate(detail.model_dump()) == detail

    err = ErrorResponse(error_code="TEST_ERR", message="Testing errors", stage="unit_test")
    assert ErrorResponse.model_validate(err.model_dump()) == err

    # Verify Literal rejection
    try:
        MatchedSkill(skill_name="Python", match_type="invalid")
        raise AssertionError("Expected ValidationError on invalid Literal")
    except Exception:
        pass

    print("All schemas validated successfully.")
