from schemas import ComparisonResult, GroundingResult, SufficiencyResult, AnalysisOutput

MIN_GROUNDED_SKILLS = 3


def check_sufficiency(
    comparison: ComparisonResult,
    grounding: GroundingResult,
    min_grounded: int = MIN_GROUNDED_SKILLS
) -> SufficiencyResult:
    """
    Evaluates whether both the resume and the JD have enough verified skills
    to produce a statistically sound, hallucination-free gap analysis.
    """
    resume_verified = [s for s in grounding.resume_skills if s.match_type in ("exact", "close")]
    jd_verified = [s for s in grounding.jd_skills if s.match_type in ("exact", "close")]

    resume_count = len(resume_verified)
    jd_count = len(jd_verified)

    passed = (resume_count >= min_grounded) and (jd_count >= min_grounded)
    reason = None

    if not passed:
        reason = (
            f"Insufficient verified skills: found {resume_count} verified in resume, "
            f"{jd_count} in job description (minimum {min_grounded} each required)."
        )

    return SufficiencyResult(
        passed=passed,
        reason=reason,
        resume_grounded_count=resume_count,
        jd_grounded_count=jd_count
    )


def build_insufficient_response(
    sufficiency: SufficiencyResult,
    comparison: ComparisonResult
) -> AnalysisOutput:
    """
    Constructs an AnalysisOutput response when sufficiency check fails,
    returning partial deterministic data while safely skipping LLM generation.
    """
    summary = (
        f"Automated generation was bypassed: {sufficiency.reason} "
        f"Please provide a more detailed resume or job description to enable comprehensive evaluation."
    )

    return AnalysisOutput(
        summary=summary,
        match_score=None,
        matched_skills=[
            {"skill": s.standardized_name, "context": s.context}
            for s in comparison.matched_skills
        ],
        missing_essential=[
            {"skill": s.standardized_name}
            for s in comparison.missing_essential
        ],
        missing_preferred=[
            {"skill": s.standardized_name}
            for s in comparison.missing_preferred
        ],
        extra_skills=[
            {"skill": s.standardized_name, "context": s.context}
            for s in comparison.extra_skills
        ],
        unverified_skills={
            "resume": [s.original_name for s in comparison.unverified_resume],
            "jd": [s.original_name for s in comparison.unverified_jd],
        },
        recommendations=[],
        generation_ran=False,
        sufficiency_detail=sufficiency.reason
    )


if __name__ == "__main__":
    print("Testing sufficiency check...")
    g_res = GroundingResult(resume_skills=[], jd_skills=[])
    comp = ComparisonResult()
    suff = check_sufficiency(comp, g_res, min_grounded=3)
    print("Passed:", suff.passed)
    print("Reason:", suff.reason)
    insufficient_resp = build_insufficient_response(suff, comp)
    print("Generation ran:", insufficient_resp.generation_ran)
