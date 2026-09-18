from schemas import GroundedSkill, GroundingResult, ComparisonResult


def compare_skills(grounding: GroundingResult) -> ComparisonResult:
    """
    Deterministically compares verified grounded skills between candidate resume and job description.
    Unverified skills are separated for transparency and not matched directly.
    """
    verified_resume = [s for s in grounding.resume_skills if s.match_type != "unverified"]
    unverified_resume = [s for s in grounding.resume_skills if s.match_type == "unverified"]

    verified_jd = [s for s in grounding.jd_skills if s.match_type != "unverified"]
    unverified_jd = [s for s in grounding.jd_skills if s.match_type == "unverified"]

    # Map normalized standardized names
    resume_map: dict[str, GroundedSkill] = {
        s.standardized_name.strip().lower(): s for s in verified_resume
    }
    jd_map: dict[str, GroundedSkill] = {
        s.standardized_name.strip().lower(): s for s in verified_jd
    }

    matched_skills: list[GroundedSkill] = []
    missing_essential: list[GroundedSkill] = []
    missing_preferred: list[GroundedSkill] = []
    extra_skills: list[GroundedSkill] = []

    # 1. Evaluate JD skills against candidate
    for norm_name, jd_skill in jd_map.items():
        if norm_name in resume_map:
            resume_skill = resume_map[norm_name]
            # Create matched skill preserving resume context and JD priority
            matched_skills.append(
                GroundedSkill(
                    original_name=resume_skill.original_name,
                    standardized_name=jd_skill.standardized_name,
                    esco_id=jd_skill.esco_id or resume_skill.esco_id,
                    match_type=resume_skill.match_type,
                    match_score=resume_skill.match_score,
                    context=resume_skill.context,
                    priority=jd_skill.priority
                )
            )
        else:
            if jd_skill.priority == "essential":
                missing_essential.append(jd_skill)
            else:
                missing_preferred.append(jd_skill)

    # 2. Evaluate Resume skills not present in JD
    for norm_name, resume_skill in resume_map.items():
        if norm_name not in jd_map:
            extra_skills.append(resume_skill)

    return ComparisonResult(
        matched_skills=matched_skills,
        missing_essential=missing_essential,
        missing_preferred=missing_preferred,
        extra_skills=extra_skills,
        unverified_resume=unverified_resume,
        unverified_jd=unverified_jd
    )


if __name__ == "__main__":
    print("Testing skill comparison...")
    g_res = GroundingResult(
        resume_skills=[
            GroundedSkill(original_name="Python", standardized_name="Python", match_type="exact", context="Work Experience"),
            GroundedSkill(original_name="Docker", standardized_name="Docker", match_type="exact", context="Projects"),
            GroundedSkill(original_name="CustomInternalTool", standardized_name="CustomInternalTool", match_type="unverified")
        ],
        jd_skills=[
            GroundedSkill(original_name="Python", standardized_name="Python", match_type="exact", priority="essential"),
            GroundedSkill(original_name="Kubernetes", standardized_name="Kubernetes", match_type="exact", priority="essential"),
            GroundedSkill(original_name="AWS", standardized_name="AWS", match_type="exact", priority="preferred")
        ]
    )
    comp = compare_skills(g_res)
    print("Matched:", [s.standardized_name for s in comp.matched_skills])
    print("Missing Essential:", [s.standardized_name for s in comp.missing_essential])
    print("Missing Preferred:", [s.standardized_name for s in comp.missing_preferred])
    print("Extra Skills:", [s.standardized_name for s in comp.extra_skills])
    print("Unverified Resume:", [s.original_name for s in comp.unverified_resume])
