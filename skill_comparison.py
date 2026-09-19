from schemas import GroundedSkill, SkillGap, MatchedSkill


def compare_skills(
    grounded_resume: list[GroundedSkill],
    grounded_jd_essential: list[GroundedSkill],
    grounded_jd_preferred: list[GroundedSkill]
) -> tuple[SkillGap, int]:
    """
    Deterministically compares grounded skills between resume and JD.
    Uses set operations keyed on standardized_name.lower().
    Computes match_score using fixed weighted formula: 80% essential, 20% preferred.
    """
    # Key resume skills by normalized standardized name
    resume_map: dict[str, GroundedSkill] = {}
    for s in grounded_resume:
        key = s.standardized_name.strip().lower()
        if key not in resume_map:
            resume_map[key] = s

    # Deduplicate and key JD essential skills
    essential_map: dict[str, GroundedSkill] = {}
    for s in grounded_jd_essential:
        key = s.standardized_name.strip().lower()
        if key not in essential_map:
            essential_map[key] = s

    # Deduplicate and key JD preferred skills (excluding any already in essential)
    preferred_map: dict[str, GroundedSkill] = {}
    for s in grounded_jd_preferred:
        key = s.standardized_name.strip().lower()
        if key not in essential_map and key not in preferred_map:
            preferred_map[key] = s

    matched: list[MatchedSkill] = []
    missing_essential: list[str] = []
    missing_preferred: list[str] = []
    extra: list[str] = []

    matched_essential_count = 0
    missing_essential_count = 0
    matched_preferred_count = 0
    missing_preferred_count = 0

    # 1. Compare Essential
    for key, jd_skill in essential_map.items():
        if key in resume_map:
            matched_essential_count += 1
            res_skill = resume_map[key]
            matched.append(
                MatchedSkill(
                    skill_name=jd_skill.standardized_name,
                    match_type=res_skill.match_type
                )
            )
        else:
            missing_essential_count += 1
            missing_essential.append(jd_skill.standardized_name)

    # 2. Compare Preferred
    for key, jd_skill in preferred_map.items():
        if key in resume_map:
            matched_preferred_count += 1
            res_skill = resume_map[key]
            matched.append(
                MatchedSkill(
                    skill_name=jd_skill.standardized_name,
                    match_type=res_skill.match_type
                )
            )
        else:
            missing_preferred_count += 1
            missing_preferred.append(jd_skill.standardized_name)

    # 3. Compute Extra (Resume skills not in essential or preferred)
    for key, res_skill in resume_map.items():
        if key not in essential_map and key not in preferred_map:
            extra.append(res_skill.standardized_name)

    # 4. Compute Match Score
    essential_total = matched_essential_count + missing_essential_count
    preferred_total = matched_preferred_count + missing_preferred_count

    essential_ratio = (matched_essential_count / essential_total) if essential_total > 0 else 1.0
    preferred_ratio = (matched_preferred_count / preferred_total) if preferred_total > 0 else 1.0

    match_score = round(100 * (0.8 * essential_ratio + 0.2 * preferred_ratio))
    # Clamp to [0, 100]
    match_score = max(0, min(100, match_score))

    skill_gap = SkillGap(
        matched=matched,
        missing_essential=missing_essential,
        missing_preferred=missing_preferred,
        extra=extra
    )

    return skill_gap, match_score


if __name__ == "__main__":
    # Test fixture with known overlap
    r_skills = [
        GroundedSkill(original_name="Python", standardized_name="Python", skill_id="custom:python", source="custom", match_type="exact"),
        GroundedSkill(original_name="Docker", standardized_name="Docker", skill_id="custom:docker", source="custom", match_type="exact"),
        GroundedSkill(original_name="FastAPI", standardized_name="FastAPI", skill_id="custom:fastapi", source="custom", match_type="close")
    ]
    jd_ess = [
        GroundedSkill(original_name="Python", standardized_name="Python", skill_id="custom:python", source="custom", match_type="exact"),
        GroundedSkill(original_name="Kubernetes", standardized_name="Kubernetes", skill_id="custom:kubernetes", source="custom", match_type="exact")
    ]
    jd_pref = [
        GroundedSkill(original_name="Docker", standardized_name="Docker", skill_id="custom:docker", source="custom", match_type="exact"),
        GroundedSkill(original_name="AWS", standardized_name="AWS", skill_id="custom:aws", source="custom", match_type="exact")
    ]

    gap, score = compare_skills(r_skills, jd_ess, jd_pref)
    print("Computed Gap:", gap.model_dump())
    print("Computed Score:", score)

    # Essential: 1/2 = 0.5. Preferred: 1/2 = 0.5. Score = round(100 * (0.8 * 0.5 + 0.2 * 0.5)) = 50.
    assert score == 50
    assert len(gap.matched) == 2
    assert gap.missing_essential == ["Kubernetes"]
    assert gap.missing_preferred == ["AWS"]
    assert gap.extra == ["FastAPI"]

    # 10 identical runs test
    for _ in range(10):
        g, s = compare_skills(r_skills, jd_ess, jd_pref)
        assert s == 50
        assert g == gap

    print("skill_comparison acceptance criteria passed!")
