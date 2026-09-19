from typing import Optional
from schemas import GroundedSkill
from esco_loader import SkillIndex


def ground_candidates(skill_names: list[str], index: SkillIndex) -> list[GroundedSkill]:
    """
    Grounds candidate skill names from Gemini against the unified SkillIndex.
    Matches first via exact match, then via fuzzy match (threshold >= 85).
    Candidates that do not match are discarded.
    """
    grounded: list[GroundedSkill] = []

    for name in skill_names:
        clean_name = name.strip()
        if not clean_name:
            continue

        # Step 1: Exact normalized match
        exact = index.exact_match(clean_name)
        if exact:
            grounded.append(
                GroundedSkill(
                    original_name=clean_name,
                    standardized_name=exact["preferred_label"],
                    skill_id=exact["skill_id"],
                    source=exact["source"],
                    match_type="exact"
                )
            )
            continue

        # Step 2: Fuzzy match (threshold >= 85)
        fuzzy = index.fuzzy_match(clean_name, threshold=85)
        if fuzzy:
            grounded.append(
                GroundedSkill(
                    original_name=clean_name,
                    standardized_name=fuzzy["preferred_label"],
                    skill_id=fuzzy["skill_id"],
                    source=fuzzy["source"],
                    match_type="close"
                )
            )
            continue

        # Step 3: Discard non-matching noise
        # Candidate is silently dropped as specified

    return grounded


if __name__ == "__main__":
    idx = SkillIndex()
    test_candidates = ["Python", "React.js", "built distributed systems"]
    result = ground_candidates(test_candidates, idx)
    print("Grounded result:", [g.model_dump() for g in result])
    assert len(result) == 2, f"Expected 2 grounded skills, got {len(result)}"
    assert result[0].original_name == "Python"
    assert result[1].original_name == "React.js"
    print("skill_grounding acceptance criteria passed!")
