from schemas import RawSkill, GroundedSkill, ExtractionResult, GroundingResult
from esco_loader import ESCOIndex

DEFAULT_FUZZY_THRESHOLD = 80


def ground_skill(raw: RawSkill, esco_index: ESCOIndex, fuzzy_threshold: int = DEFAULT_FUZZY_THRESHOLD) -> GroundedSkill:
    """
    Standardizes a raw skill against the ESCO taxonomy using exact and fuzzy matching.
    Falls back to 'unverified' if no match meets the threshold.
    """
    phrase = raw.skill_name.strip()

    # Step A: Exact match lookup
    exact = esco_index.exact_match(phrase)
    if exact:
        return GroundedSkill(
            original_name=raw.skill_name,
            standardized_name=exact["preferred_label"],
            esco_id=exact.get("uri"),
            match_type="exact",
            match_score=100.0,
            context=raw.context,
            priority=raw.priority
        )

    # Step B: Fuzzy match lookup
    fuzzy = esco_index.fuzzy_match(phrase, threshold=fuzzy_threshold)
    if fuzzy:
        return GroundedSkill(
            original_name=raw.skill_name,
            standardized_name=fuzzy["preferred_label"],
            esco_id=fuzzy.get("uri"),
            match_type="close",
            match_score=fuzzy.get("score"),
            context=raw.context,
            priority=raw.priority
        )

    # Step C: Unverified fallback
    return GroundedSkill(
        original_name=raw.skill_name,
        standardized_name=raw.skill_name,
        esco_id=None,
        match_type="unverified",
        match_score=None,
        context=raw.context,
        priority=raw.priority
    )


def ground_skills(extraction: ExtractionResult, esco_index: ESCOIndex, fuzzy_threshold: int = DEFAULT_FUZZY_THRESHOLD) -> GroundingResult:
    """Ground all extracted resume and JD skills."""
    grounded_resume = [
        ground_skill(s, esco_index, fuzzy_threshold=fuzzy_threshold)
        for s in extraction.resume_skills
    ]
    grounded_jd = [
        ground_skill(s, esco_index, fuzzy_threshold=fuzzy_threshold)
        for s in extraction.jd_skills
    ]
    return GroundingResult(resume_skills=grounded_resume, jd_skills=grounded_jd)


if __name__ == "__main__":
    print("Testing skill grounding logic...")
    # Self-contained test with dummy index if real one is not available
    class MockESCOIndex:
        def exact_match(self, phrase):
            if phrase.lower() == "python":
                return {"uri": "esco:1", "preferred_label": "Python"}
            return None
        def fuzzy_match(self, phrase, threshold=80):
            if "react" in phrase.lower():
                return {"uri": "esco:2", "preferred_label": "React", "score": 90.0}
            return None

    mock_index = MockESCOIndex()
    raw = RawSkill(skill_name="React.js", context="Projects", priority="unspecified")
    result = ground_skill(raw, mock_index)
    print("Grounded result:", result.model_dump())
