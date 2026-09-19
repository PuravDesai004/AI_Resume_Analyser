from typing import Optional
from schemas import DetailAnalysis, SkillGap


def check_sufficiency(
    resume_text: str,
    jd_text: str,
    min_words: int = 30
) -> tuple[bool, Optional[str]]:
    """
    Pure text-length gate. Evaluates whether both documents meet the minimum word count.
    Runs before any Gemini API call is made.
    """
    resume_words = len(resume_text.strip().split()) if resume_text else 0
    jd_words = len(jd_text.strip().split()) if jd_text else 0

    if resume_words < min_words and jd_words < min_words:
        return (
            False,
            f"Both documents are too short for meaningful analysis: "
            f"resume has {resume_words} words, JD has {jd_words} words (minimum {min_words} each required)."
        )
    if resume_words < min_words:
        return (
            False,
            f"Resume text is too short for analysis: found {resume_words} words (minimum {min_words} required)."
        )
    if jd_words < min_words:
        return (
            False,
            f"Job description text is too short for analysis: found {jd_words} words (minimum {min_words} required)."
        )

    return True, None


def build_insufficient_response(
    resume_id: str,
    jd_id: str,
    reason: str
) -> DetailAnalysis:
    """
    Builds a fully-shaped DetailAnalysis object when context is insufficient.
    Guarantees that the response structure is identical to a successful run.
    """
    return DetailAnalysis(
        resume_id=resume_id,
        jd_id=jd_id,
        context_sufficient=False,
        insufficient_reason=reason,
        match_score=0,
        summary=f"Analysis bypassed: {reason}",
        strengths=[],
        weaknesses=[],
        skill_gap=SkillGap(
            matched=[],
            missing_essential=[],
            missing_preferred=[],
            extra=[]
        ),
        requirement_match=[],
        recommendations=[]
    )


if __name__ == "__main__":
    short_resume = "Hello I am a python dev."
    good_resume = " ".join(["word"] * 40)
    short_jd = "Need a coder."
    good_jd = " ".join(["requirement"] * 50)

    # 1. Short resume test
    ok, reason = check_sufficiency(short_resume, good_jd)
    assert not ok
    assert "Resume text is too short" in reason

    # 2. Both short test
    ok, reason = check_sufficiency(short_resume, short_jd)
    assert not ok
    assert "Both documents are too short" in reason

    # 3. Sufficient test
    ok, reason = check_sufficiency(good_resume, good_jd)
    assert ok
    assert reason is None

    # 4. Response structure validation
    resp = build_insufficient_response("res_01", "jd_01", "Too short")
    assert DetailAnalysis.model_validate(resp.model_dump()) == resp
    assert resp.context_sufficient is False
    assert resp.match_score == 0
    print("sufficiency_check acceptance criteria passed!")
