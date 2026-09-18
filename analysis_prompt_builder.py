from google import genai
from schemas import ComparisonResult

GENERATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "A concise 3-5 sentence narrative assessment of the candidate's overall fit for the role, based strictly on the verified skill data provided."
        },
        "match_score": {
            "type": "number",
            "description": "A numerical score from 0 to 100 representing the candidate's skill alignment with the job requirements. Weight essential missing skills more heavily than preferred ones."
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "One specific, actionable recommendation for the candidate."
            },
            "description": "3-7 actionable recommendations for the candidate to improve their fit. Each should reference a specific skill or gap identified in the data."
        }
    },
    "required": ["summary", "match_score", "recommendations"]
}

ANALYSIS_SYSTEM_PROMPT = """You are a professional resume-to-job-description gap analysis engine. You will be given structured, pre-verified skill comparison data between a candidate's resume and a job description. Your task is to produce a professional assessment based STRICTLY on this data.

Rules:
1. BASE YOUR ANALYSIS ONLY on the verified skill data provided below. Do not assume skills, infer experience, or add information that is not explicitly present in the data.
2. The "match_score" should reflect the weighted alignment:
   - Missing ESSENTIAL skills should have a significant negative impact on the score.
   - Missing PREFERRED skills should have a moderate negative impact.
   - Extra skills (candidate has but JD doesn't require) should have no negative impact but may be noted positively if relevant to the domain.
   - Unverified skills should NOT influence the score — they are included for transparency only.
3. The "summary" should:
   - Open with the candidate's overall alignment level (strong match / moderate match / weak match).
   - Mention the most important matched skills.
   - Highlight the most critical gaps (especially essential missing skills).
   - Be professional and neutral in tone — do not exaggerate strengths or weaknesses.
4. The "recommendations" should:
   - Be specific and actionable (e.g., "Obtain AWS Solutions Architect certification" NOT "Learn cloud computing").
   - Prioritize addressing missing essential skills first.
   - Suggest how extra/additional skills might be leveraged or highlighted.
   - Include 3-7 recommendations, ordered by importance.
   - Reference the specific skill name from the data.
5. If any UNVERIFIED skills are listed, you may mention them in the summary with a caveat like "additionally, the candidate lists [skill] though this could not be verified against the standard taxonomy" — but do NOT factor them into the score.
6. Do NOT generate generic advice. Every recommendation must trace back to a specific skill in the provided data.
"""


def build_analysis_prompt(comparison: ComparisonResult) -> str:
    """Build the full prompt by formatting ComparisonResult into structured context."""
    sections = []

    # Section 1: Matched Skills
    if comparison.matched_skills:
        matched_lines = []
        for s in comparison.matched_skills:
            ctx = f" (found in: {s.context})" if s.context else ""
            matched_lines.append(f"  - {s.standardized_name}{ctx}")
        sections.append(
            "MATCHED SKILLS (candidate has these, and they are required by the JD):\n"
            + "\n".join(matched_lines)
        )
    else:
        sections.append("MATCHED SKILLS: None")

    # Section 2: Missing Essential
    if comparison.missing_essential:
        lines = [f"  - {s.standardized_name}" for s in comparison.missing_essential]
        sections.append(
            "MISSING ESSENTIAL SKILLS (required by JD, absent from resume):\n"
            + "\n".join(lines)
        )
    else:
        sections.append("MISSING ESSENTIAL SKILLS: None")

    # Section 3: Missing Preferred
    if comparison.missing_preferred:
        lines = [f"  - {s.standardized_name}" for s in comparison.missing_preferred]
        sections.append(
            "MISSING PREFERRED SKILLS (nice-to-have in JD, absent from resume):\n"
            + "\n".join(lines)
        )
    else:
        sections.append("MISSING PREFERRED SKILLS: None")

    # Section 4: Extra Skills
    if comparison.extra_skills:
        lines = []
        for s in comparison.extra_skills:
            ctx = f" (found in: {s.context})" if s.context else ""
            lines.append(f"  - {s.standardized_name}{ctx}")
        sections.append(
            "EXTRA SKILLS (candidate has these, but JD does not require them):\n"
            + "\n".join(lines)
        )
    else:
        sections.append("EXTRA SKILLS: None")

    # Section 5: Unverified Skills
    unverified_lines = []
    if comparison.unverified_resume:
        for s in comparison.unverified_resume:
            unverified_lines.append(f"  - [Resume] {s.original_name}")
    if comparison.unverified_jd:
        for s in comparison.unverified_jd:
            unverified_lines.append(f"  - [JD] {s.original_name}")
    if unverified_lines:
        sections.append(
            "UNVERIFIED SKILLS (could not be matched to standard taxonomy — lower confidence):\n"
            + "\n".join(unverified_lines)
        )

    # Section 6: Statistics
    stats = (
        f"STATISTICS:\n"
        f"  - Total matched: {len(comparison.matched_skills)}\n"
        f"  - Total missing essential: {len(comparison.missing_essential)}\n"
        f"  - Total missing preferred: {len(comparison.missing_preferred)}\n"
        f"  - Total extra (resume-only): {len(comparison.extra_skills)}\n"
        f"  - Total unverified: {len(comparison.unverified_resume) + len(comparison.unverified_jd)}"
    )
    sections.append(stats)

    context_block = "\n\n".join(sections)

    prompt = f"""Analyze the following pre-verified skill comparison between a candidate's resume and a job description. Produce your assessment based ONLY on this data.

{context_block}

Based on the above verified skill comparison data, provide your analysis."""

    return prompt


def get_generation_config() -> genai.types.GenerateContentConfig:
    """Return the Gemini config with response schema for analysis generation."""
    return genai.types.GenerateContentConfig(
        system_instruction=ANALYSIS_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=GENERATION_RESPONSE_SCHEMA,
        temperature=0.3,
    )


if __name__ == "__main__":
    comp = ComparisonResult()
    test_prompt = build_analysis_prompt(comp)
    print("Generated Prompt Preview:\n", test_prompt[:300])
