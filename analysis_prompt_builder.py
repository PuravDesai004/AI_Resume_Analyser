from google import genai

# TODO: authored and verified by project owner
ANALYSIS_SYSTEM_PROMPT = """# TODO: authored and verified by project owner
You are a specialized Candidate-to-Job Alignment Analyzer and Technical Skill Extraction Engine.
Your primary responsibility is to analyze a candidate resume against a job description with extreme factual rigor, zero hallucination, and strict adherence to the evidence present in the text.

=== CORE OBJECTIVES ===

1. RESUME SKILL EXTRACTION (`resume_skills`):
   - Extract technical skills, programming languages, software frameworks, libraries, developer tools, databases, cloud platforms, architectures, and technical methodologies explicitly evidenced in the resume.
   - Extract each item as a concise, standardized name (e.g., "Python", "FastAPI", "PostgreSQL", "Docker", "AWS S3", "CI/CD").
   - STRICT EVIDENCE RULE: Do NOT infer or assume skills. If a technology is not mentioned, referenced in a project, or listed under technical competencies, DO NOT include it.
   - EXCLUSIONS: Do NOT extract soft skills (e.g., "effective communicator", "team player", "leadership", "problem solver"), job titles (e.g., "Full Stack Engineer"), company names, or educational degrees as skill items.

2. JOB DESCRIPTION SKILL CLASSIFICATION:
   - Split technical requirements from the Job Description into two distinct lists:
     a. `jd_essential_skills`: Mandatory prerequisites, core requirements, minimum qualifications, or skills explicitly designated with words like "must-have", "required", "essential", "minimum", or listed under a "Requirements" header.
     b. `jd_preferred_skills`: Nice-to-have capabilities, bonus qualifications, or skills labeled with words like "preferred", "plus", "bonus", "desirable", or listed under a "Nice-to-Have" header.
   - If a requirement priority is not explicitly stated, classify foundational technologies essential to the primary role responsibilities as essential, and secondary or tangential tooling as preferred.

3. STRICT DIVISION OF LABOR (CRITICAL):
   - Do NOT attempt to calculate a numeric match score.
   - Do NOT attempt to classify which resume skills satisfy which JD skills (no matched/missing/extra classification).
   - An external deterministic taxonomy engine handles exact and semantic grounding against international standards (ESCO) and computes all numeric scores mathematically. Your role is solely accurate extraction and qualitative narrative synthesis.

4. QUALITATIVE ASSESSMENT FIELDS:
   - `summary`: Provide a 2-4 sentence executive overview evaluating candidate alignment with the target role. State role seniority fit, primary areas of alignment, and prominent qualification gaps in a neutral, objective tone.
   - `strengths`: Provide 2-5 concrete bullet points highlighting qualifications, projects, or experiences where the candidate directly satisfies or exceeds key role expectations based on explicit resume evidence.
   - `weaknesses`: Provide 2-5 specific, factual bullet points highlighting requirements, certifications, experience depths, or tooling that the JD demands but the candidate's resume lacks.
   - `requirement_match`: Provide qualitative fit percentages (0 to 100 integer) for role dimensions:
     * "Experience": Alignment with required years of experience, seniority, and domain context.
     * "Education": Alignment with required degree level or technical academic background.
     * "Responsibilities": Candidate's proven track record handling similar daily job duties.
     * (Note: "Skills" match percentage will be calculated independently by the deterministic engine).
   - `recommendations`: Provide 3-5 specific, actionable recommendations for the candidate. Each recommendation must feature:
     * `skill`: The concrete skill, tool, or domain to address.
     * `reason`: Clear justification explaining how acquiring, certifying in, or highlighting this skill closes a critical gap in the target role.
"""

GENERATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "resume_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Candidate skill names found in the resume"
        },
        "jd_essential_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Essential / required skill names found in the job description"
        },
        "jd_preferred_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Preferred / nice-to-have skill names found in the job description"
        },
        "summary": {
            "type": "string",
            "description": "Narrative summary of the candidate's alignment with the role"
        },
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Candidate strengths relative to the job requirements"
        },
        "weaknesses": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Identified weaknesses or requirement gaps"
        },
        "requirement_match": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["Skills", "Experience", "Education", "Responsibilities"]
                    },
                    "match_percent": {
                        "type": "integer",
                        "description": "Estimated match percentage (0 to 100)"
                    }
                },
                "required": ["category", "match_percent"]
            },
            "description": "Qualitative match percentages across categories"
        },
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["skill", "reason"]
            },
            "description": "Actionable recommendations for improvement"
        }
    },
    "required": [
        "resume_skills",
        "jd_essential_skills",
        "jd_preferred_skills",
        "summary",
        "strengths",
        "weaknesses",
        "requirement_match",
        "recommendations"
    ]
}


def build_analysis_prompt(resume_text: str, jd_text: str) -> str:
    """Builds prompt containing the full text of both documents."""
    return f"""Analyze the following resume and job description according to your system instructions.

--- CANDIDATE RESUME ---
{resume_text.strip()}

--- JOB DESCRIPTION ---
{jd_text.strip()}
"""


def get_generation_config() -> genai.types.GenerateContentConfig:
    """Returns Gemini GenerateContentConfig with structured response schema."""
    return genai.types.GenerateContentConfig(
        system_instruction=ANALYSIS_SYSTEM_PROMPT,
        temperature=0.0,
        seed=42,
        response_mime_type="application/json",
        response_schema=GENERATION_RESPONSE_SCHEMA
    )


if __name__ == "__main__":
    # Validate that no decision fields exist in schema
    props = GENERATION_RESPONSE_SCHEMA["properties"]
    assert "match_score" not in props, "match_score must NOT be in Gemini response schema!"
    assert "matched" not in props, "matched skills must NOT be in Gemini response schema!"
    assert "missing" not in props, "missing skills must NOT be in Gemini response schema!"
    assert "extra" not in props, "extra skills must NOT be in Gemini response schema!"
    assert "skill_gap" not in props, "skill_gap must NOT be in Gemini response schema!"

    prompt = build_analysis_prompt("Resume text here...", "JD text here...")
    assert "--- CANDIDATE RESUME ---" in prompt
    assert "--- JOB DESCRIPTION ---" in prompt

    config = get_generation_config()
    assert config.temperature == 0.0
    assert config.seed == 42
    assert config.response_mime_type == "application/json"
    print("analysis_prompt_builder acceptance criteria passed!")
