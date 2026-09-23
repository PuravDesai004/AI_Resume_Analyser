from google import genai

# =====================================================================
# 1. INGESTION-TIME JD SKILL EXTRACTION
# =====================================================================

JD_SKILL_EXTRACTION_SYSTEM_PROMPT = """# TODO: authored and verified by project owner
You are a specialized Technical Skill Extraction and Classification Engine.
Your primary responsibility is to extract and classify all technical skill requirements from the provided job description text.

=== CORE OBJECTIVES ===
1. JOB DESCRIPTION SKILL CLASSIFICATION:
   Split technical requirements from the Job Description into two distinct lists:
   a. `jd_essential_skills`: Mandatory prerequisites, core requirements, minimum qualifications, or skills explicitly designated with words like "must-have", "required", "essential", "minimum", or listed under a "Requirements" header.
   b. `jd_preferred_skills`: Nice-to-have capabilities, bonus qualifications, or skills labeled with words like "preferred", "plus", "bonus", "desirable", or listed under a "Nice-to-Have" header.
   If a requirement priority is not explicitly stated, classify foundational technologies essential to the primary role responsibilities as essential, and secondary or tangential tooling as preferred.

2. STRICT EVIDENCE & FORMATTING RULES:
   - Extract technical skills, programming languages, software frameworks, libraries, developer tools, databases, cloud platforms, architectures, and technical methodologies explicitly evidenced in the text.
   - Extract each item as a concise, standardized name (e.g., "Python", "FastAPI", "PostgreSQL", "Docker", "AWS", "CI/CD").
   - EXCLUSIONS: Do NOT extract soft skills (e.g., "effective communicator", "team player", "leadership", "problem solver"), job titles (e.g., "Full Stack Engineer"), company names, or educational degrees as skill items.
   - If no technical skills are mentioned in the text, return empty lists.
"""

JD_SKILL_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "jd_essential_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Essential / required skill names found in the job description"
        },
        "jd_preferred_skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Preferred / nice-to-have skill names found in the job description"
        }
    },
    "required": [
        "jd_essential_skills",
        "jd_preferred_skills"
    ]
}


def build_jd_skill_prompt(jd_text: str) -> str:
    """Builds prompt containing the JD text for skill extraction."""
    return f"""Extract the essential and preferred technical skills from the following job description according to your system instructions.

--- JOB DESCRIPTION ---
{jd_text.strip()}
"""


def get_jd_skill_generation_config() -> genai.types.GenerateContentConfig:
    """Returns Gemini GenerateContentConfig for JD skill extraction."""
    return genai.types.GenerateContentConfig(
        system_instruction=JD_SKILL_EXTRACTION_SYSTEM_PROMPT,
        temperature=0.0,
        seed=42,
        response_mime_type="application/json",
        response_schema=JD_SKILL_RESPONSE_SCHEMA
    )


# =====================================================================
# 2. ANALYZE-TIME CANDIDATE & GAP EVALUATION
# =====================================================================

# TODO: authored and verified by project owner
ANALYSIS_SYSTEM_PROMPT = """# TODO: authored and verified by project owner
You are a specialized Candidate-to-Job Alignment Analyzer and Technical Skill Extraction Engine.
Your primary responsibility is to analyze a candidate resume against a job description with extreme factual rigor, zero hallucination, and strict adherence to the evidence present in the text.

You will be given this role's already-determined essential and preferred skills. Treat them as ground truth — do not re-derive or second-guess them. Extract only the CANDIDATE's skills from the resume text provided, then use both to write your summary, strengths, weaknesses, requirement match ratings, and recommendations.

=== CORE OBJECTIVES ===

1. RESUME SKILL EXTRACTION (`resume_skills`):
   - Extract technical skills, programming languages, software frameworks, libraries, developer tools, databases, cloud platforms, architectures, and technical methodologies explicitly evidenced in the resume.
   - Extract each item as a concise, standardized name (e.g., "Python", "FastAPI", "PostgreSQL", "Docker", "AWS S3", "CI/CD").
   - STRICT EVIDENCE RULE: Do NOT infer or assume skills. If a technology is not mentioned, referenced in a project, or listed under technical competencies, DO NOT include it.
   - EXCLUSIONS: Do NOT extract soft skills (e.g., "effective communicator", "team player", "leadership", "problem solver"), job titles (e.g., "Full Stack Engineer"), company names, or educational degrees as skill items.

2. STRICT DIVISION OF LABOR (CRITICAL):
   - Do NOT attempt to calculate a numeric match score.
   - Do NOT attempt to classify which resume skills satisfy which JD skills (no matched/missing/extra classification).
   - An external deterministic taxonomy engine handles exact and semantic grounding against international standards (ESCO) and computes all numeric scores mathematically. Your role is solely accurate extraction and qualitative narrative synthesis.

3. QUALITATIVE ASSESSMENT FIELDS:
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
        "summary",
        "strengths",
        "weaknesses",
        "requirement_match",
        "recommendations"
    ]
}


def build_analysis_prompt(
    resume_text: str,
    jd_text: str,
    jd_essential_skills: list[str],
    jd_preferred_skills: list[str]
) -> str:
    """Builds prompt containing resume, JD, and cached JD skills as ground truth."""
    essential_str = ", ".join(jd_essential_skills) if jd_essential_skills else "None specified"
    preferred_str = ", ".join(jd_preferred_skills) if jd_preferred_skills else "None specified"
    return f"""Analyze the following candidate resume against the target job description according to your system instructions.

--- ROLE ESSENTIAL SKILLS (PRE-DETERMINED TRUTH) ---
{essential_str}

--- ROLE PREFERRED SKILLS (PRE-DETERMINED TRUTH) ---
{preferred_str}

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
    assert "jd_essential_skills" not in props, "jd_essential_skills must NOT be in analyze schema!"
    assert "jd_preferred_skills" not in props, "jd_preferred_skills must NOT be in analyze schema!"

    # Validate ingestion schema
    jd_props = JD_SKILL_RESPONSE_SCHEMA["properties"]
    assert "jd_essential_skills" in jd_props
    assert "jd_preferred_skills" in jd_props
    assert len(jd_props) == 2

    # Validate prompt builders
    prompt = build_analysis_prompt("Resume text here...", "JD text here...", ["Python"], ["Docker"])
    assert "--- CANDIDATE RESUME ---" in prompt
    assert "--- JOB DESCRIPTION ---" in prompt
    assert "--- ROLE ESSENTIAL SKILLS (PRE-DETERMINED TRUTH) ---\nPython" in prompt
    assert "--- ROLE PREFERRED SKILLS (PRE-DETERMINED TRUTH) ---\nDocker" in prompt

    jd_prompt = build_jd_skill_prompt("JD text here...")
    assert "--- JOB DESCRIPTION ---\nJD text here..." in jd_prompt

    # Validate configs
    config = get_generation_config()
    assert config.temperature == 0.0
    assert config.seed == 42
    assert config.response_mime_type == "application/json"

    jd_config = get_jd_skill_generation_config()
    assert jd_config.temperature == 0.0
    assert jd_config.seed == 42
    assert jd_config.response_mime_type == "application/json"

    print("analysis_prompt_builder acceptance criteria passed!")
