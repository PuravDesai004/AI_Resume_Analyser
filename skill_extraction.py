import json
import time
from typing import Optional
from google import genai
from dotenv import load_dotenv

from schemas import RawSkill, ExtractionResult

load_dotenv()
PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash"]

RESUME_EXTRACTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "The exact skill, technology, tool, methodology, framework, language, or certification name as mentioned in the resume."
            },
            "context": {
                "type": "string",
                "description": "The resume section where this skill was found (e.g., 'Work Experience', 'Projects', 'Skills', 'Education', 'Certifications'). Use section heading if available."
            }
        },
        "required": ["skill_name", "context"]
    }
}

JD_EXTRACTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "skill_name": {
                "type": "string",
                "description": "The exact skill, technology, tool, qualification, framework, language, or certification name as mentioned in the job description."
            },
            "priority": {
                "type": "string",
                "enum": ["essential", "preferred", "unspecified"],
                "description": "Whether this skill is required/mandatory ('essential'), nice-to-have/bonus ('preferred'), or unclear from context ('unspecified')."
            }
        },
        "required": ["skill_name", "priority"]
    }
}

RESUME_SYSTEM_PROMPT = """You are a precise skill extraction engine. Your task is to identify every distinct skill, technology, tool, programming language, framework, methodology, certification, and domain expertise mentioned in the resume text below.

Rules:
1. Extract each skill as a concise, standalone phrase (e.g., "Python", "React.js", "Agile methodology", "AWS Lambda", "PMP certification").
2. Do NOT extract soft skills (e.g., "leadership", "communication", "teamwork") — focus only on technical skills, tools, technologies, certifications, and domain-specific expertise.
3. Do NOT extract job titles, company names, or degree names — only skills.
4. If the same skill appears multiple times in different sections, extract it ONCE and use the most specific/strongest section as the context (prefer "Work Experience" or "Projects" over "Skills" if both exist).
5. For the "context" field, use the exact section heading from the resume if visible (e.g., "Technical Skills", "Professional Experience"). If no heading is visible, infer the section type from the content (e.g., "Work Experience", "Projects", "Education").
6. Extract technology-specific details where meaningful: "React.js" not just "JavaScript", "PostgreSQL" not just "database", "TensorFlow" not just "machine learning".
7. Be exhaustive — do not skip skills that are only mentioned once or briefly.
"""

JD_SYSTEM_PROMPT = """You are a precise requirement extraction engine. Your task is to identify every distinct skill, technology, tool, programming language, framework, methodology, certification, and qualification mentioned in the job description text below.

Rules:
1. Extract each requirement as a concise, standalone phrase (e.g., "Python", "Kubernetes", "CI/CD", "Bachelor's degree in Computer Science", "5+ years experience").
2. Do NOT extract soft skills (e.g., "team player", "self-motivated") — focus only on technical skills, tools, technologies, certifications, and measurable qualifications.
3. Classify each requirement's priority:
   - "essential": The JD uses language like "required", "must have", "mandatory", "minimum", "you will need", or lists it under a "Requirements" / "Must Have" section.
   - "preferred": The JD uses language like "nice to have", "preferred", "bonus", "desirable", "plus", "ideally", or lists it under a "Nice to Have" / "Preferred" section.
   - "unspecified": The priority is ambiguous from context — the skill is mentioned but without clear required/preferred signals.
4. If the same skill appears with different priority signals, use the HIGHEST priority (essential > preferred > unspecified).
5. Extract technology-specific details where meaningful: "React" not just "frontend", "AWS" not just "cloud".
6. Include qualification requirements too: "Bachelor's degree in CS", "3+ years experience with Python", "AWS certification".
7. Be exhaustive — do not skip requirements that are only mentioned once.
"""


def _generate_structured_content(client: genai.Client, contents: str, system_instruction: str, response_schema: dict) -> list[dict]:
    """Helper to execute structured generation with resilient model fallback."""
    last_err = None
    for model_name in FALLBACK_MODELS:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1,
                    )
                )
                text = response.text.strip()
                return json.loads(text)
            except Exception as e:
                last_err = e
                if "503" in str(e):
                    time.sleep(1)
                    continue
                raise e
    raise last_err if last_err else RuntimeError("Extraction failed on all models.")


def extract_resume_skills(client: genai.Client, resume_text: str) -> list[RawSkill]:
    """Extract skills from resume text using Gemini structured output."""
    parsed = _generate_structured_content(
        client=client,
        contents=f"Extract all skills from this resume:\n\n{resume_text}",
        system_instruction=RESUME_SYSTEM_PROMPT,
        response_schema=RESUME_EXTRACTION_SCHEMA
    )
    return [
        RawSkill(
            skill_name=item["skill_name"],
            context=item.get("context"),
            priority="unspecified"
        )
        for item in parsed
    ]


def extract_jd_skills(client: genai.Client, jd_text: str) -> list[RawSkill]:
    """Extract requirements from job description text using Gemini structured output."""
    parsed = _generate_structured_content(
        client=client,
        contents=f"Extract all skill requirements from this job description:\n\n{jd_text}",
        system_instruction=JD_SYSTEM_PROMPT,
        response_schema=JD_EXTRACTION_SCHEMA
    )
    return [
        RawSkill(
            skill_name=item["skill_name"],
            context=None,
            priority=item.get("priority", "unspecified")
        )
        for item in parsed
    ]


def extract_skills(client: genai.Client, resume_text: str, jd_text: str) -> ExtractionResult:
    """Convenience wrapper: extract from both documents, return ExtractionResult."""
    resume_skills = extract_resume_skills(client, resume_text)
    jd_skills = extract_jd_skills(client, jd_text)
    return ExtractionResult(resume_skills=resume_skills, jd_skills=jd_skills)


if __name__ == "__main__":
    test_client = genai.Client()
    sample_resume = "Experience with Python, FastAPI, and Docker in Work Experience section."
    sample_jd = "Required: Python, Kubernetes. Nice to have: Docker."
    print("Testing extraction...")
    result = extract_skills(test_client, sample_resume, sample_jd)
    print("Resume skills:", result.resume_skills)
    print("JD skills:", result.jd_skills)
