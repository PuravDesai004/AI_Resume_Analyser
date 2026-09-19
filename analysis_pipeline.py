import os
import json
from typing import Any
from dotenv import load_dotenv
from google import genai

import jd_index
import ranking
from schemas import (
    DetailAnalysis,
    RequirementMatch,
    Recommendation,
    ErrorResponse
)
from esco_loader import SkillIndex
from skill_grounding import ground_candidates
from skill_comparison import compare_skills
from sufficiency_check import check_sufficiency, build_insufficient_response
from analysis_prompt_builder import build_analysis_prompt, get_generation_config

load_dotenv(override=True)
GEMINI_MODEL = "gemini-3.6-flash"


class AnalysisPipeline:
    """
    Orchestrator for the AI Placement Analyzer.
    Integrates Tier 1 ranking (0 Gemini calls) and Tier 2 detail analysis (exactly 1 Gemini call).
    """

    def __init__(
        self,
        esco_csv_path: str = "data/skills_en.csv",
        custom_json_path: str = "data/custom_skills.json"
    ):
        print("[AnalysisPipeline] Initializing unified SkillIndex...")
        self.skill_index = SkillIndex(
            esco_csv_path=esco_csv_path,
            custom_json_path=custom_json_path
        )
        print(f"[AnalysisPipeline] SkillIndex ready with {len(self.skill_index.all_labels)} labels.")

        api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=api_key)
        print(f"[AnalysisPipeline] Gemini client initialized (Model: {GEMINI_MODEL}).")

    def rank(self, resume_text: str, resume_id: str) -> dict[str, Any]:
        """
        Tier 1: Rank stored JDs against a single resume.
        Guaranteed zero Gemini calls.
        """
        ranking_res = ranking.rank_jds(resume_text=resume_text, resume_id=resume_id)
        return ranking_res.model_dump()

    def analyze(self, resume_text: str, resume_id: str, jd_id: str) -> dict[str, Any]:
        """
        Tier 2: Comprehensive gap analysis.
        Sequence:
        1. Fetch JD record. If not found, return ErrorResponse.
        2. Sufficiency check. If fail, return insufficient response immediately (0 Gemini calls).
        3. Exactly ONE Gemini call to extract candidate skills & narrative fields.
        4. Ground candidate skills (resume, jd_essential, jd_preferred) in code.
        5. Deterministically compare skills & compute match_score in code.
        6. Align 'Skills' requirement_match with match_score.
        7. Merge verified and judged fields into DetailAnalysis.
        Never raises: falls back to fully-shaped DetailAnalysis on unexpected errors.
        """
        try:
            # Step 1: Fetch stored JD record
            jd_record = jd_index.get_jd_record(jd_id)
            if jd_record is None:
                return ErrorResponse(
                    error_code="JD_NOT_FOUND",
                    message=f"Job description with ID '{jd_id}' was not found in the store.",
                    stage="tier2_lookup"
                ).model_dump()

            # Step 2: Sufficiency check gate (0 Gemini calls on failure)
            is_sufficient, reason = check_sufficiency(resume_text, jd_record.full_text)
            if not is_sufficient:
                insufficient_resp = build_insufficient_response(
                    resume_id=resume_id,
                    jd_id=jd_id,
                    reason=reason or "Context insufficient"
                )
                return insufficient_resp.model_dump()

            # Step 3: Exactly ONE Gemini call (with transient retry on 503)
            prompt = build_analysis_prompt(resume_text, jd_record.full_text)
            config = get_generation_config()

            response = None
            last_err = None
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=prompt,
                        config=config
                    )
                    break
                except Exception as e:
                    last_err = e
                    if "503" in str(e) and attempt == 0:
                        import time
                        time.sleep(1.5)
                        continue
                    raise e

            if response is None:
                raise last_err if last_err else RuntimeError("Generation failed")

            response_text = response.text.strip() if response.text else "{}"
            parsed = json.loads(response_text)

            # Step 4: Ground candidates using SkillIndex
            raw_resume_skills = parsed.get("resume_skills", [])
            raw_jd_essential = parsed.get("jd_essential_skills", [])
            raw_jd_preferred = parsed.get("jd_preferred_skills", [])

            grounded_resume = ground_candidates(raw_resume_skills, self.skill_index)
            grounded_essential = ground_candidates(raw_jd_essential, self.skill_index)
            grounded_preferred = ground_candidates(raw_jd_preferred, self.skill_index)

            # Step 5: Deterministic comparison & score calculation
            skill_gap, match_score = compare_skills(
                grounded_resume=grounded_resume,
                grounded_jd_essential=grounded_essential,
                grounded_jd_preferred=grounded_preferred
            )

            # Step 6: Ensure Skills requirement_match reflects deterministic match_score
            req_matches: list[RequirementMatch] = []
            has_skills_entry = False

            for item in parsed.get("requirement_match", []):
                if not isinstance(item, dict):
                    continue
                cat = item.get("category")
                if cat == "Skills":
                    req_matches.append(RequirementMatch(category="Skills", match_percent=match_score))
                    has_skills_entry = True
                elif cat in ("Experience", "Education", "Responsibilities"):
                    raw_pct = item.get("match_percent", 0)
                    pct = max(0, min(100, int(raw_pct)))
                    req_matches.append(RequirementMatch(category=cat, match_percent=pct))

            if not has_skills_entry:
                req_matches.insert(0, RequirementMatch(category="Skills", match_percent=match_score))

            # Recommendations
            recommendations: list[Recommendation] = []
            for r in parsed.get("recommendations", []):
                if isinstance(r, dict) and "skill" in r and "reason" in r:
                    recommendations.append(
                        Recommendation(skill=str(r["skill"]), reason=str(r["reason"]))
                    )

            # Step 7: Merge into final DetailAnalysis
            detail = DetailAnalysis(
                resume_id=resume_id,
                jd_id=jd_id,
                context_sufficient=True,
                insufficient_reason=None,
                match_score=match_score,
                summary=str(parsed.get("summary", "")),
                strengths=[str(s) for s in parsed.get("strengths", [])],
                weaknesses=[str(w) for w in parsed.get("weaknesses", [])],
                skill_gap=skill_gap,
                requirement_match=req_matches,
                recommendations=recommendations
            )

            return detail.model_dump()

        except Exception as exc:
            fallback = build_insufficient_response(
                resume_id=resume_id,
                jd_id=jd_id,
                reason=f"Pipeline processing failed: {str(exc)}"
            )
            return fallback.model_dump()


if __name__ == "__main__":
    print("Testing AnalysisPipeline standalone...")
    pipeline = AnalysisPipeline()

    sample_resume = """
    Experienced Senior Backend Engineer with 5+ years designing distributed systems.
    Core competencies: Python, FastAPI, PostgreSQL, Redis, and Docker.
    Built and deployed microservices on AWS (EC2, S3, RDS) using GitHub Actions CI/CD.
    Bachelor of Science in Computer Science.
    """

    sample_jd = """
    Senior Python Developer
    Acme Tech Solutions - Remote

    Requirements:
    - 4+ years of professional software development experience.
    - Strong proficiency in Python, FastAPI, and PostgreSQL (Essential).
    - Experience with Docker containerization and Redis caching (Essential).
    - Familiarity with AWS cloud infrastructure and Kubernetes (Preferred).
    - Bachelor's degree in Computer Science or related field (Essential).
    """

    # Add sample JD to store for test
    add_res = jd_index.add_jd(
        title="Senior Python Developer",
        company="Acme Tech",
        location="Remote",
        jd_text=sample_jd
    )

    if isinstance(add_res, ErrorResponse):
        print("JD Add Error:", add_res.message)
        test_jd_id = "existing"
    else:
        test_jd_id = add_res.jd_id
        print(f"Added test JD: {test_jd_id}")

    print("\n--- Running Tier 1 Rank ---")
    rank_output = pipeline.rank(sample_resume, "resume_test_01")
    print("Rank Output:", json.dumps(rank_output, indent=2))

    print("\n--- Running Tier 2 Analyze ---")
    analyze_output = pipeline.analyze(sample_resume, "resume_test_01", test_jd_id)
    print("Analyze Output:", json.dumps(analyze_output, indent=2))
