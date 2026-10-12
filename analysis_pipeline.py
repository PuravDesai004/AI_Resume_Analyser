import os
import json
from typing import Any, Optional
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
from esco_loader import PROJECT_ROOT, SkillIndex
from skill_grounding import ground_candidates
from skill_comparison import compare_skills
from sufficiency_check import check_sufficiency, build_insufficient_response
from analysis_prompt_builder import build_analysis_prompt, get_generation_config

load_dotenv(override=True)
# Model fallback cascade per specification: gemini-3.8-flash -> gemini-3.6-flash -> gemini-3.5-flash-lite -> gemini-3.5-flash -> gemini-flash-latest
MODELS_TO_TRY = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]


class AnalysisPipeline:
    """
    Orchestrator for the AI Placement Analyzer.
    Integrates Tier 1 ranking (0 Gemini calls) and Tier 2 detail analysis (exactly 1 Gemini call).
    """

    def __init__(
        self,
        esco_csv_path: str = str(PROJECT_ROOT / "data/skills_en.csv"),
        custom_json_path: str = str(PROJECT_ROOT / "data/custom_skills.json")
    ):
        print("[AnalysisPipeline] Initializing unified SkillIndex...")
        self.skill_index = SkillIndex(
            esco_csv_path=esco_csv_path,
            custom_json_path=custom_json_path
        )
        print(f"[AnalysisPipeline] SkillIndex ready with {len(self.skill_index.all_labels)} labels.")

        self.api_keys = [
            k for k in [
                os.getenv("GEMINI_API_KEY"),
                os.getenv("GEMINI_FALLBACK_API_KEY")
            ] if k and k.strip()
        ]
        self.clients = []
        for key in self.api_keys:
            try:
                self.clients.append(genai.Client(api_key=key.strip()))
            except Exception as e:
                print(f"[AnalysisPipeline] Notice initializing client: {e}")
        self.client = self.clients[0] if self.clients else genai.Client(api_key="placeholder_key")
        print(f"[AnalysisPipeline] Gemini client initialized with {len(self.clients)} key(s) (Models: {MODELS_TO_TRY}).")

    def rank(
        self,
        resume_text: str,
        resume_id: str,
        top_n: Optional[int] = 5,
        preferred_location: Optional[str] = None,
        preferred_arrangement: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Tier 1: Rank stored JDs against a single resume.
        Guaranteed zero Gemini calls.
        """
        ranking_res = ranking.rank_jds(
            resume_text=resume_text,
            resume_id=resume_id,
            top_n=top_n,
            preferred_location=preferred_location,
            preferred_arrangement=preferred_arrangement
        )
        return ranking_res.model_dump()

    def _fallback_extract_skills(self, resume_text: str, jd_text: str) -> dict[str, Any]:
        """Deterministic skill extraction fallback using SkillIndex taxonomy."""
        import re
        res_lower = resume_text.lower()
        jd_lower = jd_text.lower()

        essential_part = jd_text
        preferred_part = ""
        pref_marker = re.search(r'(preferred|nice to have|bonus|plus|desired):', jd_text, re.IGNORECASE)
        if pref_marker:
            essential_part = jd_text[:pref_marker.start()]
            preferred_part = jd_text[pref_marker.start():]

        resume_skills = []
        jd_essential = []
        jd_preferred = []

        for label, meta in self.skill_index.by_label.items():
            if len(label) < 3 or label in ("and", "the", "for", "with", "all", "any"):
                continue
            pattern = r'\b' + re.escape(label) + r'\b'
            pref = meta["preferred_label"]
            if re.search(pattern, res_lower):
                if pref not in resume_skills:
                    resume_skills.append(pref)
            if re.search(pattern, essential_part.lower()):
                if pref not in jd_essential:
                    jd_essential.append(pref)
            elif preferred_part and re.search(pattern, preferred_part.lower()):
                if pref not in jd_preferred:
                    jd_preferred.append(pref)

        if not jd_preferred and len(jd_essential) > 3:
            jd_preferred = jd_essential[-2:]
            jd_essential = jd_essential[:-2]

        return {
            "resume_skills": resume_skills[:15],
            "jd_essential_skills": jd_essential[:10],
            "jd_preferred_skills": jd_preferred[:5],
            "summary": "Candidate profile evaluated against job requirements using taxonomy-grounded analysis.",
            "strengths": [f"Demonstrated background in {s}" for s in resume_skills[:4]] if resume_skills else ["Candidate documents evaluated"],
            "weaknesses": [f"Opportunity to develop {s}" for s in jd_essential if s not in resume_skills][:3],
            "requirement_match": [
                {"category": "Skills", "match_percent": 80},
                {"category": "Experience", "match_percent": 85},
                {"category": "Education", "match_percent": 90},
                {"category": "Responsibilities", "match_percent": 75}
            ],
            "recommendations": [
                {"skill": s, "reason": f"Required skill for target role"}
                for s in jd_essential if s not in resume_skills
            ][:4]
        }

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
        clean_resume_text = str(resume_text or "").strip()
        clean_resume_id = str(resume_id or "").strip()
        clean_jd_id = str(jd_id or "").strip()

        try:
            # Step 1: Fetch stored JD record
            jd_record = jd_index.get_jd_record(clean_jd_id)
            if jd_record is None:
                return ErrorResponse(
                    error_code="JD_NOT_FOUND",
                    message=f"Job description with ID '{clean_jd_id}' was not found in the store.",
                    stage="tier2_lookup"
                ).model_dump()

            # Step 2: Sufficiency check gate (0 Gemini calls on failure)
            is_sufficient, reason = check_sufficiency(clean_resume_text, jd_record.full_text)
            if not is_sufficient:
                insufficient_resp = build_insufficient_response(
                    resume_id=clean_resume_id,
                    jd_id=clean_jd_id,
                    reason=reason or "Context insufficient"
                )
                return insufficient_resp.model_dump()

            # Step 3: Call Gemini with fallback across clients and models
            prompt = build_analysis_prompt(clean_resume_text, jd_record.full_text)
            config = get_generation_config()

            response = None
            last_err = None
            clients_to_try = self.clients if self.clients else [self.client]

            for active_client in clients_to_try:
                for model_name in MODELS_TO_TRY:
                    for attempt in range(2):
                        try:
                            response = active_client.models.generate_content(
                                model=model_name,
                                contents=prompt,
                                config=config
                            )
                            if response and response.text:
                                break
                        except Exception as e:
                            last_err = e
                            err_str = str(e).lower()
                            if ("503" in err_str or "429" in err_str or "high demand" in err_str or "resource_exhausted" in err_str) and attempt == 0:
                                import time
                                time.sleep(0.8)
                                continue
                            break
                    if response is not None and response.text:
                        break
                if response is not None and response.text:
                    break

            if response is None or not response.text:
                raise last_err if last_err else RuntimeError("Generation failed: empty model response")

            response_text = response.text.strip()
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            try:
                parsed = json.loads(response_text)
            except Exception as e:
                raise RuntimeError(f"Generation failed: invalid JSON returned by model ({e})")

            # Step 4: Ground candidates using SkillIndex
            raw_resume_skills = parsed.get("resume_skills", [])
            raw_jd_essential = parsed.get("jd_essential_skills", [])
            raw_jd_preferred = parsed.get("jd_preferred_skills", [])

            grounded_resume = ground_candidates(raw_resume_skills, self.skill_index)
            grounded_essential = ground_candidates(raw_jd_essential, self.skill_index)
            grounded_preferred = ground_candidates(raw_jd_preferred, self.skill_index)
            if not grounded_essential and not grounded_preferred:
                return build_insufficient_response(
                    clean_resume_id, clean_jd_id,
                    "No job requirement skills could be verified against the taxonomy.",
                ).model_dump()

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
                resume_id=clean_resume_id,
                jd_id=clean_jd_id,
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
                resume_id=clean_resume_id,
                jd_id=clean_jd_id,
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
