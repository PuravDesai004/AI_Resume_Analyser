import os
import json
import time
from typing import Any, Union
from dotenv import load_dotenv
from google import genai

import jd_index
import ranking
from schemas import (
    DetailAnalysis,
    RequirementMatch,
    Recommendation,
    ErrorResponse,
    JDRecord,
    GroundedSkill
)
from esco_loader import SkillIndex
from skill_grounding import ground_candidates
from skill_comparison import compare_skills
from sufficiency_check import check_sufficiency, build_insufficient_response
from analysis_prompt_builder import (
    build_analysis_prompt,
    get_generation_config,
    build_jd_skill_prompt,
    get_jd_skill_generation_config
)

load_dotenv(override=True)
MODELS_TO_TRY = ["gemini-3.5-flash", "gemini-3.6-flash"]


class AnalysisPipeline:
    """
    Orchestrator for the AI Placement Analyzer.
    Integrates Tier 1 ranking (0 Gemini calls) and Tier 2 detail analysis (exactly 1 Gemini call).
    Caches JD skill extraction at ingestion time.
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
        print(f"[AnalysisPipeline] Gemini client initialized (Models: {MODELS_TO_TRY}).")

    def rank(self, resume_text: str, resume_id: str) -> dict[str, Any]:
        """
        Tier 1: Rank stored JDs against a single resume.
        Guaranteed zero Gemini calls.
        """
        ranking_res = ranking.rank_jds(resume_text=resume_text, resume_id=resume_id)
        return ranking_res.model_dump()

    def _extract_and_ground_jd_skills(
        self,
        jd_text: str
    ) -> tuple[list[GroundedSkill], list[GroundedSkill], bool]:
        """
        Extracts essential and preferred skills from JD text using Gemini, then grounds them.
        Returns:
            (essential_skills, preferred_skills, success)
        success is True only when Gemini call and JSON parse both complete without error.
        On any exception, returns ([], [], False).
        """
        print("[AnalysisPipeline] Extracting & grounding JD skills...")
        try:
            prompt = build_jd_skill_prompt(jd_text)
            config = get_jd_skill_generation_config()

            response = None
            last_err = None
            for model_name in MODELS_TO_TRY:
                for attempt in range(2):
                    try:
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=config
                        )
                        if response:
                            break
                    except Exception as e:
                        last_err = e
                        if ("503" in str(e) or "429" in str(e)) and attempt == 0:
                            time.sleep(1.0)
                            continue
                        break
                if response is not None:
                    break

            if response is None:
                raise last_err if last_err else RuntimeError("JD skill extraction call failed")

            response_text = response.text.strip() if response.text else "{}"
            parsed = json.loads(response_text)

            raw_essential = parsed.get("jd_essential_skills", [])
            raw_preferred = parsed.get("jd_preferred_skills", [])

            grounded_essential = ground_candidates(raw_essential, self.skill_index)
            grounded_preferred = ground_candidates(raw_preferred, self.skill_index)

            return grounded_essential, grounded_preferred, True

        except Exception as e:
            print(f"[AnalysisPipeline] Warning: JD skill extraction failed ({e}), returning unverified empty lists.")
            return [], [], False

    def ingest_jd(
        self,
        title: str,
        company: str,
        location: str,
        jd_text: str
    ) -> Union[JDRecord, ErrorResponse]:
        """
        Ingests a JD into the store:
        1. Adds JD to ChromaDB store (embeds once, enforces cap of 15).
        2. Extracts and grounds JD essential and preferred skills.
        3. Persists grounded skills to ChromaDB metadata if extraction succeeded.
        4. Returns the JDRecord.
        """
        jd_record = jd_index.add_jd(
            title=title,
            company=company,
            location=location,
            jd_text=jd_text
        )
        if isinstance(jd_record, ErrorResponse):
            return jd_record

        essential, preferred, success = self._extract_and_ground_jd_skills(jd_text)
        if success:
            jd_index.update_jd_skills(jd_record.jd_id, essential, preferred)
            jd_record.essential_skills = essential
            jd_record.preferred_skills = preferred
            jd_record.skills_cached = True

        return jd_record

    def analyze(self, resume_text: str, resume_id: str, jd_id: str) -> dict[str, Any]:
        """
        Tier 2: Comprehensive gap analysis.
        Sequence:
        1. Fetch JD record. If not found, return ErrorResponse.
        2. Sufficiency check. If fail, return insufficient response immediately (0 Gemini calls).
        3. If skills_cached is False, trigger lazy backfill once.
        4. Exactly ONE Gemini call to extract candidate resume skills & narrative fields.
        5. Ground candidate resume skills in code.
        6. Deterministically compare skills & compute match_score in code.
        7. Align 'Skills' requirement_match with match_score.
        8. Merge verified and judged fields into DetailAnalysis.
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

            # Step 3: Lazy backfill if JD skills have not been cached yet
            if not jd_record.skills_cached:
                essential, preferred, success = self._extract_and_ground_jd_skills(jd_record.full_text)
                if success:
                    jd_index.update_jd_skills(jd_record.jd_id, essential, preferred)
                    jd_record.essential_skills = essential
                    jd_record.preferred_skills = preferred
                    jd_record.skills_cached = True

            # Step 4: Exactly ONE Gemini call for candidate skills & narrative
            cached_essential_names = [s.standardized_name for s in jd_record.essential_skills]
            cached_preferred_names = [s.standardized_name for s in jd_record.preferred_skills]

            prompt = build_analysis_prompt(
                resume_text=resume_text,
                jd_text=jd_record.full_text,
                jd_essential_skills=cached_essential_names,
                jd_preferred_skills=cached_preferred_names
            )
            config = get_generation_config()

            response = None
            last_err = None
            for model_name in MODELS_TO_TRY:
                for attempt in range(2):
                    try:
                        response = self.client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=config
                        )
                        if response:
                            break
                    except Exception as e:
                        last_err = e
                        if ("503" in str(e) or "429" in str(e)) and attempt == 0:
                            time.sleep(1.0)
                            continue
                        break
                if response is not None:
                    break

            if response is None:
                raise last_err if last_err else RuntimeError("Generation failed")

            response_text = response.text.strip() if response.text else "{}"
            parsed = json.loads(response_text)

            # Step 5: Ground ONLY candidate resume skills
            raw_resume_skills = parsed.get("resume_skills", [])
            grounded_resume = ground_candidates(raw_resume_skills, self.skill_index)

            # Step 6: Deterministic comparison & score calculation using cached JD skills
            skill_gap, match_score = compare_skills(
                grounded_resume=grounded_resume,
                grounded_jd_essential=jd_record.essential_skills,
                grounded_jd_preferred=jd_record.preferred_skills
            )

            # Step 7: Ensure Skills requirement_match reflects deterministic match_score
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

            # Step 8: Merge into final DetailAnalysis
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

    # Ingest sample JD to store with cached extraction
    ingest_res = pipeline.ingest_jd(
        title="Senior Python Developer",
        company="Acme Tech",
        location="Remote",
        jd_text=sample_jd
    )

    if isinstance(ingest_res, ErrorResponse):
        print("JD Ingest Error:", ingest_res.message)
        test_jd_id = "existing"
    else:
        test_jd_id = ingest_res.jd_id
        print(f"Ingested test JD: {test_jd_id} (skills_cached: {ingest_res.skills_cached})")
        print(f"Essential skills ({len(ingest_res.essential_skills)}): {[s.standardized_name for s in ingest_res.essential_skills]}")
        print(f"Preferred skills ({len(ingest_res.preferred_skills)}): {[s.standardized_name for s in ingest_res.preferred_skills]}")

    print("\n--- Running Tier 1 Rank ---")
    rank_output = pipeline.rank(sample_resume, "resume_test_01")
    print("Rank Output:", json.dumps(rank_output, indent=2))

    print("\n--- Running Tier 2 Analyze ---")
    analyze_output = pipeline.analyze(sample_resume, "resume_test_01", test_jd_id)
    print("Analyze Output:", json.dumps(analyze_output, indent=2))
