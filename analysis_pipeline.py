import os
import json
import time
from google import genai
from dotenv import load_dotenv

from schemas import (
    ExtractionResult,
    GroundingResult,
    ComparisonResult,
    SufficiencyResult,
    AnalysisOutput
)
from esco_loader import ESCOIndex
from skill_extraction import extract_skills
from skill_grounding import ground_skills
from skill_comparison import compare_skills
from sufficiency_check import check_sufficiency, build_insufficient_response
from analysis_prompt_builder import build_analysis_prompt, get_generation_config

load_dotenv()
PRIMARY_MODEL = "gemini-3.6-flash"
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash"]


class AnalysisPipeline:
    """End-to-end 5-stage Resume x Job Description Gap Analysis Pipeline."""

    def __init__(self, esco_csv_path: str = "data/skills_en.csv"):
        """Initialize pipeline resources (ESCO taxonomy and Gemini client)."""
        print("[AnalysisPipeline] Initializing...")
        self.esco_csv_path = esco_csv_path
        self.esco_index = None

        if os.path.exists(esco_csv_path):
            print(f"[AnalysisPipeline] Loading ESCO skills from '{esco_csv_path}'...")
            self.esco_index = ESCOIndex(esco_csv_path)
            print(f"[AnalysisPipeline] ESCO index loaded ({len(self.esco_index.all_labels)} labels).")
        else:
            print(
                f"[AnalysisPipeline] WARNING: ESCO dataset not found at '{esco_csv_path}'. "
                f"Grounding will run in unverified fallback mode until 'skills_en.csv' is placed in 'data/'."
            )

        self.client = genai.Client()
        print("[AnalysisPipeline] Gemini client ready.")

    def _ensure_esco_loaded(self):
        """Lazy-reload index if file was added after server boot."""
        if self.esco_index is None and os.path.exists(self.esco_csv_path):
            print(f"[AnalysisPipeline] Detecting and loading '{self.esco_csv_path}'...")
            self.esco_index = ESCOIndex(self.esco_csv_path)

    def run(self, resume_text: str, jd_text: str) -> dict:
        """
        Executes the 5-stage pipeline:
        Extraction -> Grounding -> Comparison -> Sufficiency Check -> Generation.
        Returns a dictionary validated against AnalysisOutput schema.
        """
        self._ensure_esco_loaded()

        # ── Stage 1: Extraction ──
        print("[Stage 1] Extracting raw skills from resume and job description...")
        extraction: ExtractionResult = extract_skills(self.client, resume_text, jd_text)
        print(f"  Extracted: {len(extraction.resume_skills)} resume skills | {len(extraction.jd_skills)} JD skills")

        # ── Stage 2: Grounding against ESCO ──
        print("[Stage 2] Grounding skills against ESCO taxonomy...")
        if self.esco_index:
            grounding: GroundingResult = ground_skills(extraction, self.esco_index)
        else:
            # Fallback when taxonomy file is not yet downloaded
            from schemas import GroundedSkill
            grounding = GroundingResult(
                resume_skills=[
                    GroundedSkill(
                        original_name=s.skill_name,
                        standardized_name=s.skill_name,
                        match_type="unverified",
                        context=s.context,
                        priority=s.priority
                    ) for s in extraction.resume_skills
                ],
                jd_skills=[
                    GroundedSkill(
                        original_name=s.skill_name,
                        standardized_name=s.skill_name,
                        match_type="unverified",
                        context=s.context,
                        priority=s.priority
                    ) for s in extraction.jd_skills
                ]
            )

        resume_verified = [s for s in grounding.resume_skills if s.match_type != "unverified"]
        jd_verified = [s for s in grounding.jd_skills if s.match_type != "unverified"]
        print(f"  Verified skills: {len(resume_verified)}/{len(grounding.resume_skills)} resume | "
              f"{len(jd_verified)}/{len(grounding.jd_skills)} JD")

        # ── Stage 3: Comparison ──
        print("[Stage 3] Comparing skill sets...")
        comparison: ComparisonResult = compare_skills(grounding)
        print(f"  Matched: {len(comparison.matched_skills)} | "
              f"Missing essential: {len(comparison.missing_essential)} | "
              f"Missing preferred: {len(comparison.missing_preferred)}")

        # ── Stage 4: Sufficiency Check Gate ──
        print("[Stage 4] Evaluating sufficiency gate...")
        sufficiency: SufficiencyResult = check_sufficiency(comparison, grounding)

        if not sufficiency.passed:
            print(f"  [Sufficiency Gate] Bypassing generation: {sufficiency.reason}")
            output = build_insufficient_response(sufficiency, comparison)
            return output.model_dump()

        print(f"  [Sufficiency Gate] Passed ({sufficiency.resume_grounded_count} resume, "
              f"{sufficiency.jd_grounded_count} JD verified)")

        # ── Stage 5: Generation with Resilient Fallback ──
        print("[Stage 5] Synthesizing gap analysis...")
        prompt = build_analysis_prompt(comparison)
        gen_config = get_generation_config()

        generated = None
        last_err = None

        for model_name in FALLBACK_MODELS:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=gen_config,
                    )
                    generated = json.loads(response.text.strip())
                    break
                except Exception as e:
                    last_err = e
                    if "503" in str(e):
                        time.sleep(1)
                        continue
                    raise e
            if generated is not None:
                break

        if generated is None:
            raise last_err if last_err else RuntimeError("Failed to generate analysis.")

        # Assemble final output merging deterministic and AI-synthesized fields
        output = AnalysisOutput(
            summary=generated["summary"],
            match_score=float(generated["match_score"]),
            matched_skills=[
                {"skill": s.standardized_name, "context": s.context}
                for s in comparison.matched_skills
            ],
            missing_essential=[
                {"skill": s.standardized_name}
                for s in comparison.missing_essential
            ],
            missing_preferred=[
                {"skill": s.standardized_name}
                for s in comparison.missing_preferred
            ],
            extra_skills=[
                {"skill": s.standardized_name, "context": s.context}
                for s in comparison.extra_skills
            ],
            unverified_skills={
                "resume": [s.original_name for s in comparison.unverified_resume],
                "jd": [s.original_name for s in comparison.unverified_jd],
            },
            recommendations=generated["recommendations"],
            generation_ran=True,
            sufficiency_detail=None,
        )

        print(f"[Done] Analysis completed successfully (Match score: {output.match_score}/100)")
        return output.model_dump()


if __name__ == "__main__":
    pipeline = AnalysisPipeline()

    sample_resume = """
    PROFESSIONAL EXPERIENCE
    Senior Software Engineer at CloudCorp (2021-2024)
    - Built microservices using Python, FastAPI, and PostgreSQL.
    - Deployed systems on AWS using EC2, S3, and Lambda.
    - Configured CI/CD automation pipelines via GitHub Actions.

    PROJECTS
    - ML Fraud detection system using TensorFlow and scikit-learn.
    - Frontend dashboard in React.js and TypeScript.

    SKILLS
    Python, JavaScript, TypeScript, React.js, FastAPI, PostgreSQL, AWS, Docker, Git
    """

    sample_jd = """
    Requirements:
    - 4+ years software development experience with Python (essential)
    - Hands-on experience with AWS cloud services (essential)
    - Relational database experience with PostgreSQL (essential)
    - Frontend experience with React (preferred)
    - Docker containerization and Kubernetes knowledge (preferred)
    """

    print("Running test analysis...")
    analysis = pipeline.run(sample_resume, sample_jd)
    print(json.dumps(analysis, indent=2))
