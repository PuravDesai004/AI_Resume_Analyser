import os
import sys
import json
import time
import unittest
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schemas import (
    GroundedSkill,
    DetailAnalysis,
    ErrorResponse,
    JDRecord,
    RankingResult
)
from esco_loader import SkillIndex, normalize_label
from skill_grounding import ground_candidates
from skill_comparison import compare_skills
from sufficiency_check import check_sufficiency, build_insufficient_response
import jd_index
import ranking
from analysis_pipeline import AnalysisPipeline


class TestPlacementAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skill_index = SkillIndex(
            esco_csv_path="data/skills_en.csv",
            custom_json_path="data/custom_skills.json"
        )
        with open("tests/fixtures/sample_data.json", "r", encoding="utf-8") as f:
            cls.fixtures = json.load(f)

    @classmethod
    def tearDownClass(cls):
        jd_index.clear_store_for_testing()

    # ── 1. Index Coverage & Normalization ──

    def test_index_coverage(self):
        stats = self.skill_index.stats()
        self.assertGreater(stats["esco"], 0)
        self.assertGreater(stats["custom"], 0)
        self.assertGreater(stats["total_labels"], 0)

        # 10 hand-picked tech skills must all resolve
        tech_skills = [
            "Python", "JavaScript", "FastAPI", "Docker", "PostgreSQL",
            "Kubernetes", "React", "PyTorch", "Redis", "Git"
        ]
        for skill in tech_skills:
            res = self.skill_index.exact_match(skill)
            self.assertIsNotNone(res, f"Skill '{skill}' failed to resolve in SkillIndex!")

    def test_normalization(self):
        match1 = self.skill_index.exact_match("Python")
        match2 = self.skill_index.exact_match("  python  ")
        match3 = self.skill_index.exact_match("PYTHON")
        self.assertIsNotNone(match1)
        self.assertEqual(match1, match2)
        self.assertEqual(match2, match3)

    # ── 2. Candidate Grounding ──

    def test_grounding_exact(self):
        candidates = ["Python"]
        grounded = ground_candidates(candidates, self.skill_index)
        self.assertEqual(len(grounded), 1)
        self.assertEqual(grounded[0].match_type, "exact")
        self.assertEqual(grounded[0].source, "esco")

    def test_grounding_fuzzy(self):
        candidates = ["React.js"]
        grounded = ground_candidates(candidates, self.skill_index)
        self.assertEqual(len(grounded), 1)
        self.assertIn(grounded[0].match_type, ["exact", "close"])
        self.assertEqual(grounded[0].standardized_name, "React")

    def test_grounding_discard(self):
        candidates = ["built distributed systems", "hard worker", "fast learner"]
        grounded = ground_candidates(candidates, self.skill_index)
        self.assertEqual(len(grounded), 0, "Noise phrases must be discarded entirely!")

    def test_custom_list_coverage(self):
        match = self.skill_index.exact_match("FastAPI")
        self.assertIsNotNone(match)
        self.assertEqual(match["source"], "custom")
        self.assertEqual(match["skill_id"], "custom:fastapi")

    # ── 3. Comparison & Determinism ──

    def test_comparison_correctness(self):
        r_skills = [
            GroundedSkill(original_name="Python", standardized_name="Python", skill_id="custom:python", source="custom", match_type="exact"),
            GroundedSkill(original_name="FastAPI", standardized_name="FastAPI", skill_id="custom:fastapi", source="custom", match_type="exact"),
            GroundedSkill(original_name="Redis", standardized_name="Redis", skill_id="custom:redis", source="custom", match_type="close"),
            GroundedSkill(original_name="ExtraSkill", standardized_name="ExtraSkill", skill_id="custom:extra", source="custom", match_type="exact")
        ]
        jd_essential = [
            GroundedSkill(original_name="Python", standardized_name="Python", skill_id="custom:python", source="custom", match_type="exact"),
            GroundedSkill(original_name="Docker", standardized_name="Docker", skill_id="custom:docker", source="custom", match_type="exact")
        ]
        jd_preferred = [
            GroundedSkill(original_name="FastAPI", standardized_name="FastAPI", skill_id="custom:fastapi", source="custom", match_type="exact"),
            GroundedSkill(original_name="Kubernetes", standardized_name="Kubernetes", skill_id="custom:kubernetes", source="custom", match_type="exact")
        ]

        gap, score = compare_skills(r_skills, jd_essential, jd_preferred)
        # Essential: Python matched (1), Docker missing (1) -> 50%
        # Preferred: FastAPI matched (1), Kubernetes missing (1) -> 50%
        # Score = 100 * (0.8 * 0.5 + 0.2 * 0.5) = 50
        self.assertEqual(score, 50)
        self.assertEqual(gap.missing_essential, ["Docker"])
        self.assertEqual(gap.missing_preferred, ["Kubernetes"])
        self.assertIn("ExtraSkill", gap.extra)
        self.assertIn("Redis", gap.extra)

    def test_score_determinism(self):
        r_skills = [
            GroundedSkill(original_name="Python", standardized_name="Python", skill_id="custom:python", source="custom", match_type="exact"),
            GroundedSkill(original_name="FastAPI", standardized_name="FastAPI", skill_id="custom:fastapi", source="custom", match_type="exact")
        ]
        jd_essential = [
            GroundedSkill(original_name="Python", standardized_name="Python", skill_id="custom:python", source="custom", match_type="exact")
        ]
        jd_preferred = []

        scores = []
        for _ in range(10):
            gap, score = compare_skills(r_skills, jd_essential, jd_preferred)
            scores.append(score)

        self.assertEqual(len(set(scores)), 1, "Score must be completely deterministic across 10 runs!")
        self.assertEqual(scores[0], 100)

    # ── 4. JD Store & Cap Enforcement ──

    def test_cap_enforcement_and_one_time_embedding(self):
        # Reset store for test
        jd_index.clear_store_for_testing()
        self.assertEqual(jd_index.count(), 0)

        # Mock embedding manager to track calls and run offline
        fake_vector = [0.1] * 3072
        with patch.object(jd_index, "_get_embedder") as mock_get_embedder:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = fake_vector
            mock_get_embedder.return_value = mock_emb

            # Add 15 JDs
            for i in range(15):
                rec = jd_index.add_jd(
                    title=f"Test Role {i+1}",
                    company="TestCo",
                    location="Remote",
                    jd_text=f"Job description text for position number {i+1}"
                )
                self.assertIsInstance(rec, JDRecord)

            self.assertEqual(jd_index.count(), 15)
            self.assertEqual(mock_emb.embed_chunk.call_count, 15)

            # Attempt 16th JD -> must return ErrorResponse without touching store
            rec_16 = jd_index.add_jd(
                title="16th Role",
                company="TestCo",
                location="Remote",
                jd_text="This 16th JD should be rejected by the cap."
            )
            self.assertIsInstance(rec_16, ErrorResponse)
            self.assertEqual(rec_16.error_code, "STORE_CAP_REACHED")
            self.assertEqual(jd_index.count(), 15)
            # Embed chunk should NOT have been called for rejected 16th JD
            self.assertEqual(mock_emb.embed_chunk.call_count, 15)

    # ── 5. Ranking (Tier 1) & Zero Gemini Calls ──

    def test_ranking_correctness_and_zero_gemini(self):
        jd_index.clear_store_for_testing()

        # Add 2 JDs with distinct synthetic embeddings
        vec_py = [1.0] + [0.0] * 3071
        vec_front = [0.0, 1.0] + [0.0] * 3070

        with patch.object(jd_index, "_get_embedder") as mock_idx_emb:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.side_effect = [vec_py, vec_front]
            mock_idx_emb.return_value = mock_emb

            rec_py = jd_index.add_jd("Python Role", "Acme", "Remote", "Python Backend Dev")
            rec_front = jd_index.add_jd("Frontend Role", "Acme", "Remote", "React Frontend Dev")

        # Now rank with a Python resume (vector close to vec_py)
        with patch.object(ranking, "_get_embedder") as mock_rank_emb, \
             patch("google.genai.models.Models.generate_content") as mock_gen:

            mock_emb2 = MagicMock()
            mock_emb2.embed_chunk.return_value = [0.95, 0.05] + [0.0] * 3070
            mock_rank_emb.return_value = mock_emb2

            res = ranking.rank_jds("Python developer with FastAPI experience", "res_001")

            # Must never call Gemini
            mock_gen.assert_not_called()

            self.assertEqual(len(res.results), 2)
            self.assertEqual(res.results[0].jd_id, rec_py.jd_id)
            self.assertGreater(res.results[0].similarity_score, res.results[1].similarity_score)

    # ── 6. Sufficiency Gate ──

    def test_sufficiency_gate(self):
        short_text = "I know Python."
        normal_text = " ".join(["word"] * 35)

        # Short resume
        ok, reason = check_sufficiency(short_text, normal_text, min_words=30)
        self.assertFalse(ok)
        self.assertIn("Resume text is too short", reason)

        # Normal text
        ok, reason = check_sufficiency(normal_text, normal_text, min_words=30)
        self.assertTrue(ok)
        self.assertIsNone(reason)

        # Fully-shaped response check
        bad_resp = build_insufficient_response("res_1", "jd_1", "Short text")
        self.assertFalse(bad_resp.context_sufficient)
        self.assertEqual(bad_resp.match_score, 0)
        self.assertIsInstance(bad_resp, DetailAnalysis)

    # ── 7. Pipeline Guarantees & Gemini Handling ──

    def test_pipeline_single_gemini_call_and_no_trust_in_fabricated_fields(self):
        pipeline = AnalysisPipeline(
            esco_csv_path="data/skills_en.csv",
            custom_json_path="data/custom_skills.json"
        )

        jd_index.clear_store_for_testing()
        with patch.object(jd_index, "_get_embedder") as mock_idx_emb:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = [0.1] * 3072
            mock_idx_emb.return_value = mock_emb
            jd_rec = jd_index.add_jd("Backend Role", "Co", "Remote", " ".join(["python"] * 40))

        resume_text = " ".join(["Experienced developer with Python and FastAPI skills."] * 10)

        # Fabricate a response from Gemini that includes illegal decision fields
        fabricated_gemini_output = {
            "resume_skills": ["Python", "FastAPI"],
            "jd_essential_skills": ["Python", "Docker"],
            "jd_preferred_skills": ["FastAPI"],
            "summary": "Candidate matches backend requirements.",
            "strengths": ["Strong Python"],
            "weaknesses": ["Missing Docker"],
            "requirement_match": [
                {"category": "Skills", "match_percent": 99}, # Fabricated!
                {"category": "Experience", "match_percent": 75}
            ],
            "recommendations": [{"skill": "Docker", "reason": "Mandatory for containerization"}],
            # Malicious/fabricated fields that must be completely ignored
            "match_score": 999,
            "matched": ["Everything"],
            "missing": []
        }

        mock_resp = MagicMock()
        mock_resp.text = json.dumps(fabricated_gemini_output)

        with patch.object(pipeline.client.models, "generate_content", return_value=mock_resp) as mock_gen:
            analysis = pipeline.analyze(resume_text, "res_001", jd_rec.jd_id)

            # Exactly ONE call made
            self.assertEqual(mock_gen.call_count, 1)

            # Ignored fabricated fields
            self.assertNotEqual(analysis["match_score"], 999)
            # True score: Essential has Python (matched) & Docker (missing) -> 50%
            # Preferred has FastAPI (matched) -> 100%
            # Match score = round(100 * (0.8 * 0.5 + 0.2 * 1.0)) = round(100 * (0.4 + 0.2)) = 60
            self.assertEqual(analysis["match_score"], 60)

            # Verified Skills entry in requirement_match was aligned with true score (60)
            skills_req = next(r for r in analysis["requirement_match"] if r["category"] == "Skills")
            self.assertEqual(skills_req["match_percent"], 60)

            # Verified skill gap categories
            matched_names = [m["skill_name"] for m in analysis["skill_gap"]["matched"]]
            self.assertIn("Python (computer programming)", matched_names)
            self.assertIn("FastAPI", matched_names)
            self.assertEqual(analysis["skill_gap"]["missing_essential"], ["Docker"])

    def test_pipeline_insufficient_bypasses_gemini(self):
        pipeline = AnalysisPipeline(
            esco_csv_path="data/skills_en.csv",
            custom_json_path="data/custom_skills.json"
        )

        jd_index.clear_store_for_testing()
        with patch.object(jd_index, "_get_embedder") as mock_idx_emb:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = [0.1] * 3072
            mock_idx_emb.return_value = mock_emb
            jd_rec = jd_index.add_jd("Role", "Co", "Remote", " ".join(["requirement"] * 40))

        short_resume = "Too short resume."
        with patch.object(pipeline.client.models, "generate_content") as mock_gen:
            res = pipeline.analyze(short_resume, "res_short", jd_rec.jd_id)
            mock_gen.assert_not_called()
            self.assertFalse(res["context_sufficient"])
            self.assertEqual(res["match_score"], 0)

    def test_pipeline_generation_failure_fallback(self):
        pipeline = AnalysisPipeline(
            esco_csv_path="data/skills_en.csv",
            custom_json_path="data/custom_skills.json"
        )

        jd_index.clear_store_for_testing()
        with patch.object(jd_index, "_get_embedder") as mock_idx_emb:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = [0.1] * 3072
            mock_idx_emb.return_value = mock_emb
            jd_rec = jd_index.add_jd("Role", "Co", "Remote", " ".join(["requirement"] * 40))

        normal_resume = " ".join(["Python developer with experience."] * 15)

        # Mock a completely broken/garbled LLM output
        mock_resp = MagicMock()
        mock_resp.text = "NOT JSON GARBLED DATA"

        with patch.object(pipeline.client.models, "generate_content", return_value=mock_resp):
            res = pipeline.analyze(normal_resume, "res_norm", jd_rec.jd_id)
            self.assertFalse(res["context_sufficient"])
            self.assertTrue(
                "failed" in res["insufficient_reason"].lower() or "failure" in res["insufficient_reason"].lower()
            )
            self.assertEqual(res["match_score"], 0)
            # Response must still validate against DetailAnalysis
            validated = DetailAnalysis.model_validate(res)
            self.assertIsNotNone(validated)

    def test_pipeline_unexpected_exception_in_early_steps_falls_back(self):
        """Verify that any exception in early steps (ChromaDB or sufficiency) never raises unhandled."""
        pipeline = AnalysisPipeline(
            esco_csv_path="data/skills_en.csv",
            custom_json_path="data/custom_skills.json"
        )

        # Simulate ChromaDB raising a catastrophic ValueError during get_jd_record
        with patch.object(jd_index, "get_jd_record", side_effect=ValueError("ChromaDB connection corrupt")):
            res = pipeline.analyze("Normal length resume text " * 10, "res_err", "some_jd_id")
            self.assertFalse(res["context_sufficient"])
            self.assertEqual(res["match_score"], 0)
            self.assertIn("ChromaDB connection corrupt", res["insufficient_reason"])
            validated = DetailAnalysis.model_validate(res)
            self.assertIsNotNone(validated)

    def test_pipeline_end_to_end_reproducibility(self):
        pipeline = AnalysisPipeline(
            esco_csv_path="data/skills_en.csv",
            custom_json_path="data/custom_skills.json"
        )

        jd_index.clear_store_for_testing()
        with patch.object(jd_index, "_get_embedder") as mock_idx_emb:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = [0.1] * 3072
            mock_idx_emb.return_value = mock_emb
            jd_rec = jd_index.add_jd("Role", "Co", "Remote", " ".join(["requirement"] * 40))

        normal_resume = " ".join(["Python developer with FastAPI experience."] * 15)

        gemini_mock_output = {
            "resume_skills": ["Python", "FastAPI"],
            "jd_essential_skills": ["Python"],
            "jd_preferred_skills": ["Docker"],
            "summary": "First run narrative prose...",
            "strengths": ["Strength 1"],
            "weaknesses": ["Weakness 1"],
            "requirement_match": [{"category": "Experience", "match_percent": 80}],
            "recommendations": []
        }

        mock_resp = MagicMock()
        mock_resp.text = json.dumps(gemini_mock_output)

        with patch.object(pipeline.client.models, "generate_content", return_value=mock_resp):
            run1 = pipeline.analyze(normal_resume, "res_rep", jd_rec.jd_id)

        # Run 2: Even if Gemini narrative wording slightly changed
        gemini_mock_output2 = dict(gemini_mock_output)
        gemini_mock_output2["summary"] = "Second run narrative with slightly different phrasing."
        mock_resp.text = json.dumps(gemini_mock_output2)

        with patch.object(pipeline.client.models, "generate_content", return_value=mock_resp):
            run2 = pipeline.analyze(normal_resume, "res_rep", jd_rec.jd_id)

        # Deterministic fields must be strictly identical!
        self.assertEqual(run1["match_score"], run2["match_score"])
        self.assertEqual(run1["skill_gap"], run2["skill_gap"])
        skills_p1 = next(r["match_percent"] for r in run1["requirement_match"] if r["category"] == "Skills")
        skills_p2 = next(r["match_percent"] for r in run2["requirement_match"] if r["category"] == "Skills")
        self.assertEqual(skills_p1, skills_p2)

    def test_fixtures_skill_coverage(self):
        """Tests that key skills in all 5 fixture resumes and 5 JDs ground successfully."""
        fixture_skills = [
            # Resume skills
            "Python", "FastAPI", "Django", "PostgreSQL", "Redis", "Docker", "AWS", "Pytest", "GitHub Actions",
            "JavaScript", "TypeScript", "React", "Next.js", "Redux", "Tailwind CSS", "HTML5", "CSS3", "Jest", "Cypress", "Jira", "Git",
            "NumPy", "Pandas", "Scikit-learn", "PyTorch", "SQL", "ChromaDB", "LangChain", "Matplotlib", "Streamlit",
            "Kubernetes", "Terraform", "Ansible", "Jenkins", "GitLab CI/CD", "Linux", "Bash", "Azure",
            "Node.js", "Express.js", "MongoDB", "GCP", "Nginx",
            # JD specific skills
            "Microservices", "REST API", "Agile", "Scrum", "CI/CD"
        ]
        unresolved = []
        for s in fixture_skills:
            grounded = ground_candidates([s], self.skill_index)
            if not grounded:
                unresolved.append(s)

        self.assertEqual(
            unresolved, [],
            f"The following fixture skills failed to ground: {unresolved}"
        )

    def test_pipeline_latency(self):
        """Measures and asserts deterministic pipeline latency stays well within budget."""
        pipeline = AnalysisPipeline(
            esco_csv_path="data/skills_en.csv",
            custom_json_path="data/custom_skills.json"
        )
        jd_index.clear_store_for_testing()
        with patch.object(jd_index, "_get_embedder") as mock_idx_emb:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = [0.1] * 3072
            mock_idx_emb.return_value = mock_emb
            jd_rec = jd_index.add_jd("Role", "Co", "Remote", " ".join(["requirement"] * 40))

        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "resume_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
            "jd_essential_skills": ["Python", "PostgreSQL"],
            "jd_preferred_skills": ["Docker", "Kubernetes"],
            "summary": "Fit summary",
            "strengths": ["Python depth"],
            "weaknesses": ["Kubernetes"],
            "requirement_match": [{"category": "Experience", "match_percent": 80}],
            "recommendations": []
        })

        t0 = time.perf_counter()
        with patch.object(pipeline.client.models, "generate_content", return_value=mock_resp):
            res = pipeline.analyze(" ".join(["Python developer with FastAPI."] * 10), "res_lat", jd_rec.jd_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"\n[Latency Benchmark] Pipeline processing elapsed time: {elapsed_ms:.2f}ms")
        self.assertLess(elapsed_ms, 1000.0, f"Local pipeline processing took too long: {elapsed_ms:.2f}ms")
        self.assertEqual(res["match_score"], 90) # 100% essential (0.8) + 50% preferred (0.1) = 90%


if __name__ == "__main__":
    unittest.main()
