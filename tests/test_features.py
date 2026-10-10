import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from server import app
import jd_index
import ranking
from analysis_pipeline import AnalysisPipeline


class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_feature_1_top_n_selector_ranking(self):
        """Test top_n parameter in ranking.py, AnalysisPipeline, and /rank endpoint."""
        jd_index.clear_store_for_testing()

        fake_vec = [0.1] * 3072
        with patch.object(jd_index, "_get_embedder") as mock_emb_fn:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = fake_vec
            mock_emb_fn.return_value = mock_emb

            # Ingest 7 JDs
            created_ids = []
            for i in range(7):
                rec = jd_index.add_jd(f"Role {i}", f"Company {i}", f"Location {i}", f"Job text {i} " * 10)
                created_ids.append(rec.jd_id)

        with patch.object(ranking, "_get_embedder") as mock_rank_emb_fn:
            mock_rank_emb = MagicMock()
            mock_rank_emb.embed_chunk.return_value = fake_vec
            mock_rank_emb_fn.return_value = mock_rank_emb

            # Test default top_n=5
            res_default = ranking.rank_jds("Resume text " * 10, "res_test")
            self.assertEqual(len(res_default.results), 5)
            self.assertEqual([c.rank for c in res_default.results], [1, 2, 3, 4, 5])

            # Test top_n=2
            res_2 = ranking.rank_jds("Resume text " * 10, "res_test", top_n=2)
            self.assertEqual(len(res_2.results), 2)
            self.assertEqual([c.rank for c in res_2.results], [1, 2])

            # Test top_n=None (All)
            res_all = ranking.rank_jds("Resume text " * 10, "res_test", top_n=None)
            self.assertEqual(len(res_all.results), 7)
            self.assertEqual([c.rank for c in res_all.results], list(range(1, 8)))

            # Test top_n=0 (All)
            res_zero = ranking.rank_jds("Resume text " * 10, "res_test", top_n=0)
            self.assertEqual(len(res_zero.results), 7)

            # Test top_n=10 with only 7 JDs (no crash, returns 7)
            res_10 = ranking.rank_jds("Resume text " * 10, "res_test", top_n=10)
            self.assertEqual(len(res_10.results), 7)

            # Test API /rank with top_n=3
            api_resp_3 = self.client.post("/rank", json={
                "resume_text": "Resume text " * 10,
                "resume_id": "res_test",
                "top_n": 3
            })
            self.assertEqual(api_resp_3.status_code, 200)
            self.assertEqual(len(api_resp_3.json()["results"]), 3)

            # Test API /rank with top_n=None
            api_resp_all = self.client.post("/rank", json={
                "resume_text": "Resume text " * 10,
                "resume_id": "res_test",
                "top_n": None
            })
            self.assertEqual(api_resp_all.status_code, 200)
            self.assertEqual(len(api_resp_all.json()["results"]), 7)

    def test_feature_6_preferences_boost(self):
        """Test additive preference boost and clamping in ranking."""
        jd_index.clear_store_for_testing()

        # Let vec_a and vec_b have unit norm:
        # e.g., resume vector [1, 0, ...]
        # JD 1 vector: [0.72, sqrt(1 - 0.72^2), ...] -> cosine similarity = 0.72
        # JD 2 vector: [0.99, sqrt(1 - 0.99^2), ...] -> cosine similarity = 0.99 (test clamp)
        import math
        vec_resume = [1.0] + [0.0] * 3071
        vec_jd1 = [0.72, math.sqrt(1 - 0.72**2)] + [0.0] * 3070
        vec_jd2 = [0.99, math.sqrt(1 - 0.99**2)] + [0.0] * 3070

        with patch.object(jd_index, "_get_embedder") as mock_emb_fn:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.side_effect = [vec_jd1, vec_jd2]
            mock_emb_fn.return_value = mock_emb

            rec1 = jd_index.add_jd(
                "Software Engineer", "Acme", "Remote, New York", "Some text " * 10
            )
            rec2 = jd_index.add_jd(
                "Senior Engineer", "Beta", "London, On-site", "Some text " * 10
            )

        with patch.object(ranking, "_get_embedder") as mock_rank_emb_fn:
            mock_rank_emb = MagicMock()
            mock_rank_emb.embed_chunk.return_value = vec_resume
            mock_rank_emb_fn.return_value = mock_rank_emb

            # 1. No preferences: raw similarity
            # rec1: 0.72 -> 72
            # rec2: 0.99 -> 99
            res_no_pref = ranking.rank_jds("Resume text", "res_001")
            scores_no_pref = {c.jd_id: c.similarity_score for c in res_no_pref.results}
            self.assertEqual(scores_no_pref[rec1.jd_id], 72)
            self.assertEqual(scores_no_pref[rec2.jd_id], 99)

            # 2. Location preference: "New York" -> +0.05
            # rec1 gets 0.72 + 0.05 = 0.77 -> 77
            res_loc = ranking.rank_jds(
                "Resume text", "res_001", preferred_location="New York"
            )
            scores_loc = {c.jd_id: c.similarity_score for c in res_loc.results}
            self.assertEqual(scores_loc[rec1.jd_id], 77)
            self.assertEqual(scores_loc[rec2.jd_id], 99)

            # 3. Work arrangement preference: "Remote" -> +0.03
            # rec1 gets 0.72 + 0.03 = 0.75 -> 75
            res_arr = ranking.rank_jds(
                "Resume text", "res_001", preferred_arrangement="Remote"
            )
            scores_arr = {c.jd_id: c.similarity_score for c in res_arr.results}
            self.assertEqual(scores_arr[rec1.jd_id], 75)
            self.assertEqual(scores_arr[rec2.jd_id], 99)

            # 4. Both preferences: "New York" (+0.05) and "Remote" (+0.03) -> +0.08
            # rec1 gets 0.72 + 0.05 + 0.03 = 0.80 -> 80
            res_both = ranking.rank_jds(
                "Resume text",
                "res_001",
                preferred_location="New York",
                preferred_arrangement="Remote"
            )
            scores_both = {c.jd_id: c.similarity_score for c in res_both.results}
            self.assertEqual(scores_both[rec1.jd_id], 80)

            # 5. Clamp to 100: rec2 (0.99) with "London" (+0.05) and "On-site" (+0.03) -> 0.99 + 0.08 = 1.07 -> clamped to 100
            res_clamped = ranking.rank_jds(
                "Resume text",
                "res_001",
                preferred_location="London",
                preferred_arrangement="On-site"
            )
            scores_clamped = {c.jd_id: c.similarity_score for c in res_clamped.results}
            self.assertEqual(scores_clamped[rec2.jd_id], 100)

            # 6. API call with preferences
            api_resp = self.client.post("/rank", json={
                "resume_text": "Resume text",
                "resume_id": "res_001",
                "preferred_location": "New York",
                "preferred_arrangement": "Remote"
            })
            self.assertEqual(api_resp.status_code, 200)
            api_scores = {c["jd_id"]: c["similarity_score"] for c in api_resp.json()["results"]}
            self.assertEqual(api_scores[rec1.jd_id], 80)

    def test_feature_3_download_report(self):
        """Test markdown gap analysis report generation."""
        from report_generator import generate_markdown_report

        mock_analysis = {
            "resume_id": "res_100",
            "jd_id": "jd_200",
            "context_sufficient": True,
            "match_score": 88,
            "summary": "Candidate shows exceptional backend alignment.",
            "strengths": ["Deep Python expertise", "FastAPI architecture experience"],
            "weaknesses": ["Lacks production Kubernetes operations"],
            "skill_gap": {
                "matched": [{"skill_name": "Python", "match_type": "exact"}, {"skill_name": "FastAPI", "match_type": "exact"}],
                "missing_essential": ["Docker"],
                "missing_preferred": ["Kubernetes"],
                "extra": ["Flask"]
            },
            "requirement_match": [
                {"category": "Skills", "match_percent": 88},
                {"category": "Experience", "match_percent": 85}
            ],
            "recommendations": [
                {"skill": "Docker", "reason": "Required for microservice deployments"}
            ]
        }

        md = generate_markdown_report(mock_analysis, "Jane Doe", "Senior Python Engineer")

        # Verify key sections and formatting
        self.assertIn("# Gap Analysis Report: Jane Doe vs Senior Python Engineer", md)
        self.assertIn("## Match Score: 88%", md)
        self.assertIn("Candidate shows exceptional backend alignment.", md)
        self.assertIn("- Deep Python expertise", md)
        self.assertIn("- Lacks production Kubernetes operations", md)
        self.assertIn("| **Matched Skills** | Python, FastAPI |", md)
        self.assertIn("| **Missing Essential** | Docker |", md)
        self.assertIn("| **Missing Preferred** | Kubernetes |", md)
        self.assertIn("| **Candidate Extra** | Flask |", md)
        self.assertIn("| Skills | 88% |", md)
        self.assertIn("| Docker | Required for microservice deployments |", md)


if __name__ == "__main__":
    unittest.main()
