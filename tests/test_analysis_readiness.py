import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from schemas import RankingResult
import server


class AnalysisReadinessTests(unittest.TestCase):
    def test_missing_taxonomy_returns_actionable_error(self):
        with patch.object(server, "analyzer", None), patch(
            "analysis_pipeline.AnalysisPipeline", side_effect=FileNotFoundError("Missing data/skills_en.csv"),
        ):
            result = TestClient(server.app).post("/analyze", json={
                "resume_text": "Python engineer", "resume_id": "a", "jd_id": "b",
            })
        self.assertEqual(result.status_code, 503)
        self.assertIn("skills_en.csv", result.json()["detail"])

    def test_ranking_does_not_initialize_taxonomy(self):
        with patch("ranking.rank_jds", return_value=RankingResult(resume_id="a")) as rank, \
                patch.object(server, "get_analyzer", side_effect=AssertionError("Must not load taxonomy")):
            result = TestClient(server.app).post("/rank", json={
                "resume_text": "Python engineer", "resume_id": "a", "top_n": 2,
                "preferred_location": " Remote ",
            })
        self.assertEqual(result.status_code, 200)
        self.assertEqual(rank.call_args.kwargs["preferred_location"], "Remote")
        self.assertEqual(rank.call_args.kwargs["top_n"], 2)


if __name__ == "__main__":
    unittest.main()
