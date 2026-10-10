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


if __name__ == "__main__":
    unittest.main()
