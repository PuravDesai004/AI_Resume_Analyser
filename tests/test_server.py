import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import app
import jd_index


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_jds_count(self):
        response = self.client.get("/jds/count")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("count", data)
        self.assertEqual(data["cap"], 15)

    def test_post_jds_and_rank(self):
        jd_index.clear_store_for_testing()

        # Add a JD via API with mocked embedding
        fake_vec = [0.1] * 3072
        with patch.object(jd_index, "_get_embedder") as mock_emb_fn:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = fake_vec
            mock_emb_fn.return_value = mock_emb

            resp = self.client.post("/jds", json={
                "title": "Cloud Architect",
                "company": "SkyNet",
                "location": "Remote",
                "jd_text": "Looking for AWS, Docker, Kubernetes experience."
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("jd_id", data)
            self.assertEqual(data["title"], "Cloud Architect")
            jd_id = data["jd_id"]

        # Rank via API with mocked resume embedding
        import ranking
        with patch.object(ranking, "_get_embedder") as mock_rank_emb_fn:
            mock_rank_emb = MagicMock()
            mock_rank_emb.embed_chunk.return_value = fake_vec
            mock_rank_emb_fn.return_value = mock_rank_emb

            rank_resp = self.client.post("/rank", json={
                "resume_text": "Cloud engineer with Docker and AWS experience.",
                "resume_id": "res_cloud_01"
            })
            self.assertEqual(rank_resp.status_code, 200)
            rank_data = rank_resp.json()
            self.assertEqual(rank_data["resume_id"], "res_cloud_01")
            self.assertEqual(len(rank_data["results"]), 1)
            self.assertEqual(rank_data["results"][0]["jd_id"], jd_id)

    def test_post_analyze_insufficient_bypasses(self):
        jd_index.clear_store_for_testing()
        with patch.object(jd_index, "_get_embedder") as mock_emb_fn:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = [0.1] * 3072
            mock_emb_fn.return_value = mock_emb

            rec = jd_index.add_jd("Title", "Co", "Loc", "Long enough job description text " * 10)

        # Send insufficient resume
        analyze_resp = self.client.post("/analyze", json={
            "resume_text": "Too short.",
            "resume_id": "res_short",
            "jd_id": rec.jd_id
        })
        self.assertEqual(analyze_resp.status_code, 200)
        data = analyze_resp.json()
        self.assertFalse(data["context_sufficient"])
        self.assertEqual(data["match_score"], 0)
        self.assertIn("Resume text is too short", data["insufficient_reason"])

    def test_post_jds_cap_exceeded_returns_409(self):
        jd_index.clear_store_for_testing()
        with patch.object(jd_index, "_get_embedder") as mock_emb_fn:
            mock_emb = MagicMock()
            mock_emb.embed_chunk.return_value = [0.1] * 3072
            mock_emb_fn.return_value = mock_emb

            for i in range(15):
                jd_index.add_jd(f"Role {i}", "Co", "Loc", "Text " * 10)

            # 16th JD via API
            resp = self.client.post("/jds", json={
                "title": "Role 16",
                "company": "Co",
                "location": "Loc",
                "jd_text": "Text " * 10
            })
            self.assertEqual(resp.status_code, 409)
            self.assertEqual(resp.json()["detail"]["error_code"], "STORE_CAP_REACHED")

    def test_post_analyze_jd_not_found_returns_404(self):
        resp = self.client.post("/analyze", json={
            "resume_text": "Normal length resume text " * 10,
            "resume_id": "res_001",
            "jd_id": "nonexistent_jd_id"
        })
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["detail"]["error_code"], "JD_NOT_FOUND")


if __name__ == "__main__":
    unittest.main()
