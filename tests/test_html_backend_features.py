"""Contracts needed by the HTML controls; providers and settings are isolated."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import chromadb
from fastapi.testclient import TestClient
import server
import embedding_manager
import jd_index
import ranking
import decision_ranking
from analysis_pipeline import AnalysisPipeline


class HTMLBackendFeaturesTests(unittest.TestCase):
    def test_settings_refresh_clients_without_discarding_job_vectors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = root / ".env"
            env.write_text("GEMINI_API_KEY=old-test-key\nGROQ_API_KEY=old-groq\nOPENROUTER_API_KEY=old-router-key\nAPI_BASE_URL=http://127.0.0.1:8010\n")
            with patch.object(server, "PROJECT_ROOT", root), \
                    patch.dict(os.environ, {"GEMINI_API_KEY": "old-test-key", "GROQ_API_KEY": "old-groq", "OPENROUTER_API_KEY": "old-router-key"}), \
                    patch.object(server, "analyzer", object()), \
                    patch.object(server, "_chat", object()), \
                    patch.object(server, "_rag", object()), \
                    patch.object(embedding_manager, "GEMINI_API_KEY", "old-test-key"), \
                    patch.object(embedding_manager, "_DEFAULT_EMBEDDER", object()), \
                    patch.object(jd_index, "_embedder", object()), \
                    patch.object(ranking, "_embedder", object()), \
                    patch.object(ranking, "_embed_resume") as resume_cache, \
                    patch.object(decision_ranking, "clear_cache") as decision_cache, \
                    patch.object(jd_index, "count", return_value=2):
                result = TestClient(server.app).post("/api/config", json={
                    "gemini_api_key": "new-test-key", "generation_model": "selected-model", "rerank": True,
                })
                self.assertEqual(result.status_code, 200)
                self.assertIsNone(server.analyzer)
                self.assertIsNone(jd_index._embedder)
                self.assertIsNone(ranking._embedder)
                self.assertIsNone(embedding_manager._DEFAULT_EMBEDDER)
                self.assertEqual(embedding_manager.GEMINI_API_KEY, "new-test-key")
                resume_cache.cache_clear.assert_called_once()
                decision_cache.assert_called_once()
                self.assertIn("GROQ_API_KEY=old-groq", env.read_text())
                self.assertIn("OPENROUTER_API_KEY=old-router-key", env.read_text())
                config = TestClient(server.app).get("/api/config").json()
                self.assertEqual(config["generation_model"], "selected-model")
                self.assertTrue(config["rerank"])
                self.assertNotIn("new-test-key", json.dumps(config))
                self.assertTrue(config["openrouter_configured"])
                self.assertNotIn("old-router-key", json.dumps(config))
                client = TestClient(server.app)
                self.assertEqual(client.post("/api/config", json={"openrouter_api_key": "new-router-key"}).status_code, 200)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "new-router-key")
                self.assertIn("OPENROUTER_API_KEY=new-router-key", env.read_text())
                self.assertIn("GEMINI_API_KEY=new-test-key", env.read_text())
                self.assertNotIn("new-router-key", client.get("/api/config").text)
                self.assertEqual(client.post("/api/config", json={"openrouter_api_key": ""}).status_code, 200)
                self.assertFalse(client.get("/api/config").json()["openrouter_configured"])
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "")

    def test_selected_model_is_used_for_detailed_analysis(self):
        pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
        provider = MagicMock()
        provider.models.generate_content.return_value.text = json.dumps({
            "resume_skills": [], "jd_essential_skills": [], "jd_preferred_skills": [],
        })
        pipeline.clients = [provider]
        pipeline.client = provider
        pipeline.skill_index = object()
        with patch.dict(os.environ, {"RAG_GENERATION_MODEL": "selected-model"}), \
                patch("analysis_pipeline.jd_index.get_jd_record", return_value=SimpleNamespace(full_text="Job text")), \
                patch("analysis_pipeline.check_sufficiency", return_value=(True, None)), \
                patch("analysis_pipeline.ground_candidates", return_value=[]):
            pipeline.analyze("Resume text", "resume", "job")
        self.assertEqual(provider.models.generate_content.call_args.kwargs["model"], "selected-model")

    def test_removed_key_does_not_fall_back_to_cached_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}), \
                patch.object(embedding_manager, "GEMINI_API_KEY", "stale-key"), \
                patch.object(embedding_manager.genai, "Client") as client:
            embedding_manager.EmbeddingManager()
        self.assertEqual(client.call_args.kwargs["api_key"], "")

    def test_library_empty_question_and_provider_error_are_actionable(self):
        client = TestClient(server.app)
        self.assertEqual(client.post("/query", params={"query_text": "  "}).status_code, 422)
        with patch.object(server, "get_rag", side_effect=RuntimeError("Internal provider detail")):
            response = client.post("/query", params={"query_text": "A question"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("provider settings", response.json()["detail"])
        self.assertNotIn("Internal provider detail", response.text)

    def test_upload_size_limit_is_enforced_by_api(self):
        response = TestClient(server.app).post("/api/upload-document", files={
            "file": ("oversized.txt", b"x" * (5 * 1024 * 1024 + 1), "text/plain"),
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("5 MB", response.json()["detail"])


class SampleJobLoaderTests(unittest.TestCase):
    def setUp(self):
        self.db = chromadb.EphemeralClient()
        self.collection = self.db.create_collection("sample_loader_" + uuid4().hex)
        self.embedder = MagicMock()
        self.embedder.embed_chunk.return_value = [1.0, 0.0, 0.0]
        self.patches = [
            patch.object(jd_index, "_get_collection", return_value=self.collection),
            patch.object(jd_index, "_get_embedder", return_value=self.embedder),
            patch.object(jd_index, "_RECORD_CACHE", {}),
            patch.object(jd_index, "_EMBEDDING_CACHE", {}),
            patch.object(jd_index, "_CACHE_INITIALIZED", False),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.db.delete_collection, self.collection.name)
        self.client = TestClient(server.app)

    def test_all_five_samples_persist_and_repeat_loading_keeps_existing_jobs(self):
        existing = jd_index.add_jd("My job", "My company", "Remote", "My own job description")
        first = self.client.post("/api/load-fixtures/sample")
        self.assertEqual(first.status_code, 200)
        data = first.json()
        self.assertEqual(data["count"], 5)
        self.assertEqual(data["stored_count"], 6)
        self.assertEqual(len({j["backend_id"] for j in data["jds"]}), 5)
        stored_ids = self.client.get("/jds/ids").json()["ids"]
        self.assertIn(existing.jd_id, stored_ids)
        self.assertTrue(all(j["backend_id"] in stored_ids for j in data["jds"]))
        second = self.client.post("/api/load-fixtures/sample")
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["jds"], data["jds"])
        self.assertEqual(self.collection.count(), 6)
        self.assertEqual(self.embedder.embed_chunk.call_count, 6)

    def test_samples_load_beyond_old_store_cap(self):
        for i in range(11):
            jd_index.add_jd(f"My role {i}", "My company", "Remote", f"My description {i}")
        result = self.client.post("/api/load-fixtures/sample")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.collection.count(), 16)
        self.assertEqual(self.embedder.embed_chunk.call_count, 16)

    def test_embedding_failure_does_not_claim_samples_are_saved(self):
        self.embedder.embed_chunk.side_effect = RuntimeError("Provider unavailable")
        result = self.client.post("/api/load-fixtures/sample")
        self.assertEqual(result.status_code, 503)
        self.assertNotIn("jds", result.json())
        self.assertEqual(self.collection.count(), 0)


if __name__ == "__main__":
    unittest.main()
