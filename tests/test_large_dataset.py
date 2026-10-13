"""Large library ingestion and the strict 25-job decision boundary."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from uuid import uuid4

import chromadb
from fastapi.testclient import TestClient
import server
import jd_index
import ranking
import decision_ranking as decision
from schemas import JDRecord
from scripts.generate_india_demo import build_dataset
from location_preferences import resolve_city


def snapshot(size):
    return [(JDRecord(jd_id=f"job_{i:03}", title=f"Role {i}", company="Demo", location="Mumbai",
                      snippet="Python", full_text=f"Python backend role {i}"), [1, i / 100]) for i in range(size)]


class BoundedDecisionTests(unittest.TestCase):
    def setUp(self):
        decision.clear_cache()
        ranking._embed_resume.cache_clear()

    def tearDown(self):
        decision.clear_cache()
        ranking._embed_resume.cache_clear()

    def test_all_jobs_and_top_five_use_same_bounded_candidate_pool(self):
        embedder = MagicMock()
        embedder.embed_chunk.return_value = [1, 0]
        candidates = []
        def assess(resume, jobs):
            candidates.append([j.jd_id for j in jobs])
            return decision.DecisionOutcome({j.jd_id: decision.Assessment(i * 4, "Strong", .9)
                                             for i, j in enumerate(jobs)}, "applied", "Applied", "test")
        with patch.object(ranking.jd_index, "get_ranking_snapshot", return_value=snapshot(105)), \
                patch.object(ranking, "_get_embedder", return_value=embedder), \
                patch.object(ranking, "assess_shortlist", side_effect=assess):
            all_jobs = ranking.rank_jds("Resume", "r", top_n=0)
            top_five = ranking.rank_jds("Resume", "r", top_n=5)
            null_jobs = ranking.rank_jds("Resume", "r", top_n=None)
        self.assertTrue(all(len(pool) == 25 for pool in candidates))
        self.assertTrue(all(pool == candidates[0] for pool in candidates))
        self.assertEqual(len(all_jobs.results), 105)
        self.assertEqual(len(null_jobs.results), 105)
        self.assertEqual(all_jobs.decision_shortlist_size, 25)
        self.assertEqual(all_jobs.total_jobs, 105)
        self.assertEqual([c.jd_id for c in top_five.results], [c.jd_id for c in all_jobs.results[:5]])
        self.assertEqual(all_jobs.results[0].jd_id, "job_024")
        self.assertTrue(all(c.score_source == "mercury" for c in all_jobs.results[:25]))
        self.assertTrue(all(c.score_source == "similarity" and c.fit_status == "not_reviewed"
                            and c.model_fit_score is None for c in all_jobs.results[25:]))
        self.assertEqual([c.jd_id for c in all_jobs.results[25:]], [j.jd_id for j, _ in snapshot(105)[25:]])

    def test_direct_provider_call_never_sends_more_than_25_jobs_and_batches_them(self):
        calls = []
        def provider(*args, **kwargs):
            body = kwargs["json"]
            calls.append(body)
            return SimpleNamespace(status_code=200, json=lambda: {"answers": {
                name: {"type": "score", "score": 2, "probabilities": {"0": 0, "1": 0, "2": 1}, "confidence": 1}
                for name in body["questions"]}})
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-test-key", "RANK_DECISION_ENABLED": "true"}), \
                patch("decision_ranking.requests.post", side_effect=provider):
            jobs = [j for j, _ in snapshot(105)]
            result = decision.assess_shortlist("Resume", jobs)
            again = decision.assess_shortlist("Resume", jobs)
        self.assertEqual(result.status, "applied")
        self.assertEqual(len(result.assessments), 25)
        self.assertEqual([len(body["questions"]) for body in calls], [10, 10, 5])
        self.assertTrue(all(len(json.dumps(body).encode()) < 60000 for body in calls))
        sent = {key for body in calls for key in body["state"]["jobs"]}
        self.assertEqual(sent, {j.jd_id for j in jobs[:25]})
        self.assertEqual(again.assessments, result.assessments)

    def test_malformed_later_batch_preserves_baseline_without_partial_cache(self):
        valid = {"type": "score", "score": 2, "probabilities": {"0": 0, "1": 0, "2": 1}, "confidence": 1}
        def provider(*args, **kwargs):
            body = kwargs["json"]
            data = {"answers": {name: valid for name in body["questions"]}}
            if "job_010" in body["state"]["jobs"]:
                data = {"answers": {}}
            return SimpleNamespace(status_code=200, json=lambda: data)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "fake-test-key", "RANK_DECISION_ENABLED": "true"}), \
                patch("decision_ranking.requests.post", side_effect=provider):
            result = decision.assess_shortlist("Resume", [j for j, _ in snapshot(25)])
        self.assertEqual(result.status, "unavailable")
        self.assertEqual(len(decision._cache), 0)


class IndiaDatasetTests(unittest.TestCase):
    def setUp(self):
        self.db = chromadb.EphemeralClient()
        self.collection = self.db.create_collection("india_" + uuid4().hex)
        self.embedder = MagicMock()
        self.embedder.embed_chunk.return_value = [1., 0., 0.]
        self.embedder.embed_chunks_batch.side_effect = lambda texts: [[1., 0., 0.] for text in texts]
        for item in [patch.object(jd_index, "_get_collection", return_value=self.collection),
                     patch.object(jd_index, "_get_embedder", return_value=self.embedder),
                     patch.object(jd_index, "_RECORD_CACHE", {}), patch.object(jd_index, "_EMBEDDING_CACHE", {}),
                     patch.object(jd_index, "_CACHE_INITIALIZED", False)]:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.db.delete_collection, self.collection.name)
        self.client = TestClient(server.app)

    def test_dataset_style_diversity_and_real_city_resolution(self):
        dataset = build_dataset()
        disk = json.loads((Path(server.PROJECT_ROOT) / "datasets/india_demo_100.json").read_text())
        self.assertEqual(dataset, disk)
        self.assertEqual(len(dataset["jds"]), 100)
        self.assertEqual(len(dataset["resumes"]), 100)
        self.assertEqual(len({j["text"] for j in dataset["jds"]}), 100)
        self.assertEqual(len({r["text"] for r in dataset["resumes"]}), 100)
        self.assertEqual(len({r["id"] for r in dataset["resumes"]}), 100)
        self.assertEqual(len({p["role_family"] for p in dataset["demo_pairs"]}), 20)
        for job in dataset["jds"]:
            self.assertIn("Essential Requirements:", job["text"])
            self.assertIn("Preferred:", job["text"])
            self.assertGreater(len(job["text"].split()), 30)
            self.assertEqual(resolve_city(job["location"]).country, "IN")
        for resume in dataset["resumes"]:
            self.assertGreater(len(resume["text"].split()), 30)
            self.assertIn("Education:", resume["text"])
            self.assertIn("India", resume["text"])

    def test_hundred_jobs_persist_reloading_is_idempotent_and_user_job_is_kept(self):
        personal = jd_index.add_jd("My job", "My company", "Mumbai", "Personal job text")
        response = self.client.post("/api/load-fixtures/india")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["jds"]), 100)
        self.assertEqual(len(data["resumes"]), 100)
        self.assertEqual(jd_index.count(), 101)
        self.assertIn(personal.jd_id, jd_index.get_jd_ids())
        self.assertEqual(self.embedder.embed_chunks_batch.call_count, 5)
        repeat = self.client.post("/api/load-fixtures/india").json()
        self.assertEqual(repeat["jds"], data["jds"])
        self.assertEqual(self.embedder.embed_chunks_batch.call_count, 5)
        self.assertIsNone(self.client.get("/jds/count").json()["cap"])

    def test_demo_profiles_can_be_recovered_read_only_by_another_browser(self):
        self.assertEqual(self.client.get("/api/loaded-demo-resumes").json(), {"datasets": []})
        self.client.post("/api/load-fixtures/india").raise_for_status()
        before = self.embedder.embed_chunks_batch.call_count
        data = self.client.get("/api/loaded-demo-resumes").json()
        self.assertEqual([d["id"] for d in data["datasets"]], ["india"])
        self.assertEqual(len(data["datasets"][0]["resumes"]), 100)
        self.assertEqual(self.embedder.embed_chunks_batch.call_count, before)
        self.assertEqual(jd_index.count(), 100)
        jd_index.delete_jd(jd_index.get_jd_ids()[0])
        self.assertEqual(self.client.get("/api/loaded-demo-resumes").json(), {"datasets": []})

    def test_partial_bulk_failure_is_explicit_and_retry_reuses_saved_batch(self):
        self.embedder.embed_chunks_batch.side_effect = [[[1., 0., 0.]] * 20, RuntimeError("Unavailable")]
        response = self.client.post("/api/load-fixtures/india")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(jd_index.count(), 20)
        original_ids = set(jd_index.get_jd_ids())
        self.embedder.embed_chunks_batch.side_effect = lambda texts: [[1., 0., 0.] for _ in texts]
        retry = self.client.post("/api/load-fixtures/india")
        self.assertEqual(retry.status_code, 200)
        self.assertEqual(jd_index.count(), 100)
        self.assertTrue(original_ids.issubset(jd_index.get_jd_ids()))

    def test_dimension_mismatch_cannot_erase_existing_jobs(self):
        first = jd_index.add_jd("Original", "Co", "Mumbai", "Original job")
        self.embedder.embed_chunk.return_value = [1., 0.]
        with self.assertRaises(RuntimeError):
            jd_index.add_jd("Incompatible", "Co", "Pune", "Different job")
        self.assertEqual(jd_index.get_jd_ids(), [first.jd_id])


if __name__ == "__main__":
    unittest.main()
