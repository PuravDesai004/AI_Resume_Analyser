"""Mercury's typed API, reranking, cache isolation and complete fallback contract."""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests
import decision_ranking as decision
import ranking
from schemas import JDRecord


def job(jd_id, text="Python or Node.js required.", location="Remote"):
    return JDRecord(jd_id=jd_id, title=jd_id, company="Example", location=location,
                    snippet=text, full_text=text)


def answer(probabilities=(0.1, 0.2, 0.7), confidence=0.8):
    return {"type": "score", "score": sum(i * p for i, p in enumerate(probabilities)),
            "probabilities": {str(i): p for i, p in enumerate(probabilities)},
            "confidence": confidence}


def reply(*answers, status=200):
    return SimpleNamespace(status_code=status, json=lambda: {
        "answers": {f"job_{i}": value for i, value in enumerate(answers)}})


class MercuryContractTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "OPENROUTER_API_KEY": "fake-openrouter-key-for-tests",
            "OPENROUTER_DECISION_MODEL": decision.DEFAULT_MODEL,
            "RANK_DECISION_ENABLED": "true", "OPENROUTER_DECISION_MIN_CONFIDENCE": "0.5",
        })
        self.environment.start()
        self.post = patch("decision_ranking.requests.post")
        self.api = self.post.start()
        decision.clear_cache()
        ranking._embed_resume.cache_clear()

    def tearDown(self):
        decision.clear_cache()
        ranking._embed_resume.cache_clear()
        self.post.stop()
        self.environment.stop()

    def test_missing_key_and_disabled_never_contact_provider(self):
        os.environ["OPENROUTER_API_KEY"] = ""
        self.assertEqual(decision.assess_shortlist("Resume", [job("a")]).status, "not_configured")
        os.environ["OPENROUTER_API_KEY"] = "fake-test-key"
        os.environ["RANK_DECISION_ENABLED"] = "false"
        self.assertEqual(decision.assess_shortlist("Resume", [job("a")]).status, "disabled")
        self.api.assert_not_called()

    def test_typed_full_documents_request_and_assessment(self):
        self.api.return_value = reply(answer())
        outcome = decision.assess_shortlist("Entire resume\nFinal experience", [job("a", "JD start\nLast requirement")])
        self.assertEqual(outcome.status, "applied")
        self.assertEqual(outcome.assessments["a"].label, "Strong")
        self.assertEqual(outcome.assessments["a"].score, 80)
        args, kwargs = self.api.call_args
        self.assertEqual(args[0], decision.ENDPOINT)
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer fake-openrouter-key-for-tests")
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["timeout"], (3.0, 20.0))
        body = kwargs["json"]
        self.assertEqual(body["model"], decision.DEFAULT_MODEL)
        self.assertEqual(body["state"]["resume"], "Entire resume\nFinal experience")
        self.assertEqual(body["state"]["jobs"]["a"]["description"], "JD start\nLast requirement")
        question = body["questions"]["job_0"]
        self.assertEqual(question["criteria"], decision.LEVELS)
        self.assertIn("'A or B' is satisfied by either", question["instructions"])
        self.assertIn("not instructions", question["instructions"])

    def test_cache_reuses_pairs_and_invalidates_changed_content_model_and_key(self):
        self.api.return_value = reply(answer(), answer())
        decision.assess_shortlist("Resume", [job("a"), job("b")])
        decision.assess_shortlist("Resume", [job("b"), job("a")])
        self.assertEqual(self.api.call_count, 1)
        self.api.return_value = reply(answer())
        decision.assess_shortlist("Resume", [job("a", "Updated JD"), job("b")])
        self.assertEqual(set(self.api.call_args.kwargs["json"]["state"]["jobs"]), {"a"})
        self.api.return_value = reply(answer(), answer())
        for name, value in [("OPENROUTER_DECISION_MODEL", "changed-model"),
                            ("OPENROUTER_API_KEY", "changed-test-key")]:
            os.environ[name] = value
            decision.assess_shortlist("Resume", [job("a"), job("b")])
        decision.assess_shortlist("Updated resume", [job("a"), job("b")])
        self.assertEqual(self.api.call_count, 5)
        with patch.object(decision, "RUBRIC_VERSION", "changed-rubric"):
            decision.assess_shortlist("Updated resume", [job("a"), job("b")])
        self.assertEqual(self.api.call_count, 6)

    def test_expired_cache_is_refreshed(self):
        self.api.return_value = reply(answer())
        with patch("decision_ranking.time.monotonic", return_value=100):
            decision.assess_shortlist("Resume", [job("a")])
        with patch("decision_ranking.time.monotonic", return_value=2000):
            decision.assess_shortlist("Resume", [job("a")])
        self.assertEqual(self.api.call_count, 2)

    def test_http_failures_and_timeout_fall_back_without_leaking_provider_content(self):
        for status in (401, 402, 403, 429, 500, 302):
            with self.subTest(status=status):
                decision.clear_cache()
                self.api.return_value = reply(status=status)
                result = decision.assess_shortlist("Resume", [job("a")])
                self.assertEqual(result.status, "unavailable")
                self.assertEqual(result.assessments, {})
                self.assertIn("Similarity order retained", result.note)
                self.assertNotIn(os.environ["OPENROUTER_API_KEY"], result.note)
        decision.clear_cache()
        self.api.side_effect = requests.Timeout("sensitive provider error")
        self.assertEqual(decision.assess_shortlist("Resume", [job("a")]).status, "unavailable")
        before = self.api.call_count
        self.assertNotIn("sensitive", decision.assess_shortlist("Resume", [job("a")]).note)
        self.assertEqual(self.api.call_count, before, "Cooldown prevents immediate failed retries")

    def test_malformed_batch_is_rejected_and_not_partially_cached(self):
        bad_answers = []
        for field, value in [("score", True), ("score", float("nan")), ("score", 3),
                             ("score", 0), ("confidence", 1.5), ("type", "choice"),
                             ("probabilities", {"0": 0.1, "1": 0.2}),
                             ("probabilities", {"0": 0.2, "1": 0.2, "2": 0.2})]:
            malformed = copy.deepcopy(answer())
            malformed[field] = value
            bad_answers.append(malformed)
        for bad in bad_answers:
            with self.subTest(answer=bad):
                decision.clear_cache()
                self.api.return_value = reply(answer(), bad)
                result = decision.assess_shortlist("Resume", [job("a"), job("b")])
                self.assertEqual(result.status, "unavailable")
                self.assertEqual(len(decision._cache), 0)
        for data in ({"answers": {}}, {"answers": {"job_0": answer(), "unexpected": answer()}}, []):
            decision.clear_cache()
            self.api.return_value = SimpleNamespace(status_code=200, json=lambda: data)
            self.assertEqual(decision.assess_shortlist("Resume", [job("a")]).status, "unavailable")

    def test_oversize_inputs_fall_back_without_truncation_or_api_call(self):
        result = decision.assess_shortlist("x" * 60_000, [job("a")])
        self.assertEqual(result.status, "input_too_long")
        self.api.assert_not_called()

    def test_uncertain_shortlist_retains_baseline_and_can_reuse_cached_scores(self):
        self.api.return_value = reply(answer(confidence=0.3))
        self.assertEqual(decision.assess_shortlist("Resume", [job("a")]).status, "uncertain")
        os.environ["OPENROUTER_DECISION_MIN_CONFIDENCE"] = "0.2"
        self.assertEqual(decision.assess_shortlist("Resume", [job("a")]).status, "applied")
        self.assertEqual(self.api.call_count, 1)

    def test_reranking_changes_order_preserves_similarity_and_shortlist(self):
        embedder = MagicMock()
        embedder.embed_chunk.return_value = [1, 0]
        snapshot = [(job("a"), [1, 0]), (job("b"), [0.6, 0.8]), (job("c"), [0, 1])]
        self.api.return_value = reply(answer((0.8, 0.2, 0)), answer((0, 0.1, 0.9)), answer((1, 0, 0)))
        with patch.object(ranking, "_get_embedder", return_value=embedder), \
                patch.object(ranking.jd_index, "get_ranking_snapshot", return_value=snapshot):
            result = ranking.rank_jds("Python and React", "resume", top_n=2)
        self.assertEqual([card.jd_id for card in result.results], ["b", "a"])
        self.assertEqual([card.embedding_similarity for card in result.results], [60, 100])
        self.assertEqual([card.baseline_rank for card in result.results], [2, 1])
        self.assertEqual([card.rank for card in result.results], [1, 2])
        self.assertEqual(result.ranking_method, "mercury")
        self.assertEqual(set(self.api.call_args.kwargs["json"]["state"]["jobs"]), {"a", "b", "c"})

    def test_preference_points_apply_once_and_raw_similarity_is_separate(self):
        embedder = MagicMock()
        embedder.embed_chunk.return_value = [1, 0]
        snapshot = [(job("a", location="Pune (On-site)"), [0.8, 0.6]),
                    (job("b", location="Mumbai (Remote)"), [0.6, 0.8])]
        # b's 8 points beat a's 3-point model edge; a led the original similarity.
        self.api.return_value = reply(answer((0, 0.14, 0.86)), answer((0, 0.2, 0.8)))
        with patch.object(ranking, "_get_embedder", return_value=embedder), \
                patch.object(ranking.jd_index, "get_ranking_snapshot", return_value=snapshot):
            result = ranking.rank_jds("Resume", "resume", preferred_location="Mumbai", preferred_arrangement="Remote")
        self.assertEqual(result.results[0].jd_id, "b")
        self.assertEqual(result.results[0].preference_boost, 8)
        self.assertEqual(result.results[0].embedding_similarity, 60)
        self.assertEqual(result.results[0].similarity_score, 68)

    def test_low_confidence_does_not_reorder_even_if_model_scores_differ(self):
        embedder = MagicMock()
        embedder.embed_chunk.return_value = [1, 0]
        snapshot = [(job("a"), [1, 0]), (job("b"), [0, 1])]
        self.api.return_value = reply(answer((1, 0, 0)), answer((0, 0, 1), confidence=0.1))
        with patch.object(ranking, "_get_embedder", return_value=embedder), \
                patch.object(ranking.jd_index, "get_ranking_snapshot", return_value=snapshot):
            result = ranking.rank_jds("Resume", "resume")
        self.assertEqual([c.jd_id for c in result.results], ["a", "b"])
        self.assertTrue(all(c.model_fit == "Unclear" for c in result.results))
        self.assertEqual(result.ranking_method, "similarity")


if __name__ == "__main__":
    unittest.main()
