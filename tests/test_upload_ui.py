import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def response(status=200, body=None):
    return SimpleNamespace(status_code=status, json=lambda: body or {"count": 0, "cap": None})


class UploadUITests(unittest.TestCase):
    def test_repeated_ranking_reuses_ingestion_and_analysis_and_keeps_candidate(self):
        stored, calls = {}, []

        def get(url, **kw):
            return response(body={"ids": list(stored)}) if url.endswith("/jds/ids") else response()

        def delete(url, **kw):
            calls.append(("DELETE", url))
            stored.clear()
            return response()

        def post(url, **kw):
            calls.append(("POST", url))
            payload = kw["json"]
            if url.endswith("/jds"):
                jid = f"stored-{len(stored)}"
                stored[jid] = payload
                return response(body={"jd_id": jid})
            if url.endswith("/rank"):
                return response(body={"results": [{
                    "jd_id": jid, "rank": i, "similarity_score": 80,
                    "title": job["title"], "company": job["company"],
                    "location": job["location"], "snippet": "Python SQL",
                } for i, (jid, job) in enumerate(stored.items(), 1)]})
            return response(body={
                "resume_id": payload["resume_id"], "jd_id": payload["jd_id"],
                "context_sufficient": True, "match_score": 80, "summary": "Python matches.",
            })

        with patch("requests.get", side_effect=get), patch("requests.delete", side_effect=delete), patch("requests.post", side_effect=post):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
            next(b for b in app.button if b.label == "Load 5 PDF Resumes and 5 PDF JDs").click().run()
            button = lambda: next(b for b in app.button if b.label == "Analyze Placement Fit (Embed and Rank)")
            button().click().run()
            ranked_id = app.session_state["active_ranked_resume_id"]
            first_calls = list(calls)
            button().click().run()
            self.assertEqual(len(app.exception), 0)
            extra_calls = calls[len(first_calls):]
            self.assertEqual(len(extra_calls), 1)
            self.assertTrue(extra_calls[0][0] == "POST" and extra_calls[0][1].endswith("/rank"))
            candidate = next(s for s in app.selectbox if s.label == "Select Candidate Resume for Analysis")
            candidate.set_value(candidate.options[1]).run()
            self.assertEqual(app.session_state["active_ranked_resume_id"], ranked_id)
            first_name = next(r["name"] for r in app.session_state["resumes"] if r["id"] == ranked_id)
            self.assertTrue(any("Target Resume:" in m.value and first_name in m.value for m in app.markdown))
            while app.session_state["resumes"]:
                next(b for b in app.button if b.label == "Remove Resume").click().run()
                self.assertEqual(len(app.exception), 0)
            self.assertIsNone(app.session_state["rank_results"])

    def test_failed_reset_and_ingestion_do_not_use_local_fallback(self):
        with patch("requests.get", return_value=response()), patch("requests.delete", return_value=response(500, {"detail": "Store unavailable"})), \
                patch("jd_index.clear_store_for_testing", side_effect=AssertionError("Must not silently reset locally")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
            app.session_state["resumes"] = [{"id": "r", "name": "Resume", "text": "Python", "words": 1, "status": "Short", "warning": "Short"}]
            next(b for b in app.button if b.label == "Reset Session and Clear Store").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.session_state["resumes"]), 1)
            self.assertTrue(any("Store unavailable" in e.value for e in app.error))

        with patch("requests.get", return_value=response()), patch("requests.delete", return_value=response()), \
                patch("requests.post", return_value=response(503, {"detail": "Embedding service unavailable"})), \
                patch("jd_index.add_jd", side_effect=AssertionError("Must not retry an HTTP error locally")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
            next(b for b in app.button if b.label == "Load 5 PDF Resumes and 5 PDF JDs").click().run()
            next(b for b in app.button if b.label == "Analyze Placement Fit (Embed and Rank)").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Embedding service unavailable" in e.value for e in app.error))

    def test_removing_jd_frees_backend_slot_and_clears_stale_results(self):
        calls = []
        with patch("requests.get", return_value=response()), \
                patch("requests.delete", side_effect=lambda url, **kw: calls.append(url) or response()):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
            next(b for b in app.button if b.label == "Load 5 PDF Resumes and 5 PDF JDs").click().run()
            jobs = app.session_state["jds"]
            jobs[0]["backend_id"] = "stored-to-remove"
            app.session_state["jds"] = jobs
            app.session_state["rank_results"] = [{"jd_id": "stored-to-remove"}]
            next(b for b in app.button if b.label == "Remove JD").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any(url.endswith("/jds/stored-to-remove") for url in calls))
            self.assertEqual(len(app.session_state["jds"]), 4)
            self.assertIsNone(app.session_state["rank_results"])

    def test_upload_error_and_duplicate_notice_survive_reruns_until_dismissed(self):
        path = next((ROOT / "sample_pdfs/resumes").glob("*.pdf"))
        selected = [SimpleNamespace(name=path.name, getvalue=path.read_bytes),
                    SimpleNamespace(name="broken.pdf", getvalue=lambda: b"invalid PDF")]
        with patch("requests.get", return_value=response()), patch("requests.delete", return_value=response()), \
                patch("streamlit.file_uploader", side_effect=lambda *a, **kw: selected if kw["key"] == "res_file_input" else []):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
            next(b for b in app.button if b.label == "Load 5 PDF Resumes and 5 PDF JDs").click().run()
            original_ids = [r["id"] for r in app.session_state["resumes"]]
            next(b for b in app.button if b.label == "Add Uploaded Resumes").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual([r["id"] for r in app.session_state["resumes"]], original_ids)
            self.assertTrue(any("already loaded" in e.value for e in app.info))
            self.assertTrue(any("broken.pdf" in e.value for e in app.error))
            app.run()
            self.assertTrue(any("broken.pdf" in e.value for e in app.error))
            next(b for b in app.button if b.label == "Dismiss upload messages").click().run()
            self.assertFalse(any("broken.pdf" in e.value for e in app.error))

    def test_missing_taxonomy_warning_persists_while_ranking_remains_available(self):
        card = {"jd_id": "stored", "rank": 1, "similarity_score": 80, "title": "Backend",
                "company": "Partner", "location": "Remote", "snippet": "Python SQL"}
        calls = []

        def post(url, **kw):
            calls.append(url)
            if url.endswith("/jds"):
                return response(body={"jd_id": "stored"})
            if url.endswith("/rank"):
                return response(body={"results": [card]})
            return response(503, {"detail": "ESCO taxonomy missing: data/skills_en.csv"})

        with patch("requests.get", return_value=response()), patch("requests.delete", return_value=response()), \
                patch("requests.post", side_effect=post), patch("analysis_pipeline.AnalysisPipeline", side_effect=AssertionError("No local retry")):
            app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
            next(b for b in app.button if b.label == "Load 5 PDF Resumes and 5 PDF JDs").click().run()
            next(b for b in app.button if b.label == "Analyze Placement Fit (Embed and Rank)").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state["rank_results"], [card])
            self.assertTrue(any("skills_en.csv" in e.value for e in app.warning))
            self.assertEqual(sum(url.endswith("/analyze") for url in calls), 1)
            self.assertFalse(any("Analysis complete!" in e.value for e in app.success))
            app.run()
            self.assertTrue(any("skills_en.csv" in e.value for e in app.warning))


if __name__ == "__main__":
    unittest.main()
