import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def response(status=200, body=None):
    return SimpleNamespace(status_code=status, json=lambda: body or {"count": 0, "cap": 15})


class UploadUITests(unittest.TestCase):
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
