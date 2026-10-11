import unittest
from unittest.mock import patch
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

SCRIPT = '''
from chat_ui import render_chat
resumes = [
    {"id": "a", "name": "alice.pdf", "text": "Python experience", "pages": [{"text": "Python experience", "page": 2}]},
    {"id": "b", "name": "bob.txt", "text": "Java experience"},
]
render_chat("http://test.invalid", resumes, [], lambda resume_id, jd_id: None)
'''


class ChatUITests(unittest.TestCase):
    def test_question_reuses_index_and_sends_history_with_pdf_pages(self):
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs["json"]))
            if url.endswith("/chat/documents"):
                body = {"context_id": "a" * 64, "chunk_count": 1, "reused": False}
            else:
                body = {"answer": "Python experience [1].", "context_sufficient": True, "citations": [{
                    "citation_id": 1, "chunk_id": "a", "source": "alice.pdf",
                    "document_type": "resume", "page": 2, "excerpt": "Python experience",
                }]}
            return SimpleNamespace(status_code=200, json=lambda: body, raise_for_status=lambda: None)

        with patch("chat_ui.requests.post", side_effect=post):
            app = AppTest.from_string(SCRIPT, default_timeout=20).run()
            self.assertEqual(len(app.exception), 0)
            app.chat_input[0].set_value("What experience is listed?").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(calls[0][1]["resume"]["pages"][0]["page"], 2)
            app.chat_input[0].set_value("Tell me more.").run()
            self.assertEqual(len(calls), 3)
            self.assertEqual(len(calls[-1][1]["history"]), 2)
            self.assertTrue(any("page 2" in caption.value for caption in app.caption))
            app.selectbox[0].set_value(1).run()
            self.assertEqual(len(app.chat_message), 0)
            app.chat_input[0].set_value("What experience is listed?").run()
            self.assertEqual(calls[-2][1]["resume"]["document_id"], "b")
            self.assertEqual(calls[-1][1]["history"], [])
            app.button[0].click().run()
            self.assertEqual(len(app.chat_message), 0)


if __name__ == "__main__":
    unittest.main()
