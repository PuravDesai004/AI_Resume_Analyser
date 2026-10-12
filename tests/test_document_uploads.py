import unittest
from pathlib import Path
from types import SimpleNamespace

from document_parsing import extract_document
from document_uploads import prepare_uploads


def upload(name, content):
    return SimpleNamespace(name=name, getvalue=lambda: content)


class DocumentUploadTests(unittest.TestCase):
    def prepare(self, files, existing=(), kind="resume", cap=10):
        return prepare_uploads(files, existing, kind, lambda text: ("Valid", "Ready"), 5 * 1024 * 1024, cap)

    def test_fixture_pdf_already_loaded_is_skipped_without_changing_identity(self):
        path = next((Path(__file__).resolve().parents[1] / "sample_pdfs/resumes").glob("*.pdf"))
        text, pages = extract_document(path.read_bytes(), path.name)
        existing = [{"id": "original", "text": "  " + text.replace("\n", "\n\n") + "  "}]
        records, notices = self.prepare([upload(path.name, path.read_bytes())], existing)
        self.assertEqual(records, [])
        self.assertIn("already loaded", notices[0][1])
        self.assertEqual(existing[0]["id"], "original")

    def test_identical_content_in_same_batch_skipped_but_updated_same_filename_added(self):
        files = [upload("resume.txt", b"Python engineer"), upload("copy.txt", b"Python engineer"),
                 upload("resume.txt", b"Python and SQL engineer")]
        records, notices = self.prepare(files)
        self.assertEqual(len(records), 2)
        self.assertNotEqual(records[0]["id"], records[1]["id"])
        self.assertEqual([level for level, _ in notices], ["success", "info", "success"])

    def test_invalid_empty_oversize_files_do_not_discard_valid_upload(self):
        records, notices = self.prepare([
            upload("broken.pdf", b"invalid PDF"), upload("blank.txt", b"  "),
            upload("large.txt", b"a" * (5 * 1024 * 1024 + 1)), upload("valid.txt", b"SQL experience"),
        ], kind="jd")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source"], "valid.txt")
        self.assertIsNone(records[0]["pages"][0]["page"])
        self.assertEqual([level for level, _ in notices], ["error", "error", "error", "success"])

    def test_duplicate_at_capacity_is_identified_and_new_files_rejected(self):
        records, notices = self.prepare(
            [upload("copy.txt", b"Existing"), upload("new.txt", b"New content")],
            existing=[{"text": "Existing"}], cap=1,
        )
        self.assertEqual(records, [])
        self.assertEqual([level for level, _ in notices], ["info", "warning"])


if __name__ == "__main__":
    unittest.main()
