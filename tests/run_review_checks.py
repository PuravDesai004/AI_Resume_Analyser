"""Run the chat/upload checks and existing API regressions in isolated storage."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chromadb
from esco_loader import SkillIndex


def main():
    # Keep test writes separate from the application's stored documents.
    storage = ROOT / "scratch" / ("review_checks_" + uuid4().hex)
    storage.mkdir(parents=True)
    taxonomy = storage / "skills_en.csv"
    taxonomy.write_text("conceptUri,preferredLabel,altLabels\n", encoding="utf-8")
    factory = chromadb.PersistentClient
    with patch("chromadb.PersistentClient", side_effect=lambda *a, **k: factory(path=str(storage / "db"))), \
            patch("analysis_pipeline.SkillIndex", side_effect=lambda **kw: SkillIndex(str(taxonomy), str(ROOT / "data/custom_skills.json"))), \
            patch("analysis_pipeline.genai.Client"):
        # The existing API tests exercise sufficiency gates without model calls.
        # The minimal taxonomy is a test fixture, never a replacement for ESCO.
        suite = unittest.TestSuite()
        for pattern in ("test_chat*.py", "test_document_uploads.py", "test_upload_ui.py", "test_analysis_readiness.py", "test_server.py"):
            suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
