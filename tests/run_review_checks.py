"""Run all offline regressions in isolated storage, without changing app data."""
from pathlib import Path
import sys
import unittest
import logging
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import chromadb


def main():
    # Keep test writes separate from the application's stored documents.
    storage = ROOT / "scratch" / ("review_checks_" + uuid4().hex)
    storage.mkdir(parents=True)
    logging.getLogger("streamlit.runtime.scriptrunner_utils.script_run_context").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime.state.session_state_proxy").setLevel(logging.ERROR)
    factory = chromadb.PersistentClient
    with patch("chromadb.PersistentClient", side_effect=lambda *a, **k: factory(path=str(storage / "db"))), \
            patch("analysis_pipeline.genai.Client"), patch("requests.get") as get:
        get.return_value.status_code = 200
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
