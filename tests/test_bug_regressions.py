"""Offline reproductions of storage, scoring, and embedding-space bugs."""
import unittest
import tempfile
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from uuid import uuid4

import chromadb
from fastapi.testclient import TestClient
from langchain_core.documents import Document

import jd_index
import ranking
from chat_schemas import GeneratedChatAnswer
from chat_service import ChatService, ChatServiceError
from document_parsing import extract_document
from rag_engine.embedding_manager import EmbeddingManager
from schemas import GroundedSkill
from skill_comparison import compare_skills


class StoreRegressionTests(unittest.TestCase):
    def setUp(self):
        self.db = chromadb.EphemeralClient()
        self.collection = self.db.create_collection("regression_" + uuid4().hex)
        self.patches = [
            patch.object(jd_index, "_collection", self.collection),
            patch.object(jd_index, "_CACHE_INITIALIZED", False),
            patch.dict(jd_index._RECORD_CACHE, clear=True),
            patch.dict(jd_index._EMBEDDING_CACHE, clear=True),
        ]
        for item in self.patches:
            item.start()
        self.embedder = MagicMock()
        self.embedder.embed_chunk.return_value = [1.0, 0.0, 0.0]
        self.embed_patch = patch.object(jd_index, "_get_embedder", return_value=self.embedder)
        self.embed_patch.start()

    def tearDown(self):
        self.embed_patch.stop()
        for item in reversed(self.patches):
            item.stop()
        self.db.delete_collection(self.collection.name)

    def add(self, title="Python role"):
        return jd_index.add_jd(title, "Co", "Remote", "Python APIs with Docker")

    def test_failed_first_cache_read_is_retried(self):
        coll = MagicMock()
        coll.get.side_effect = [RuntimeError("Temporary read failure"), {
            "ids": ["external"], "metadatas": [{}], "documents": ["Python"],
            "embeddings": [[1.0, 0.0, 0.0]],
        }]
        with patch.object(jd_index, "_get_collection", return_value=coll):
            with self.assertRaises(RuntimeError):
                jd_index.count()
            self.assertFalse(jd_index._CACHE_INITIALIZED)
            self.assertEqual(jd_index.count(), 1)

    def test_cache_observes_external_replacement_even_at_same_count(self):
        old = self.add()
        self.collection.delete(ids=[old.jd_id])
        self.collection.add(ids=["replacement"], documents=["Rust"], embeddings=[[1., 0., 0.]], metadatas=[{"title": "Rust role"}])
        self.assertEqual(jd_index.count(), 1)
        self.assertIsNone(jd_index.get_jd_record(old.jd_id))
        self.assertEqual(jd_index.get_jd_record("replacement").full_text, "Rust")
        self.assertEqual(jd_index.get_all_jd_embeddings()[0][0], "replacement")

    def test_failed_reset_does_not_discard_cache(self):
        record = self.add()
        coll = MagicMock()
        coll.get.return_value = {"ids": [record.jd_id]}
        coll.delete.side_effect = RuntimeError("Delete failed")
        with patch.object(jd_index, "_get_collection", return_value=coll):
            with self.assertRaises(RuntimeError):
                jd_index.clear_store_for_testing()
        self.assertIn(record.jd_id, jd_index._RECORD_CACHE)
        self.assertEqual(self.collection.count(), 1)

    def test_duplicate_ingestion_reuses_vector_and_id(self):
        first, second = self.add(), self.add()
        self.assertEqual(first.jd_id, second.jd_id)
        self.assertEqual(jd_index.count(), 1)
        self.embedder.embed_chunk.assert_called_once()

    def test_concurrent_requests_cannot_exceed_cap(self):
        with ThreadPoolExecutor(max_workers=5) as pool:
            records = list(pool.map(lambda i: self.add(f"Role {i}"), range(20)))
        self.assertEqual(sum(hasattr(r, "jd_id") for r in records), 15)
        self.assertEqual(jd_index.count(), 15)
        self.assertEqual(self.embedder.embed_chunk.call_count, 15)

    def test_repeated_ranking_reuses_resume_embedding_and_reads_one_snapshot(self):
        self.add()
        resume_embedder = MagicMock()
        resume_embedder.embed_chunk.return_value = [1., 0., 0.]
        tracked = MagicMock(wraps=self.collection)
        with patch.object(ranking, "_get_embedder", return_value=resume_embedder), \
                patch.object(jd_index, "_get_collection", return_value=tracked):
            first = ranking.rank_jds("Python resume", "alice")
            second = ranking.rank_jds("Python resume", "alice", preferred_location="Remote")
            self.assertEqual(first.results[0].jd_id, second.results[0].jd_id)
            resume_embedder.embed_chunk.assert_called_once()
            self.assertEqual(tracked.get.call_count, 2)
            ranking.rank_jds("Changed Rust resume", "alice")
            self.assertEqual(resume_embedder.embed_chunk.call_count, 2)

    def test_api_empty_jd_ids_and_individual_deletion(self):
        import server
        api = TestClient(server.app)
        result = api.post("/jds", json={"title": "Role", "company": "Co", "location": "Remote", "jd_text": "   "})
        self.assertEqual(result.status_code, 422)
        self.embedder.embed_chunk.assert_not_called()
        record = self.add()
        self.assertIn(record.jd_id, api.get("/jds/ids").json()["ids"])
        self.assertEqual(api.delete(f"/jds/{record.jd_id}").status_code, 200)
        self.assertEqual(api.get("/jds/ids").json()["ids"], [])


def skill(name="Python", match_type="exact"):
    return GroundedSkill(original_name=name, standardized_name=name, skill_id="custom:" + name,
                         source="custom", match_type=match_type)


class ScoringRegressionTests(unittest.TestCase):
    def test_empty_requirements_do_not_produce_a_perfect_score(self):
        self.assertEqual(compare_skills([skill()], [], [])[1], 0)

    def test_fuzzy_jd_match_is_not_reported_as_exact(self):
        gap, _ = compare_skills([skill()], [skill(match_type="close")], [])
        self.assertEqual(gap.matched[0].match_type, "close")

    def test_exact_duplicate_resume_evidence_takes_precedence(self):
        gap, _ = compare_skills([skill(match_type="close"), skill()], [skill()], [])
        self.assertEqual(gap.matched[0].match_type, "exact")

    def test_pipeline_normalizes_ids_and_rejects_unverified_requirements(self):
        from analysis_pipeline import AnalysisPipeline
        pipeline = AnalysisPipeline.__new__(AnalysisPipeline)
        pipeline.skill_index = MagicMock()
        pipeline.skill_index.exact_match.return_value = None
        pipeline.skill_index.fuzzy_match.return_value = None
        pipeline.client = MagicMock()
        pipeline.client.models.generate_content.return_value.text = '{"resume_skills":[],"jd_essential_skills":[],"jd_preferred_skills":[]}'
        with patch.object(jd_index, "get_jd_record", return_value=SimpleNamespace(full_text="requirement " * 40)):
            result = pipeline.analyze("resume " * 40, " alice ", " job ")
        self.assertFalse(result["context_sufficient"])
        self.assertEqual(result["match_score"], 0)
        self.assertEqual((result["resume_id"], result["jd_id"]), ("alice", "job"))


class EmbeddingRegressionTests(unittest.TestCase):
    def test_cleanup_removes_session_from_all_embedding_spaces(self):
        db = chromadb.EphemeralClient()
        collections = [db.create_collection("resume_chat_" + uuid4().hex) for _ in range(2)]
        session = str(uuid4())
        try:
            for coll in collections:
                coll.add(ids=["selected", "other"], documents=["Python", "Rust"], embeddings=[[1., 0., 0.]] * 2,
                         metadatas=[{"session_id": session}, {"session_id": "other-session"}])
            with patch("chromadb.PersistentClient", return_value=db):
                self.assertEqual(ChatService().delete_session(session), 2)
            self.assertEqual([c.count() for c in collections], [1, 1])
        finally:
            for coll in collections:
                db.delete_collection(coll.name)

    def manager(self):
        manager = EmbeddingManager("gemini-embedding-001")
        manager._client = MagicMock()
        return manager

    def test_remote_failure_after_indexing_never_switches_query_space(self):
        manager = self.manager()
        manager._client.models.embed_content.return_value = SimpleNamespace(embeddings=[SimpleNamespace(values=[1., 0., 0.])])
        manager.embed_chunks_batch(["Python"])
        manager._client.models.embed_content.side_effect = RuntimeError("Provider unavailable")
        with patch.object(manager, "_get_local_model", side_effect=AssertionError("Must not mix vectors")):
            with self.assertRaises(RuntimeError):
                manager.embed_chunk("What skills?")
        self.assertEqual(manager.embedding_space, "gemini-embedding-001")

    def test_persisted_remote_vectors_pin_query_space_after_restart(self):
        manager = self.manager()
        manager.bind_embedding_space(manager.embedding_space)
        manager._client.models.embed_content.side_effect = RuntimeError("Unavailable")
        with patch.object(manager, "_get_local_model", side_effect=AssertionError("Must not mix persisted vectors")):
            with self.assertRaises(RuntimeError):
                manager.embed_chunk("Skills?")

    def test_missing_client_cannot_switch_a_bound_space(self):
        manager = self.manager()
        manager.bind_embedding_space(manager.embedding_space)
        with patch.object(EmbeddingManager, "client", new=property(lambda self: None)):
            with self.assertRaises(RuntimeError):
                manager.embed_chunks_batch(["Python"])

    def test_first_failure_switches_to_identifiable_local_space(self):
        import numpy as np
        manager = self.manager()
        manager._client.models.embed_content.side_effect = RuntimeError("Unavailable")
        local = MagicMock()
        local.encode.return_value = np.array([[1., 0., 0.]])
        with patch.object(manager, "_get_local_model", return_value=local):
            self.assertEqual(manager.embed_chunks_batch(["Python"]), [[1., 0., 0.]])
        self.assertEqual(manager.embedding_space, "sentence-transformers:all-MiniLM-L6-v2")

    def test_chat_fallback_uses_separate_collection_and_all_vectors(self):
        from test_chat import context
        db = chromadb.EphemeralClient()
        prefix = "fallback_" + uuid4().hex
        collections = {}

        def get_collection(name, **kwargs):
            if name not in collections:
                collections[name] = db.create_collection(prefix + str(len(collections)))
            return collections[name]

        embedder = MagicMock()
        embedder.embedding_space = "gemini-embedding-001"

        def embed(texts):
            embedder.embedding_space = "sentence-transformers:all-MiniLM-L6-v2"
            return [[1., 0., 0.] for _ in texts]

        embedder.embed_chunks_batch.side_effect = embed
        client = SimpleNamespace(get_or_create_collection=get_collection)
        service = ChatService(embedder=embedder, splitter=lambda d: [(p.text, -1) for p in d.pages])
        try:
            with patch("chromadb.PersistentClient", return_value=client):
                result = service.prepare(context())
                self.assertEqual(len(collections), 2)
                self.assertEqual([c.count() for c in collections.values()], [0, 1])
                self.assertTrue(service.prepare(context(session=uuid4())).chunk_count)
                self.assertEqual(result.chunk_count, 1)
        finally:
            for coll in collections.values():
                db.delete_collection(coll.name)

    def test_empty_embeddings_and_answers_are_rejected(self):
        with self.assertRaises(ValueError):
            self.manager().embed_chunk(" ")
        generated = GeneratedChatAnswer(answer="[1]", context_sufficient=True, cited_chunk_ids=["a"])
        selected = [{"id": "a", "document": "Python", "metadata": {"source": "r.txt", "document_type": "resume"}}]
        with self.assertRaises(ChatServiceError):
            ChatService._answer(generated, selected)


class LegacyRAGRegressionTests(unittest.TestCase):
    def test_pdf_with_images_remains_readable_without_optional_ocr(self):
        import pymupdf
        import chunking
        from rag_engine import chunking as chat_chunking
        with tempfile.TemporaryDirectory() as folder:
            pdf = pymupdf.open()
            page = pdf.new_page()
            page.insert_text((40, 40), "Python experience")
            pixel = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 1, 1), False)
            pixel.clear_with(255)
            page.insert_image(pymupdf.Rect(40, 80, 60, 100), pixmap=pixel)
            pdf.save(str(Path(folder) / "resume.pdf"))
            pdf.close()
            for module in (chunking, chat_chunking):
                with patch.object(module, "find_spec", return_value=None):
                    docs = module.read_documents(folder)
                self.assertIn("Python experience", docs[0].page_content)

    def test_taxonomy_cache_reloads_when_source_files_change(self):
        from esco_loader import SkillIndex
        with tempfile.TemporaryDirectory() as folder:
            csv = Path(folder) / "skills.csv"
            custom = Path(folder) / "custom.json"
            csv.write_text("conceptUri,preferredLabel,altLabels\n", encoding="utf-8")
            custom.write_text('[{"id":"custom:python","preferred_label":"Python"}]', encoding="utf-8")
            first = SkillIndex(str(csv), str(custom))
            custom.write_text('[{"id":"custom:rust","preferred_label":"Rust language"}]', encoding="utf-8")
            second = SkillIndex(str(csv), str(custom))
            self.assertIsNotNone(second.exact_match("Rust language"))
            self.assertIsNone(second.exact_match("Python"))
            self.assertIsNotNone(first.exact_match("Python"))

    def test_default_taxonomy_paths_work_outside_project_directory(self):
        from esco_loader import SkillIndex
        from analysis_pipeline import AnalysisPipeline
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            try:
                os.chdir(folder)
                self.assertIsNotNone(SkillIndex().exact_match("Python"))
                with patch("analysis_pipeline.genai.Client"):
                    self.assertIsNotNone(AnalysisPipeline().skill_index.exact_match("Python"))
            finally:
                os.chdir(previous)

    def test_legacy_generation_falls_back_after_missing_model(self):
        import query
        rag = query.RAGPipeline()
        rag._embedder = MagicMock()
        rag._client = MagicMock()
        rag._client.models.generate_content.side_effect = [RuntimeError("404 model missing"), SimpleNamespace(text="Recovered answer")]
        corpus = [{"id": "a", "document": "Python", "metadata": {}}]
        with patch.object(query, "get_all_chunks_from_db", return_value=corpus), \
                patch.object(query, "query_collection", return_value={"ids": [["a"]], "documents": [["Python"]], "metadatas": [[{}]]}), \
                patch.object(query, "_get_shared_cross_encoder", return_value=None):
            self.assertEqual(rag.run("Skills?"), "Recovered answer")
        self.assertEqual(rag._client.models.generate_content.call_count, 2)

    def test_ingestion_keeps_distinct_sources_and_is_idempotent(self):
        import chroma_db
        db = chromadb.EphemeralClient()
        coll = db.create_collection("ingestion_" + uuid4().hex)
        a = Document(page_content="Python", metadata={"source": "a.txt"})
        b = Document(page_content="Python", metadata={"source": "b.txt"})
        try:
            with patch.object(chroma_db, "collection", coll):
                chroma_db.add_chunks_to_collection([a], [[1., 0., 0.]])
                chroma_db.add_chunks_to_collection([b], [[1., 0., 0.]])
                chroma_db.add_chunks_to_collection([a], [[1., 0., 0.]])
                self.assertEqual(coll.count(), 2)
                chroma_db.add_chunks_to_collection([a, b], [[1., 0., 0.]] * 2)
                self.assertEqual(coll.count(), 2)
                chroma_db.add_chunks_to_collection([], [])
                with self.assertRaises(ValueError):
                    chroma_db.add_chunks_to_collection([a], [])
        finally:
            db.delete_collection(coll.name)

    def test_keyword_search_handles_blank_corpus_and_fusion_keeps_sources(self):
        import retrieval
        from rag_engine import retrieval as chat_retrieval
        a, b = {"id": "a", "document": "Python"}, {"id": "b", "document": "Python"}
        self.assertEqual(len(retrieval.merge_rrf([a, b], [(a, 1)])), 2)
        for module in (retrieval, chat_retrieval):
            self.assertIsNone(module.build_bm25_index([{"document": "   "}]))

    def test_legacy_query_refreshes_corpus_after_upload(self):
        import query
        a = {"id": "a", "document": "Python", "metadata": {}}
        b = {"id": "b", "document": "Rust", "metadata": {}}
        rag = query.RAGPipeline()
        rag._embedder = MagicMock()
        rag._client = MagicMock()
        rag._client.models.generate_content.return_value.text = "Answer"
        with patch.object(query, "get_all_chunks_from_db", side_effect=[[a], [a, b]]), \
                patch.object(query, "query_collection", return_value={"ids": [["a"]], "documents": [["Python"]], "metadatas": [[{}]]}), \
                patch.object(query, "_get_shared_cross_encoder", return_value=None), \
                patch.object(query, "build_prompt", return_value="prompt") as prompt:
            rag.run("Skills?")
            rag.run("Rust?")
        self.assertIn("Rust", [c["document"] for c in prompt.call_args.args[1]])

    def test_utf8_bom_does_not_create_an_empty_upload(self):
        text, pages = extract_document(b"\xef\xbb\xbf  ", "blank.txt")
        self.assertEqual((text, pages), ("", []))


if __name__ == "__main__":
    unittest.main()
