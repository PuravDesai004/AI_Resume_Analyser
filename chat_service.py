"""Connect document RAG to immutable, session-scoped resume/JD contexts."""
import hashlib
import json
import os
import re
from pathlib import Path
from threading import RLock
from dotenv import load_dotenv

load_dotenv(override=True)

from chat_schemas import (
    ChatCitation, ChatContextRequest, ChatContextResponse, ChatRequest, ChatResponse,
    GeneratedChatAnswer,
)
from rag_engine.retrieval import (
    bm25_search, build_bm25_index, get_all_chunks_from_db, merge_rrf,
    query_collection, rerank,
)

INSUFFICIENT_ANSWER = "The selected documents do not provide enough information to answer that."


class ChatServiceError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(message)


def split_pages(document):
    from langchain_core.documents import Document
    from rag_engine.chunking import CHUNK_OVERLAP, CHUNK_SIZE, split_documents
    pages = [
        Document(page_content=p.text, metadata={"page": p.page or -1})
        for p in document.pages if p.text.strip()
    ]
    return [
        (c.page_content, c.metadata.get("page", -1))
        for c in split_documents(pages, CHUNK_SIZE, CHUNK_OVERLAP)
        if c.page_content.strip()
    ]


class ChatService:
    def __init__(self, collection=None, embedder=None, generator=None, splitter=split_pages):
        self.model = os.getenv("RAG_EMBEDDING_MODEL", "gemini-embedding-001")
        self._collection = collection
        self._embedder = embedder
        self._generator = generator
        self._splitter = splitter
        self._encoder = None
        self._encoder_attempted = False
        self._lock = RLock()

    @property
    def collection(self):
        if self._collection is None:
            import chromadb
            path = Path(__file__).resolve().parent / "my_local_db"
            client = chromadb.PersistentClient(path=str(path))
            suffix = hashlib.sha256(self.model.encode()).hexdigest()[:12]
            self._collection = client.get_or_create_collection(
                name=f"resume_chat_{suffix}", metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def embedder(self):
        if self._embedder is None:
            from rag_engine.embedding_manager import EmbeddingManager
            try:
                self._embedder = EmbeddingManager(self.model)
            except ValueError as exc:
                raise ChatServiceError(503, str(exc)) from exc
        return self._embedder

    @property
    def generator(self):
        if self._generator is None:
            from rag_engine.query import RAGPipeline
            self._generator = RAGPipeline()
        return self._generator

    @staticmethod
    def scope(session_id, context_id):
        return {"$and": [
            {"session_id": {"$eq": str(session_id)}},
            {"context_id": {"$eq": context_id}},
        ]}

    def prepare(self, request: ChatContextRequest):
        encoded = json.dumps({
            "version": 1, "embedding_model": self.model,
            "context": request.model_dump(mode="json"),
        }, sort_keys=True, ensure_ascii=False)
        context_id = hashlib.sha256(encoded.encode()).hexdigest()
        scope = self.scope(request.session_id, context_id)
        with self._lock:
            existing = self.collection.get(where=scope, include=["metadatas"])
            if existing["ids"]:
                return ChatContextResponse(
                    context_id=context_id, chunk_count=len(existing["ids"]), reused=True,
                )
            chunks = []
            for kind, doc in (("resume", request.resume), ("jd", request.jd)):
                if doc is None:
                    continue
                parts = self._splitter(doc)
                if not parts:
                    raise ChatServiceError(422, f"The {kind} has no readable text.")
                for text, page in parts:
                    chunks.append((text, doc.source, kind, page))
            if request.analysis:
                chunks.append((request.analysis.model_dump_json(), "Placement analysis", "analysis", -1))
            texts = [c[0] for c in chunks]
            text_hashes = [hashlib.sha256(t.encode()).hexdigest() for t in texts]
            cached = self.collection.get(where={"$and": [
                {"session_id": {"$eq": str(request.session_id)}},
                {"text_hash": {"$in": list(dict.fromkeys(text_hashes))}},
            ]}, include=["embeddings", "metadatas"])
            vectors = {}
            if cached.get("embeddings") is not None:
                for meta, vector in zip(cached["metadatas"], cached["embeddings"]):
                    vectors[meta["text_hash"]] = list(vector)
            missing = dict((h, text) for h, text in zip(text_hashes, texts) if h not in vectors)
            # Obtain all embeddings before storing, so a remote failure cannot expose
            # a partially indexed context. Typical resume/JD contexts are small.
            try:
                missing_hashes, missing_texts = list(missing), list(missing.values())
                new_vectors = []
                for start in range(0, len(missing_texts), 90):
                    new_vectors.extend(self.embedder.embed_chunks_batch(missing_texts[start:start + 90]))
                if len(new_vectors) != len(missing_texts):
                    raise RuntimeError("Embedding count does not match document chunks.")
                vectors.update(zip(missing_hashes, new_vectors))
                embeddings = [vectors[h] for h in text_hashes]
            except ChatServiceError:
                raise
            except Exception as exc:
                raise ChatServiceError(503, "Document indexing failed. Check Gemini access and retry.") from exc
            ids = [f"{context_id}:{i}" for i in range(len(chunks))]
            metadata = [{
                "session_id": str(request.session_id), "context_id": context_id,
                "resume_id": request.resume.document_id,
                "jd_id": request.jd.document_id if request.jd else "",
                "source": source, "document_type": kind, "page": page, "text_hash": text_hash,
            } for (_, source, kind, page), text_hash in zip(chunks, text_hashes)]
            try:
                self.collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadata)
            except Exception:
                self.collection.delete(where=scope)
                raise
            return ChatContextResponse(context_id=context_id, chunk_count=len(chunks), reused=False)

    def _rerank(self, query, chunks):
        if os.getenv("RAG_RERANK", "false").lower() == "true":
            if not self._encoder_attempted:
                self._encoder_attempted = True
                try:
                    from sentence_transformers import CrossEncoder
                    self._encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                except Exception:
                    self._encoder = None
            try:
                return rerank(query, chunks, self._encoder, top_n=len(chunks))
            except Exception:
                pass
        return chunks

    def ask(self, request: ChatRequest):
        with self._lock:
            scope = self.scope(request.session_id, request.context_id)
            # Rebuild BM25 from the current snapshot; never share an index across
            # candidates or rely on an upload-stale singleton corpus.
            corpus = get_all_chunks_from_db(self.collection, scope)
            if not corpus:
                raise ChatServiceError(404, "Chat context was not found. Prepare the selected documents again.")
            history = request.history[-6:]
            search_text = "\n".join([m.content[:2_000] for m in history] + [request.message])
            try:
                dense = query_collection(
                    search_text, self.embedder, min(12, len(corpus)), self.collection, scope,
                )
            except ChatServiceError:
                raise
            except Exception as exc:
                raise ChatServiceError(503, "Document search failed. Check Gemini access and retry.") from exc
            vector = [
                {"id": cid, "document": text, "metadata": meta}
                for cid, text, meta in zip(dense["ids"][0], dense["documents"][0], dense["metadatas"][0])
            ]
            keyword = bm25_search(search_text, build_bm25_index(corpus), corpus, 12)
            ranked = self._rerank(search_text, merge_rrf(vector, keyword))
            selected = ranked[:6]
            # Give short documents full coverage, and always include authoritative
            # analysis plus evidence from each selected document for comparisons.
            if sum(len(c["document"]) for c in corpus) <= 12_000:
                extras = corpus
            else:
                extras = [c for c in corpus if c["metadata"]["document_type"] == "analysis"]
                for kind in ("resume", "jd"):
                    extras += [c for c in ranked if c["metadata"]["document_type"] == kind][:2]
            seen = {c["id"] for c in selected}
            for chunk in extras:
                if chunk["id"] not in seen:
                    selected.append(chunk)
                    seen.add(chunk["id"])
            try:
                generated = self.generator.run(request.message, selected, history)
                generated = GeneratedChatAnswer.model_validate(generated)
            except Exception as exc:
                raise ChatServiceError(502, "Chat models are unavailable or returned an invalid answer. Please retry.") from exc
            return self._answer(generated, selected)

    @staticmethod
    def _answer(generated, selected):
        if not generated.context_sufficient:
            return ChatResponse(answer=INSUFFICIENT_ANSWER, context_sufficient=False)
        ids = list(dict.fromkeys(generated.cited_chunk_ids))
        available = {c["id"]: c for c in selected}
        if not ids or any(cid not in available for cid in ids):
            raise ChatServiceError(502, "The answer contained invalid sources. Please retry.")
        citations = []
        for index, cid in enumerate(ids, 1):
            chunk = available[cid]
            meta = chunk["metadata"]
            page = meta.get("page", -1)
            citations.append(ChatCitation(
                citation_id=index, chunk_id=cid, source=meta["source"],
                document_type=meta["document_type"], page=page if page > 0 else None,
                excerpt=chunk["document"][:800],
            ))
        # Source cards are assigned numbers here, never from model-generated
        # numbering. Discard leftover inline markers from either provider.
        answer = re.sub(r"\s*\[\d+\]", "", generated.answer).strip()
        return ChatResponse(answer=answer, context_sufficient=True, citations=citations)

    def delete_session(self, session_id):
        with self._lock:
            where = {"session_id": {"$eq": str(session_id)}}
            count = len(self.collection.get(where=where, include=["metadatas"])["ids"])
            self.collection.delete(where=where)
            return count

    def delete_document(self, session_id, document_id):
        with self._lock:
            where = {"$and": [
                {"session_id": {"$eq": str(session_id)}},
                {"$or": [{"resume_id": {"$eq": document_id}}, {"jd_id": {"$eq": document_id}}]},
            ]}
            count = len(self.collection.get(where=where, include=["metadatas"])["ids"])
            self.collection.delete(where=where)
            return count
