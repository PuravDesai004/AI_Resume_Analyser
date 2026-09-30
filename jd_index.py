import uuid
from typing import Optional, Union

import chroma_db
from embedding_manager import EmbeddingManager
from schemas import JDRecord, ErrorResponse

JD_STORE_CAP = 15

# Use ChromaDB client from existing chroma_db.py to avoid modifying Phase 2 files
_client = chroma_db.client
_collection = None
_embedder = None


# In-memory caches to eliminate redundant ChromaDB SQLite queries for <=15 records
_RECORD_CACHE: dict[str, JDRecord] = {}
_EMBEDDING_CACHE: dict[str, list[float]] = {}
_CACHE_INITIALIZED = False


def _get_collection():
    """Gets or initializes the ChromaDB collection without redundant count() queries."""
    global _collection
    if _collection is None:
        _collection = _client.get_or_create_collection(name="job_descriptions")
    return _collection


def _get_embedder() -> EmbeddingManager:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingManager()
    return _embedder


def _ensure_cache_synced():
    """Populates in-memory cache from ChromaDB on startup if needed."""
    global _CACHE_INITIALIZED
    if not _CACHE_INITIALIZED:
        try:
            coll = _get_collection()
            data = coll.get(include=["metadatas", "documents", "embeddings"])
            ids = data.get("ids", [])
            metas = data.get("metadatas", []) or []
            docs = data.get("documents", []) or []
            embs = data.get("embeddings", [])

            _RECORD_CACHE.clear()
            _EMBEDDING_CACHE.clear()

            for i, j_id in enumerate(ids):
                meta = metas[i] if i < len(metas) else {}
                doc = docs[i] if i < len(docs) else ""
                _RECORD_CACHE[j_id] = JDRecord(
                    jd_id=j_id,
                    title=str(meta.get("title", "")),
                    company=str(meta.get("company", "")),
                    location=str(meta.get("location", "")),
                    snippet=str(meta.get("snippet", "")),
                    full_text=doc
                )
                if embs is not None and i < len(embs):
                    _EMBEDDING_CACHE[j_id] = list(embs[i])
        except Exception:
            pass
        _CACHE_INITIALIZED = True


def count() -> int:
    """Current number of stored job descriptions."""
    _ensure_cache_synced()
    return len(_RECORD_CACHE)


def add_jd(
    title: str,
    company: str,
    location: str,
    jd_text: str
) -> Union[JDRecord, ErrorResponse]:
    """
    Adds a new JD to the store if below the 15-record cap.
    Embeds jd_text once and stores vector + metadata in ChromaDB and memory cache.
    """
    _ensure_cache_synced()
    current_count = len(_RECORD_CACHE)
    if current_count >= JD_STORE_CAP:
        return ErrorResponse(
            error_code="STORE_CAP_REACHED",
            message=f"JD store cap of {JD_STORE_CAP} records has been reached",
            stage="jd_store"
        )

    clean_text = str(jd_text or "").strip()
    snippet = clean_text[:150].strip()
    if len(clean_text) > 150:
        snippet += "..."

    embedder = _get_embedder()
    embedding = embedder.embed_chunk(clean_text)

    jd_id = f"jd_{uuid.uuid4().hex[:8]}"

    # Ensure strict ChromaDB primitive types (str, int, float, bool)
    metadata = {
        "title": str(title or "").strip(),
        "company": str(company or "").strip(),
        "location": str(location or "").strip(),
        "snippet": snippet
    }

    _get_collection().add(
        ids=[jd_id],
        embeddings=[embedding],
        documents=[clean_text],
        metadatas=[metadata]
    )

    record = JDRecord(
        jd_id=jd_id,
        title=metadata["title"],
        company=metadata["company"],
        location=metadata["location"],
        snippet=snippet,
        full_text=clean_text
    )

    _RECORD_CACHE[jd_id] = record
    _EMBEDDING_CACHE[jd_id] = embedding

    return record


def get_all_jd_embeddings() -> list[tuple[str, list[float]]]:
    """Fetch all stored JD ids and their vectors for Tier 1 ranking from memory cache."""
    _ensure_cache_synced()
    if _EMBEDDING_CACHE:
        return list(_EMBEDDING_CACHE.items())

    data = _get_collection().get(include=["embeddings"])
    ids = data.get("ids", [])
    embeddings = data.get("embeddings", [])
    if embeddings is None or len(embeddings) == 0:
        return []
    for i, j_id in enumerate(ids):
        _EMBEDDING_CACHE[j_id] = list(embeddings[i])
    return list(zip(ids, embeddings))


def get_jd_record(jd_id: str) -> Optional[JDRecord]:
    """Fetch one JD's full text and metadata for Tier 2 analysis with in-memory lookup."""
    _ensure_cache_synced()
    if jd_id in _RECORD_CACHE:
        return _RECORD_CACHE[jd_id]

    data = _get_collection().get(ids=[jd_id], include=["metadatas", "documents"])
    ids = data.get("ids", [])
    if not ids:
        return None

    meta = data["metadatas"][0] if data.get("metadatas") else {}
    doc = data["documents"][0] if data.get("documents") else ""
    record = JDRecord(
        jd_id=jd_id,
        title=str(meta.get("title", "")),
        company=str(meta.get("company", "")),
        location=str(meta.get("location", "")),
        snippet=str(meta.get("snippet", "")),
        full_text=doc
    )
    _RECORD_CACHE[jd_id] = record
    return record


def clear_store_for_testing() -> None:
    """Clears all stored records in ChromaDB and memory caches."""
    global _collection, _CACHE_INITIALIZED
    _RECORD_CACHE.clear()
    _EMBEDDING_CACHE.clear()
    _CACHE_INITIALIZED = True

    coll = _get_collection()
    try:
        data = coll.get()
        ids = data.get("ids", [])
        if ids:
            coll.delete(ids=ids)
    except Exception:
        try:
            _client.delete_collection(name="job_descriptions")
        except Exception:
            pass
        _collection = _client.get_or_create_collection(name="job_descriptions")


if __name__ == "__main__":
    print("Testing jd_index...")
    print("Initial count:", count())
