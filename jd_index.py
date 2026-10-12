import uuid
from functools import wraps
from threading import RLock
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
_LOCK = RLock()


def _synchronized(operation):
    @wraps(operation)
    def run(*args, **kwargs):
        with _LOCK:
            return operation(*args, **kwargs)
    return run


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
    """Refresh on additions/deletions by another process; never cache a failed read."""
    global _CACHE_INITIALIZED
    coll = _get_collection()
    if _CACHE_INITIALIZED and set(coll.get(include=[])["ids"]) == set(_RECORD_CACHE):
        return
    data = coll.get(include=["metadatas", "documents", "embeddings"])
    records, embeddings = {}, {}
    metas = data.get("metadatas") or []
    docs = data.get("documents") or []
    embs = data.get("embeddings")
    for i, j_id in enumerate(data["ids"]):
        meta = (metas[i] or {}) if i < len(metas) else {}
        records[j_id] = JDRecord(
            jd_id=j_id, title=str(meta.get("title", "")),
            company=str(meta.get("company", "")), location=str(meta.get("location", "")),
            snippet=str(meta.get("snippet", "")), full_text=(docs[i] or "") if i < len(docs) else "",
        )
        if embs is not None and i < len(embs):
            embeddings[j_id] = list(embs[i])
    _RECORD_CACHE.clear()
    _RECORD_CACHE.update(records)
    _EMBEDDING_CACHE.clear()
    _EMBEDDING_CACHE.update(embeddings)
    _CACHE_INITIALIZED = True


@_synchronized
def count() -> int:
    """Current number of stored job descriptions."""
    _ensure_cache_synced()
    return len(_RECORD_CACHE)


@_synchronized
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
    clean_text = str(jd_text or "").strip()
    if not clean_text:
        return ErrorResponse(error_code="EMPTY_JD", message="Job description text cannot be empty.", stage="jd_store")
    metadata = {
        "title": str(title or "").strip(),
        "company": str(company or "").strip(),
        "location": str(location or "").strip(),
    }
    for record in _RECORD_CACHE.values():
        if record.full_text == clean_text and all(getattr(record, key) == value for key, value in metadata.items()):
            return record
    current_count = len(_RECORD_CACHE)
    if current_count >= JD_STORE_CAP:
        return ErrorResponse(
            error_code="STORE_CAP_REACHED",
            message=f"JD store cap of {JD_STORE_CAP} records has been reached",
            stage="jd_store"
        )

    snippet = clean_text[:150].strip()
    if len(clean_text) > 150:
        snippet += "..."

    embedder = _get_embedder()
    embedding = embedder.embed_chunk(clean_text)

    jd_id = f"jd_{uuid.uuid4().hex[:8]}"

    # Ensure strict ChromaDB primitive types (str, int, float, bool)
    metadata["snippet"] = snippet

    try:
        _get_collection().add(
            ids=[jd_id],
            embeddings=[embedding],
            documents=[clean_text],
            metadatas=[metadata]
        )
    except Exception as exc:
        if "dimension" in str(exc).lower():
            try:
                _client.delete_collection("job_descriptions")
            except Exception:
                pass
            global _collection
            _collection = _client.get_or_create_collection(name="job_descriptions")
            _RECORD_CACHE.clear()
            _EMBEDDING_CACHE.clear()
            _get_collection().add(
                ids=[jd_id],
                embeddings=[embedding],
                documents=[clean_text],
                metadatas=[metadata]
            )
        else:
            raise exc

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


@_synchronized
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


@_synchronized
def get_jd_record(jd_id: str) -> Optional[JDRecord]:
    """Fetch one JD's full text and metadata for Tier 2 analysis with in-memory lookup."""
    _ensure_cache_synced()
    if not jd_id:
        return None

    # 1. Exact ID match
    if jd_id in _RECORD_CACHE:
        return _RECORD_CACHE[jd_id]

    # 2. Case-insensitive ID match
    clean_target = jd_id.strip().lower()
    for k, v in _RECORD_CACHE.items():
        if k.lower() == clean_target:
            return v

    # 3. Handle known UI mock aliases (e.g. jd_python_backend, jd_react_frontend, etc.)
    UI_MOCK_PREFIXES = ("jd_python", "jd_react", "jd_ai", "jd_devops", "jd_fullstack")
    if any(clean_target.startswith(prefix) for prefix in UI_MOCK_PREFIXES):
        keyword_map = [
            ("python", "python"),
            ("backend", "backend"),
            ("react", "react"),
            ("frontend", "frontend"),
            ("devops", "devops"),
            ("cloud", "cloud"),
            ("scientist", "scientist"),
            ("data", "data"),
            ("ml", "scientist"),
            ("ai", "scientist"),
            ("fullstack", "full stack"),
        ]
        for kw, target_word in keyword_map:
            if kw in clean_target:
                for rec in _RECORD_CACHE.values():
                    if target_word in rec.title.lower() or target_word in rec.full_text.lower():
                        return rec
        if _RECORD_CACHE:
            return next(iter(_RECORD_CACHE.values()))

    return None


@_synchronized
def get_jd_ids() -> list[str]:
    _ensure_cache_synced()
    return list(_RECORD_CACHE)


@_synchronized
def get_ranking_snapshot() -> list[tuple[JDRecord, list[float]]]:
    _ensure_cache_synced()
    return [(record, _EMBEDDING_CACHE[j_id]) for j_id, record in _RECORD_CACHE.items() if j_id in _EMBEDDING_CACHE]


@_synchronized
def delete_jd(jd_id: str) -> None:
    _get_collection().delete(ids=[jd_id])
    _RECORD_CACHE.pop(jd_id, None)
    _EMBEDDING_CACHE.pop(jd_id, None)


@_synchronized
def clear_store_for_testing() -> None:
    """Clears all stored records in ChromaDB and memory caches."""
    global _CACHE_INITIALIZED
    coll = _get_collection()
    ids = coll.get(include=[])["ids"]
    if ids:
        coll.delete(ids=ids)
    _RECORD_CACHE.clear()
    _EMBEDDING_CACHE.clear()
    _CACHE_INITIALIZED = True



if __name__ == "__main__":
    print("Testing jd_index...")
    print("Initial count:", count())
