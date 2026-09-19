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


def _get_collection():
    """Dynamically gets or refreshes the ChromaDB collection to prevent stale collection UUID errors."""
    global _collection
    if _collection is not None:
        try:
            _ = _collection.count()
            return _collection
        except Exception:
            _collection = None
    _collection = _client.get_or_create_collection(name="job_descriptions")
    return _collection


def _get_embedder() -> EmbeddingManager:
    global _embedder
    if _embedder is None:
        _embedder = EmbeddingManager()
    return _embedder


def count() -> int:
    """Current number of stored job descriptions."""
    return _get_collection().count()


def add_jd(
    title: str,
    company: str,
    location: str,
    jd_text: str
) -> Union[JDRecord, ErrorResponse]:
    """
    Adds a new JD to the store if below the 15-record cap.
    Embeds jd_text once and stores vector + metadata in ChromaDB.
    """
    current_count = count()
    if current_count >= JD_STORE_CAP:
        return ErrorResponse(
            error_code="STORE_CAP_REACHED",
            message=f"JD store cap of {JD_STORE_CAP} records has been reached",
            stage="jd_store"
        )

    jd_id = f"jd_{uuid.uuid4().hex[:8]}"
    clean_text = jd_text.strip()
    snippet = clean_text[:150].strip()
    if len(clean_text) > 150:
        snippet += "..."

    embedder = _get_embedder()
    embedding = embedder.embed_chunk(clean_text)

    metadata = {
        "title": title.strip(),
        "company": company.strip(),
        "location": location.strip(),
        "snippet": snippet
    }

    _get_collection().add(
        ids=[jd_id],
        embeddings=[embedding],
        documents=[clean_text],
        metadatas=[metadata]
    )

    return JDRecord(
        jd_id=jd_id,
        title=title.strip(),
        company=company.strip(),
        location=location.strip(),
        snippet=snippet,
        full_text=clean_text
    )


def get_all_jd_embeddings() -> list[tuple[str, list[float]]]:
    """Fetch all stored JD ids and their vectors for Tier 1 ranking."""
    data = _get_collection().get(include=["embeddings"])
    ids = data.get("ids", [])
    embeddings = data.get("embeddings", [])
    if embeddings is None or len(embeddings) == 0:
        return []
    return list(zip(ids, embeddings))


def get_jd_record(jd_id: str) -> Optional[JDRecord]:
    """Fetch one JD's full text and metadata for Tier 2 analysis."""
    data = _get_collection().get(ids=[jd_id], include=["metadatas", "documents"])
    ids = data.get("ids", [])
    if not ids:
        return None

    meta = data["metadatas"][0]
    doc = data["documents"][0]
    return JDRecord(
        jd_id=jd_id,
        title=meta.get("title", ""),
        company=meta.get("company", ""),
        location=meta.get("location", ""),
        snippet=meta.get("snippet", ""),
        full_text=doc
    )


def clear_store_for_testing() -> None:
    """Clears all stored records without deleting the collection object."""
    coll = _get_collection()
    try:
        data = coll.get()
        ids = data.get("ids", [])
        if ids:
            coll.delete(ids=ids)
    except Exception:
        global _collection
        try:
            _client.delete_collection(name="job_descriptions")
        except Exception:
            pass
        _collection = _client.get_or_create_collection(name="job_descriptions")


if __name__ == "__main__":
    print("Testing jd_index...")
    print("Initial count:", count())
