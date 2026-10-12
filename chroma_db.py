import chromadb
import hashlib
import json
from pathlib import Path

# Creates the './my_local_db' directory automatically if it doesn't exist
client = chromadb.PersistentClient(path=str(Path(__file__).resolve().parent / "my_local_db"))

# Safer method: Fetches 'my_documents' if it exists, or creates it if it's new
collection = client.get_or_create_collection(name="my_documents")

def add_chunks_to_collection(chunks, embeddings):
    if len(chunks) != len(embeddings):
        raise ValueError("Embedding count does not match document chunks.")
    if not chunks:
        return
    documents = [chunk.page_content for chunk in chunks]
    metadatas = []
    for chunk in chunks:
        page = chunk.metadata.get("page", None)
        metadatas.append({
            "source": chunk.metadata.get("source", "unknown"), # .get(key, default_value)
            "page": (page + 1) if page is not None else -1,
        })

    ids, occurrences = [], {}
    for text, meta in zip(documents, metadatas):
        content_key = json.dumps([text, meta], sort_keys=True)
        ordinal = occurrences.get(content_key, 0)
        occurrences[content_key] = ordinal + 1
        ids.append(hashlib.sha256(json.dumps([content_key, ordinal]).encode()).hexdigest())
    collection.upsert(
        ids = ids,
        embeddings = embeddings,
        documents = documents,
        metadatas = metadatas
    )
    print(f"Added {len(chunks)} chunks to collection '{collection}'")

if __name__ == "__main__":
    print("Collections in DB:", client.list_collections())
    print("Total documents stored:", collection.count())

    if collection.count() > 0:
        print(collection.peek(5))
    else:
        print("No documents in collection yet — nothing added so far.")
