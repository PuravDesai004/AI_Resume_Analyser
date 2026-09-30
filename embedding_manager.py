import os
from typing import Optional
from google import genai
from dotenv import load_dotenv

# CONFIG
EMBEDDING_MODEL = "gemini-embedding-001" # Google's current embedding model
load_dotenv(override=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_DEFAULT_EMBEDDER: Optional["EmbeddingManager"] = None


def get_embedding_manager(model: str = EMBEDDING_MODEL) -> "EmbeddingManager":
    """Returns singleton instance of EmbeddingManager to prevent redundant client creation."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None or _DEFAULT_EMBEDDER.model != model:
        _DEFAULT_EMBEDDER = EmbeddingManager(model=model)
    return _DEFAULT_EMBEDDER


# EMBEDDING MANAGER CLASS
class EmbeddingManager:
    _instance_cache: dict[str, "EmbeddingManager"] = {}

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        self.client = genai.Client(api_key=api_key)

    def embed_chunk(self, text: str) -> list[float]:
        """Embeds a single piece of text, returns one vector."""
        clean = (text or "").strip()
        if not clean:
            # Return standard 3072 zero vector for empty inputs to prevent API 400 errors
            return [0.0] * 3072
        response = self.client.models.embed_content(
            model=self.model,
            contents=clean
        )
        return response.embeddings[0].values

    def embed_chunks_batch(self, chunk_list: list[str]) -> list[list[float]]:
        """Embeds multiple chunks in a single API call."""
        if not chunk_list:
            return []
        # Filter and sanitize empty chunks
        sanitized = [c.strip() if c and c.strip() else " " for c in chunk_list]
        response = self.client.models.embed_content(
            model=self.model,
            contents=sanitized
        )
        return [embedding.values for embedding in response.embeddings]