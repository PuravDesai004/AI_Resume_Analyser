import logging
import os
from dotenv import load_dotenv

from rag_engine.paths import ENV_PATH

logger = logging.getLogger(__name__)

# CONFIG
EMBEDDING_MODEL = "gemini-embedding-001" # Google's current embedding model
load_dotenv(ENV_PATH)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# EMBEDDING MANAGER CLASS
class EmbeddingManager:
    def __init__(self, model=None):
        self.model = model or os.getenv("RAG_EMBEDDING_MODEL", EMBEDDING_MODEL)
        self._client = None
        self._local_model = None
        self._prefer_local = any(x in self.model.lower() for x in ("minilm", "sentence", "bge"))
        self._local_model_name = self.model if self._prefer_local else "all-MiniLM-L6-v2"
        self._remote_used = False

    @property
    def embedding_space(self):
        """The actual vector space, including the local fallback's model identity."""
        return f"sentence-transformers:{self._local_model_name}" if self._prefer_local else self.model

    def bind_embedding_space(self, space):
        """Pin retrieval to the space of vectors loaded from persistent storage."""
        if space != self.embedding_space:
            raise ValueError("Stored embeddings belong to a different vector space.")
        if not self._prefer_local:
            self._remote_used = True

    @property
    def client(self):
        if self._client is None and not self._prefer_local:
            api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
            if api_key:
                try:
                    from google import genai
                    self._client = genai.Client(api_key=api_key, http_options={"timeout": 30000})
                except Exception as exc:
                    logger.warning("Failed to initialize Gemini client for embeddings: %s", exc)
        return self._client

    def _get_local_model(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer
            try:
                self._local_model = SentenceTransformer(self._local_model_name)
            except Exception:
                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._local_model_name = "all-MiniLM-L6-v2"
        return self._local_model

    def embed_chunk(self, text):
        """Embeds a single piece of text, returns one vector."""
        clean = (text or "").strip()
        if not clean:
            raise ValueError("Cannot embed empty text.")

        if not self._prefer_local and self.client:
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=clean,
                    config={"task_type": "RETRIEVAL_QUERY"},
                )
                vector = response.embeddings[0].values
                self._remote_used = True
                return vector
            except Exception as exc:
                if self._remote_used:
                    # A query/batch must use the same space as previously indexed text.
                    raise RuntimeError("Gemini embeddings are unavailable; retry without changing the indexed vector space.") from exc
                logger.warning(
                    "Gemini embedding failed (%s, HTTP %s); falling back to local SentenceTransformer.",
                    type(exc).__name__, getattr(exc, "code", "n/a"),
                )

        if self._remote_used:
            raise RuntimeError("Gemini embeddings are unavailable; retry without changing the indexed vector space.")
        self._prefer_local = True

        vec = self._get_local_model().encode(clean)
        return vec.tolist()

    def embed_chunks_batch(self, chunk_list):
        """Embeds multiple chunks in a single API call."""
        if not chunk_list:
            return []
        sanitized = [c.strip() if c and c.strip() else " " for c in chunk_list]

        if not self._prefer_local and self.client:
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=sanitized,
                    config={"task_type": "RETRIEVAL_DOCUMENT"},
                )
                vectors = [embedding.values for embedding in response.embeddings]
                if len(vectors) != len(sanitized):
                    raise RuntimeError("Embedding count does not match input texts.")
                self._remote_used = True
                return vectors
            except Exception as exc:
                if self._remote_used:
                    raise RuntimeError("Gemini embeddings are unavailable; retry without changing the indexed vector space.") from exc
                logger.warning(
                    "Gemini batch embedding failed (%s, HTTP %s); falling back to local SentenceTransformer.",
                    type(exc).__name__, getattr(exc, "code", "n/a"),
                )

        if self._remote_used:
            raise RuntimeError("Gemini embeddings are unavailable; retry without changing the indexed vector space.")
        self._prefer_local = True

        vecs = self._get_local_model().encode(sanitized)
        return [v.tolist() for v in vecs]
