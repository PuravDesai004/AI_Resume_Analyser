import os
from google import genai
from dotenv import load_dotenv

from rag_engine.paths import ENV_PATH

# CONFIG
EMBEDDING_MODEL = "gemini-embedding-001" # Google's current embedding model
load_dotenv(ENV_PATH)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# EMBEDDING MANAGER CLASS
class EmbeddingManager:
    def __init__(self, model=EMBEDDING_MODEL):
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
        if not api_key:
            raise ValueError("Set GEMINI_API_KEY in .env to index and search documents.")
        self.client = genai.Client(api_key=api_key, http_options={"timeout": 30000})

    def embed_chunk(self, text):
        """Embeds a single piece of text, returns one vector."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        return response.embeddings[0].values

    def embed_chunks_batch(self, chunk_list):
        """Embeds multiple chunks in a single API call."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=chunk_list,
            config={"task_type": "RETRIEVAL_DOCUMENT"},
        )
        return [embedding.values for embedding in response.embeddings]
